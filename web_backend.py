"""
STSMIRS — Web Backend
======================
Mirrors the complete interactive_demo.py flow as REST API endpoints.

Steps mirrored:
  1. Connection info
  2. Enrollment (all fields: name, phone, emergency, medical, police, nationality, passport, hotel)
  3. Use existing adm_ref (with on-chain verification)
  4. IoT Zone-wide mode
  5. Detection event simulation (AI or IoT)
  6. Confidence routing (discard / human review / auto-process)
  7. Central server zone score update
  8. Step 5: requestEmergencyAccess on-chain (UUID mode only)
  9. Step 6: ADM verify authorization + selective release (UUID mode only)
  10. Step 7: confirmRelease audit trail (UUID mode only)
  11. IoT zone alert summary
  12. Admin mode: view logs, edit zone scores
"""

import json
import os
import sys
import threading
import time
import logging
from typing import Optional

from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from web3 import Web3

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adm import ADM, KMD
from central_server import CentralServer, DetectionEvent, EventType, EVENT_TO_EMERGENCY

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Flask Setup ───────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# ── Load Environment ──────────────────────────────────────────────
load_dotenv()
SEPOLIA_RPC_URL = os.getenv("SEPOLIA_RPC_URL")
PRIVATE_KEY = os.getenv("PRIVATE_KEY", "")
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")

if not PRIVATE_KEY.startswith("0x"):
    PRIVATE_KEY = "0x" + PRIVATE_KEY

# ── Load Contract ABI ─────────────────────────────────────────────
ABI_PATH = os.path.join(
    os.path.dirname(__file__),
    "blockchain", "artifacts", "contracts", "STSMIRS.sol", "STSMIRS.json",
)
try:
    with open(ABI_PATH, "r") as f:
        CONTRACT_ABI = json.load(f)["abi"]
except Exception as e:
    logger.error(f"Failed to load ABI: {e}")
    CONTRACT_ABI = []

# ── Connect Web3 ──────────────────────────────────────────────────
w3 = Web3(Web3.HTTPProvider(SEPOLIA_RPC_URL))
account = None
contract = None
connection_info = {}

if w3.is_connected():
    account = w3.eth.account.from_key(PRIVATE_KEY)
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(CONTRACT_ADDRESS),
        abi=CONTRACT_ABI,
    )
    balance = float(w3.from_wei(w3.eth.get_balance(account.address), "ether"))
    connection_info = {
        "connected": True,
        "account": account.address,
        "balance": round(balance, 4),
        "contract": CONTRACT_ADDRESS,
    }
    logger.info(f"✓ Connected | Account: {account.address} | Balance: {balance:.4f} ETH")
else:
    connection_info = {"connected": False}
    logger.error("Failed to connect to Sepolia")

# ── Initialize Subsystems ─────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
kmd = KMD(db_path=os.path.join(SCRIPT_DIR, "live_kmd.db"))
adm = ADM(kmd=kmd)

# Pass contract=None — web backend handles blockchain flow explicitly (Steps 5-7)
# CentralServer with contract=None only does score tracking.
server = CentralServer(contract=None, web3=w3, account=account.address if account else None, adm=adm)

# ── Persistent adm_ref index (mirrors .adm_ref_index.json) ───────
PERSISTENT_INDEX = os.path.join(SCRIPT_DIR, ".adm_ref_index.json")
adm_ref_index: dict = {}   # adm_ref -> id_hash (hex string)
if os.path.exists(PERSISTENT_INDEX):
    with open(PERSISTENT_INDEX, "r") as f:
        adm_ref_index = json.load(f)

def save_adm_ref_index():
    with open(PERSISTENT_INDEX, "w") as f:
        json.dump(adm_ref_index, f, indent=2)

# ── TX Helper (mirrors send_tx in demo) ───────────────────────────
def send_tx(tx_func):
    if not account or not w3:
        raise Exception("Web3 not connected")
    base_gas = w3.eth.gas_price
    gas_price = int(base_gas * 1.2)
    tx = tx_func.build_transaction({
        "from": account.address,
        "nonce": w3.eth.get_transaction_count(account.address),
        "gas": 300000,
        "gasPrice": gas_price,
    })
    signed = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=240)
    if receipt.status != 1:
        raise Exception(f"Transaction failed with status {receipt.status}")
    return receipt

# ── API: Status / Connection Info ─────────────────────────────────

@app.route("/api/status", methods=["GET"])
def get_status():
    """Connection info: account, balance, contract address."""
    return jsonify(connection_info)

# ── API: Zone Scores ──────────────────────────────────────────────

@app.route("/api/zones", methods=["GET"])
def get_zones():
    return jsonify(server.get_zone_status())

# IMPORTANT: /api/zones/reset must be defined BEFORE /api/zones/<zone_id>
# otherwise Flask matches 'reset' as a zone_id
@app.route("/api/zones/reset", methods=["POST", "OPTIONS"])
def reset_zones():
    """Reset all zone scores back to 100 (fresh start)."""
    if request.method == "OPTIONS":
        return jsonify({"ok": True})
    for zone_id in list(server.zones.keys()):
        server.set_zone_score(zone_id, 100.0)
    return jsonify({"ok": True, "message": "All zones reset to 100"})

@app.route("/api/zones/<zone_id>", methods=["POST", "OPTIONS"])
def set_zone(zone_id):
    """Admin: manually set a zone's safety score."""
    if request.method == "OPTIONS":
        return jsonify({"ok": True})
    data = request.json
    new_score = float(data.get("score", 100))
    server.set_zone_score(zone_id, new_score)
    return jsonify({"ok": True, "zone_id": zone_id, "score": new_score})

# ── API: Event Log ────────────────────────────────────────────────

@app.route("/api/events", methods=["GET"])
def get_events():
    return jsonify({
        "processed": server.processed_events,
        "queue": [
            {
                "tourist_id": e.tourist_id,
                "event_type": e.event_type.value,
                "confidence": e.confidence,
                "zone_id": e.zone_id,
                "source": e.source,
            }
            for e in server.review_queue
        ],
    })

# ── API: Enrollment ───────────────────────────────────────────────

@app.route("/api/tourists", methods=["GET"])
def list_tourists():
    """Returns all enrolled adm_refs so UI can show them for selection."""
    return jsonify({"tourists": list(adm_ref_index.keys())})

@app.route("/api/enroll", methods=["POST"])
def enroll_tourist():
    """
    Mirrors demo Step 2 (enroll mode):
    Fields: name, phone_number, emergency_contact, medical_history,
            past_police_records, nationality, passport_number, hotel
    """
    data = request.json
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "name is required"}), 400

    tourist_identity = {
        "name": name,
        "face": "face_encoding_placeholder",
        "phone_number": data.get("phone_number", ""),
        "emergency_contact": data.get("emergency_contact", ""),
        "medical_history": data.get("medical_history", ""),
        "past_police_records": data.get("past_police_records", ""),
        "nationality": data.get("nationality", ""),
        "passport_number": data.get("passport_number", ""),
        "hotel": data.get("hotel", ""),
    }

    try:
        # Encrypt identity (AES-256-GCM + RSA-4096)
        adm_ref, id_hash = adm.encrypt_identity(tourist_identity)

        # Enroll on-chain
        owner_pub_key = adm.get_public_key_pem().decode()[:64] + "..."
        receipt = send_tx(contract.functions.enrollIdentity(id_hash, adm_ref, owner_pub_key))

        # Verify on-chain
        on_chain = contract.functions.getIdentity(id_hash).call()

        # Save to persistent index
        adm_ref_index[adm_ref] = id_hash.hex()
        save_adm_ref_index()

        return jsonify({
            "success": True,
            "adm_ref": adm_ref,
            "id_hash": "0x" + id_hash.hex(),
            "on_chain_adm_ref_preview": on_chain[1][:16] + "...",
            "tx_hash": "0x" + receipt.transactionHash.hex(),
            "block": receipt.blockNumber,
            "identity_profile": tourist_identity,
        })
    except Exception as e:
        logger.error(f"Enrollment failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/verify-uuid", methods=["GET", "POST"])
def verify_uuid():
    """
    Step 1 (use existing): verify adm_ref is in the index and enrolled on-chain.
    Supports both GET (?adm_ref=...) and POST {adm_ref: ...}
    """
    if request.method == "GET":
        adm_ref = request.args.get("adm_ref", "").strip().replace("adm_ref:", "").strip()
    else:
        data = request.json or {}
        adm_ref = data.get("adm_ref", "").strip().replace("adm_ref:", "").strip()

    if adm_ref not in adm_ref_index:
        return jsonify({"error": "adm_ref not found in saved enrollments"}), 404

    id_hash_hex = adm_ref_index[adm_ref]
    id_hash = bytes.fromhex(id_hash_hex)

    try:
        on_chain = contract.functions.getIdentity(id_hash).call()
        return jsonify({
            "ok": True,
            "adm_ref": adm_ref,
            "id_hash": "0x" + id_hash_hex,
            "on_chain_adm_ref_preview": on_chain[1][:16] + "...",
        })
    except Exception as e:
        return jsonify({"error": f"Not enrolled on-chain: {e}"}), 400

# ── API: Detection (Steps 3-7) ────────────────────────────────────

@app.route("/api/detect", methods=["POST"])
def detect():
    """
    Full flow mirror of Steps 3-7 in interactive_demo.py.

    Body:
      source: "AI_CAMERA" | "IOT_SENSOR"
      event_type: "FALL" | "PANIC" | "AGGRESSION" | "SURGE" | "CROWD_DENSITY"
      confidence: float (0.0 - 1.0)
      zone_id: string
      tourist_id: string (adm_ref, required for AI_CAMERA)
    """
    data = request.json
    source = data.get("source", "AI_CAMERA")
    zone_id = data.get("zone_id", "A")
    confidence = float(data.get("confidence", 0.95))
    tourist_id = data.get("tourist_id")

    # Strip adm_ref: prefix
    if tourist_id:
        tourist_id = tourist_id.replace("adm_ref:", "").strip()

    try:
        event_type = EventType(data.get("event_type"))
    except ValueError:
        return jsonify({"error": "Invalid event_type"}), 400

    emergency_type = EVENT_TO_EMERGENCY.get(event_type, "MEDICAL_PROBLEM")

    result = {
        "source": source,
        "event_type": event_type.value,
        "confidence": confidence,
        "zone_id": zone_id,
        "tourist_id": tourist_id,
        "emergency_type": emergency_type,
        "steps": [],
    }

    # ── Step 3 summary ────────────────────────────────────────────
    result["steps"].append({
        "step": 3,
        "title": "AI/Sensor Detection Event",
        "detection": event_type.value,
        "source": source,
        "confidence": confidence,
        "zone": zone_id,
        "emergency_type": emergency_type,
    })

    # ── Step 4: Confidence Routing ────────────────────────────────
    if confidence < 0.70:
        result["steps"].append({
            "step": 4,
            "title": "Central Server: Confidence Routing",
            "action": "DISCARDED",
            "reason": f"Confidence {confidence:.2f} < 0.70 — too low. No action taken.",
        })
        result["final_action"] = "DISCARDED"
        return jsonify(result)

    if confidence < 0.90:
        # Queue for human review — do NOT process yet
        event = DetectionEvent(
            event_type=event_type,
            confidence=confidence,
            source=source,
            zone_id=zone_id,
            tourist_id=tourist_id if source == "AI_CAMERA" else None,
        )
        server.review_queue.append(event)
        result["steps"].append({
            "step": 4,
            "title": "Central Server: Confidence Routing",
            "action": "QUEUED_FOR_HUMAN_REVIEW",
            "reason": f"Confidence {confidence:.2f} is medium (0.70-0.89). Awaiting officer decision.",
        })
        result["final_action"] = "QUEUED_FOR_HUMAN_REVIEW"
        return jsonify(result)

    # High confidence — process immediately
    return _process_and_run_blockchain(event_type, confidence, source, zone_id, tourist_id, result)

@app.route("/api/review/approve", methods=["POST"])
def approve():
    data = request.json
    index = data.get("index", 0)
    if not server.review_queue or index >= len(server.review_queue):
        return jsonify({"error": "invalid index or empty queue"}), 400

    event = server.review_queue[index]
    result = {
        "source": event.source,
        "event_type": event.event_type.value,
        "confidence": event.confidence,
        "zone_id": event.zone_id,
        "tourist_id": event.tourist_id,
        "emergency_type": EVENT_TO_EMERGENCY.get(event.event_type, "MEDICAL_PROBLEM"),
        "steps": [{"step": 4, "title": "Central Server: Confidence Routing", "action": "OFFICER_APPROVED",
                   "reason": "Officer manually approved the queued event."}],
    }

    return _process_and_run_blockchain(
        event.event_type, event.confidence, event.source, event.zone_id,
        event.tourist_id.replace("ZONE:", "").replace(event.zone_id, "").strip() or event.tourist_id,
        result, queue_index=index
    )

@app.route("/api/review/reject", methods=["POST"])
def reject():
    data = request.json
    index = data.get("index", 0)
    if not server.review_queue or index >= len(server.review_queue):
        return jsonify({"error": "invalid index or empty queue"}), 400
    
    event = server.review_queue.pop(index)
    
    # Save the rejected event to the history so it actually gets cleared and logged
    event_entry = {
        "event_type": event.event_type.value,
        "confidence": event.confidence,
        "zone_id": event.zone_id,
        "source": event.source,
        "action": "HUMAN_REJECTED",
        "steps": [{
            "step": 4, 
            "title": "Central Server: Officer Review", 
            "action": "REJECTED",
            "reason": "Officer deemed this a false positive."
        }],
    }
    server.processed_events.append(event_entry)
    server._save_events()
    
    return jsonify({"action": "REJECTED", "event_type": event.event_type.value})

# ── Core processing logic ─────────────────────────────────────────

def _process_and_run_blockchain(event_type, confidence, source, zone_id, tourist_id, result, queue_index=None):
    """
    Mirrors Steps 4-7 of interactive_demo.py:
      4. Central Server score update
      5. requestEmergencyAccess (UUID mode only)
      6. ADM verify + selective release (UUID mode only)
      7. confirmRelease (UUID mode only)
      OR IoT zone alert summary
    """
    # Pop from queue if this came from approve
    if queue_index is not None:
        if server.review_queue and queue_index < len(server.review_queue):
            server.review_queue.pop(queue_index)

    # ── Step 4: Process in Central Server ────────────────────────
    event = DetectionEvent(
        event_type=event_type,
        confidence=confidence,
        source=source,
        zone_id=zone_id,
        tourist_id=tourist_id if source == "AI_CAMERA" and tourist_id else None,
    )
    ingest_result = server.ingest_event(event)

    penalty = server.SCORE_PENALTIES.get(event_type, 15.0) * confidence
    zone_score = server.zones[zone_id].score if zone_id in server.zones else 100.0

    result["steps"].append({
        "step": 4,
        "title": "Central Server: Confidence Routing + Score Tracking",
        "action": "EMERGENCY_TRIGGERED",
        "penalty_applied": round(penalty, 2),
        "zone_score_after": round(zone_score, 1),
        "event_count": server.zones[zone_id].event_count if zone_id in server.zones else 1,
    })
    result["zone_score"] = round(zone_score, 1)

    # ── Steps 5-7 (UUID / AI Camera) or IoT summary ──────────────
    is_uuid_mode = (source == "AI_CAMERA" and tourist_id and tourist_id in adm_ref_index)

    if is_uuid_mode:
        id_hash_hex = adm_ref_index[tourist_id]
        id_hash = bytes.fromhex(id_hash_hex)
        emergency_type = EVENT_TO_EMERGENCY.get(event_type, "MEDICAL_PROBLEM")
        score_int = int(zone_score)

        # Step 5: requestEmergencyAccess
        step5 = {
            "step": 5,
            "title": f"Requesting Emergency Access On-Chain ({emergency_type})",
            "emergency_type": emergency_type,
            "zone_score": score_int,
        }
        try:
            receipt5 = send_tx(contract.functions.requestEmergencyAccess(id_hash, emergency_type, score_int))
            step5["tx_hash"] = "0x" + receipt5.transactionHash.hex()
            step5["block"] = receipt5.blockNumber

            logs = contract.events.EmergencyAccessGranted().process_receipt(receipt5)
            if logs:
                step5["event_emitted"] = {
                    "type": logs[0].args.emergencyType,
                    "score": logs[0].args.score,
                }
        except Exception as e:
            step5["error"] = str(e)
            result["steps"].append(step5)
            result["final_action"] = "ERROR"
            return jsonify(result)

        result["steps"].append(step5)

        # Step 6: ADM Authorization + Selective Release
        step6 = {
            "step": 6,
            "title": "ADM Blockchain Authorization Check + Selective Release",
        }
        try:
            authorized_type = adm.verify_authorization(
                contract, id_hash, tourist_id, web3_instance=w3, receipt=receipt5
            )
            if not authorized_type:
                step6["authorized"] = False
                step6["message"] = "AUTHORIZATION DENIED by smart contract. ADM will NOT decrypt."
                result["steps"].append(step6)
                result["final_action"] = "AUTHORIZATION_DENIED"
                return jsonify(result)

            step6["authorized"] = True
            step6["authorized_type"] = authorized_type

            released = adm.selective_release(tourist_id, authorized_type)
            step6["released_fields"] = released if released else {}
            step6["withheld_message"] = f"Only minimum required fields for {authorized_type} released. All else withheld."

        except Exception as e:
            step6["error"] = str(e)
            result["steps"].append(step6)
            result["final_action"] = "ERROR"
            return jsonify(result)

        result["steps"].append(step6)

        # Step 7: Confirm Release (Audit Trail)
        step7 = {
            "step": 7,
            "title": "Confirming Release On-Chain (Immutable Audit Trail)",
        }
        try:
            receipt7 = send_tx(contract.functions.confirmRelease(id_hash, account.address))
            step7["tx_hash"] = "0x" + receipt7.transactionHash.hex()
            step7["block"] = receipt7.blockNumber
            logs7 = contract.events.ReleaseConfirmed().process_receipt(receipt7)
            step7["confirmed"] = bool(logs7)
            step7["message"] = "ReleaseConfirmed — immutable proof of data access recorded on-chain."
        except Exception as e:
            step7["error"] = str(e)

        result["steps"].append(step7)
        result["final_action"] = "COMPLETE"

    else:
        # IoT Zone Alert Summary (no identity, no blockchain calls)
        result["steps"].append({
            "step": 5,
            "title": "IoT Zone Alert Summary",
            "zone": zone_id,
            "event_type": event_type.value,
            "emergency_type": EVENT_TO_EMERGENCY.get(event_type, "MEDICAL_PROBLEM"),
            "zone_score": round(zone_score, 1),
            "message": "No individual identity involved. Blockchain identity access and selective release are not applicable. Zone alert logged to central server for responder dispatch.",
        })
        result["final_action"] = "IOT_ZONE_ALERT"

    # Save processed events list
    event_entry = {
        "event_type": event_type.value,
        "confidence": confidence,
        "zone_id": zone_id,
        "source": source,
        "action": result["final_action"],
        "steps": result["steps"],
    }
    server.processed_events.append(event_entry)
    server._save_events()

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
