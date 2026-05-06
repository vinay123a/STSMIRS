"""
STSMIRS — Interactive Live Demo
================================
Enter tourist details, simulate an AI detection event, and watch the
Central Server automatically route it through confidence scoring →
blockchain trigger → ADM selective release.
"""

import json
import os
import sys
import time

# Ensure the script's directory is on sys.path so local modules resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force UTF-8 output on Windows so Unicode characters (✓ ✗ ─ │ ⚠) render correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from web3 import Web3

# ── Setup ──────────────────────────────────────────────────────
load_dotenv()

SEPOLIA_RPC_URL = os.getenv("SEPOLIA_RPC_URL")
PRIVATE_KEY = os.getenv("PRIVATE_KEY", "")
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")

if not SEPOLIA_RPC_URL or not PRIVATE_KEY or not CONTRACT_ADDRESS:
    print("ERROR: .env file is missing values.")
    sys.exit(1)

if not PRIVATE_KEY.startswith("0x"):
    PRIVATE_KEY = "0x" + PRIVATE_KEY

# ── Load ABI ──────────────────────────────────────────────────
ABI_PATH = os.path.join(
    os.path.dirname(__file__),
    "blockchain", "artifacts", "contracts", "STSMIRS.sol", "STSMIRS.json",
)
with open(ABI_PATH) as f:
    CONTRACT_ABI = json.load(f)["abi"]

# ── Connect ───────────────────────────────────────────────────
print("=" * 60)
print("  STSMIRS — Interactive Live Demo (Sepolia)")
print("=" * 60)

w3 = Web3(Web3.HTTPProvider(SEPOLIA_RPC_URL))
if not w3.is_connected():
    print("ERROR: Cannot connect to Sepolia. Check your internet and RPC URL.")
    sys.exit(1)

account = w3.eth.account.from_key(PRIVATE_KEY)
contract = w3.eth.contract(
    address=Web3.to_checksum_address(CONTRACT_ADDRESS),
    abi=CONTRACT_ABI,
)
balance = w3.from_wei(w3.eth.get_balance(account.address), "ether")
print(f"\n✓ Connected  |  Account: {account.address}  |  Balance: {balance:.4f} ETH\n")

from adm import ADM, KMD
from central_server import CentralServer, DetectionEvent, EventType, EVENT_TO_EMERGENCY

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
kmd = KMD(db_path=os.path.join(_SCRIPT_DIR, "live_kmd.db"))
adm = ADM(kmd=kmd)

# ── Helper ────────────────────────────────────────────────────
def send_tx(tx_func):
    base_gas = w3.eth.gas_price
    gas_price = int(base_gas * 1.2)  # 20% tip above base for faster inclusion
    tx = tx_func.build_transaction({
        "from": account.address,
        "nonce": w3.eth.get_transaction_count(account.address),
        "gas": 300000,
        "gasPrice": gas_price,
    })
    signed = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"  TX: {tx_hash.hex()}  (waiting for block...)", end="", flush=True)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=240)  # 4 min for slow blocks
    if receipt.status != 1:
        print(f"  ✗ TX FAILED: {receipt.status}")
        raise Exception(f"Transaction failed with status {receipt.status}")
    print(f"  ✓ Block #{receipt.blockNumber}")
    return receipt

# ── Step 1: Start with Existing UUID or Enrollment ──────────
print("─" * 60)
print("STEP 1 — Start: Use Existing UUID or Enroll New Tourist")
print("─" * 60)
print()

PERSISTENT_INDEX = os.path.join(os.path.dirname(__file__), ".adm_ref_index.json")
adm_ref_index = {}
if os.path.exists(PERSISTENT_INDEX):
    with open(PERSISTENT_INDEX, "r") as f:
        adm_ref_index = json.load(f)

enrolled_tourists = []
enrolled_hashes = dict(adm_ref_index)  # Map adm_ref → id_hash for verification

print("  Select mode:")
print("    1. Use an existing adm_ref")
print("    2. Enroll new tourist(s)")
print("    3. Admin Mode (View Logs & Edit Zones)")
mode = input("  Choose [1/2/3]: ").strip() or "1"

if mode == "3":
    # Ensure the CentralServer instance exists (created later in the script)
    # If not yet created, we will instantiate it now using the already‑available objects.
    try:
        server
    except NameError:
        server = CentralServer(
            contract=contract,
            web3=w3,
            account=account.address,
            adm=adm,
        )
    while True:
        print("\n--- Event Log History ---")
        if not server.processed_events:
            print("  No events recorded yet.")
        else:
            for idx, ev in enumerate(server.processed_events):
                print(f"  {idx+1}. {ev.get('event_type')} (conf: {ev.get('confidence')}) in {ev.get('zone_id')} -> {ev.get('action')}")
                
        zones = server.get_zone_status()
        print("\n--- Current zones and scores ---")
        if not zones:
            print("  No zones have been created yet.")
        else:
            for zid, info in zones.items():
                print(f"  {zid}: score={info['score']:.1f}, events={info['event_count']}")
        zid = input("\nEnter zone id to edit (or press Enter to exit): ").strip()
        if not zid:
            break
        try:
            new_score = float(input(f"Enter new score for {zid} (0‑100): ").strip())
        except ValueError:
            print("  ✗ Invalid score – must be a number.")
            continue
        server.set_zone_score(zid, new_score)
    # After editing, ask if user wants to continue to event simulation
    cont = input("\n  Continue to event simulation? (y/n): ").strip().lower()
    if cont != "y":
        print("\n✓ Admin session ended.")
        kmd.close()
        sys.exit(0)
    mode = "1"

# ── Identity Selection: UUID (AI Camera) or Zone (IoT) ───────
id_mode = None
adm_ref = None
id_hash = None
zone_id_selected = "ZONE_A"

if mode == "1":
    print("\n  How will this event identify the subject?")
    print("    1. UUID (AI Camera — specific person)")
    print("    2. Zone-wide (IoT Sensor — no specific person)")
    id_mode = input("  Choose [1/2]: ").strip() or "1"

    if id_mode == "1":
        adm_ref = input("\n  Enter adm_ref (UUID): ").strip()
        if adm_ref.startswith("adm_ref:"):
            adm_ref = adm_ref.replace("adm_ref:", "").strip()

        if adm_ref in adm_ref_index:
            id_hash = bytes.fromhex(adm_ref_index[adm_ref])
        else:
            print("  ✗ adm_ref not found in saved enrollments. Exiting.")
            kmd.close()
            sys.exit(1)

        print(f"\n  ✓ Using existing identity: {adm_ref}")
    else:
        zone_id_selected = input("\n  Enter Zone ID (default ZONE_A): ").strip() or "ZONE_A"
        print(f"\n  ✓ Zone-wide event selected: {zone_id_selected}")
        print("    No specific tourist identity — IoT sensor mode.")
else:
    while True:
        print(f"Enrolling tourist #{len(enrolled_tourists) + 1}")
        name             = input("  Full name: ").strip()
        if not name:
            break  # Allow empty name to stop enrolling
        phone            = input("  Phone number: ").strip()
        emergency        = input("  Emergency contact number: ").strip()
        medical_history  = input("  Medical history: ").strip()
        police_records   = input("  Past police records: ").strip()
        nationality      = input("  Nationality: ").strip()
        passport         = input("  Passport number: ").strip()
        hotel            = input("  Hotel / accommodation: ").strip()

        tourist_identity = {
            "name": name,
            "face": "face_encoding_placeholder",
            "phone_number": phone,
            "emergency_contact": emergency,
            "medical_history": medical_history,
            "past_police_records": police_records,
            "nationality": nationality,
            "passport_number": passport,
            "hotel": hotel,
        }

        print(f"\n  Tourist profile built:")
        for k, v in tourist_identity.items():
            print(f"    {k}: {v}")

        adm_ref, id_hash = adm.encrypt_identity(tourist_identity)
        print(f"\n  ✓ Encrypted (AES-256-GCM + RSA-4096)")
        print(f"    adm_ref: {adm_ref}")
        print(f"    id_hash: 0x{id_hash.hex()}")

        owner_pub_key = adm.get_public_key_pem().decode()[:64] + "..."
        print(f"\n  Enrolling on Sepolia...")
        receipt = send_tx(contract.functions.enrollIdentity(id_hash, adm_ref, owner_pub_key))

        on_chain = contract.functions.getIdentity(id_hash).call()
        print(f"  ✓ Verified on-chain. admRef: {on_chain[1][:16]}...")

        enrolled_tourists.append({
            "name": name,
            "id_hash": id_hash,
            "adm_ref": adm_ref,
            "identity": tourist_identity,
        })
        enrolled_hashes[adm_ref] = id_hash
        adm_ref_index[adm_ref] = id_hash.hex()

        print(f"\n  Enrolled {len(enrolled_tourists)} tourist(s) so far.")
        another = input("\n  Enroll another tourist? (y/n): ").strip().lower()
        if another != 'y':
            break
        print()

    if not enrolled_tourists:
        print("No tourists enrolled. Exiting.")
        kmd.close()
        sys.exit(0)

    print("\n  How will this event identify the subject?")
    print("    1. UUID (AI Camera — specific person)")
    print("    2. Zone-wide (IoT Sensor — no specific person)")
    id_mode = input("  Choose [1/2]: ").strip() or "1"

    if id_mode == "1":
        print("\n  Enrolled tourists (UUID references):")
        for ar in enrolled_hashes.keys():
            print(f"  • {ar}")

        adm_ref = input("\n  Select tourist by UUID: ").strip()
        if adm_ref not in enrolled_hashes:
            print("  ✗ ERROR: adm_ref not found in current enrollments.")
            kmd.close()
            sys.exit(1)

        print(f"\n  ✓ Identity selected: {adm_ref}")
    else:
        zone_id_selected = input("\n  Enter Zone ID (default ZONE_A): ").strip() or "ZONE_A"
        print(f"\n  ✓ Zone-wide event selected: {zone_id_selected}")
        print("    No specific tourist identity — IoT sensor mode.")

# ── Resolve identity (UUID mode only) ────────────────────────
if id_mode == "1":
    print("\n  ✓ Identity selected (no personal info revealed).")
    print(f"    Processing detection under this encrypted identity...")

    # Retrieve the id_hash for this adm_ref
    id_hash_value = enrolled_hashes.get(adm_ref)
    if isinstance(id_hash_value, bytes):
        id_hash = id_hash_value
    else:
        id_hash = bytes.fromhex(id_hash_value) if id_hash_value else None

    if not id_hash:
        print("  ✗ ERROR: Could not find id_hash for selected identity.")
        kmd.close()
        sys.exit(1)

    # Verify the selected identity is enrolled on-chain before emergency access.
    try:
        on_chain_identity = contract.functions.getIdentity('0x' + id_hash.hex()).call()
        print(f"  ✓ Verified enrolled identity on-chain. admRef: {on_chain_identity[1][:16]}...")
    except Exception as err:
        print("  ✗ ERROR: selected id_hash is not enrolled on-chain.")
        print(f"    Details: {err}")
        print("    The emergency access request would fail with 'identity not found'.")
        kmd.close()
        sys.exit(1)
else:
    print(f"\n  ✓ IoT mode — zone-wide event for {zone_id_selected}")
    print("    Skipping identity verification (no specific person).")


# ── Step 3: Simulate AI Detection ────────────────────────────
print("\n" + "─" * 60)
print("STEP 3 — Simulate AI/Sensor Detection Event")
print("─" * 60)
if id_mode == "1":
    print("""
  What did the AI camera detect?

  1. FALL               → mapped to MEDICAL_PROBLEM
  2. PANIC              → mapped to MEDICAL_EMERGENCY
  3. AGGRESSION         → mapped to VIOLENT_FIGHT
""")
    det_choice = input("  What did the AI detect? [1-3]: ").strip() or "1"
    det_map = {
        "1": EventType.FALL,
        "2": EventType.PANIC,
        "3": EventType.AGGRESSION,
    }
    source = "AI_CAMERA"
else:
    print("""
  What did the IoT sensor detect?

  1. SURGE              → mapped to MEDICAL_EMERGENCY
  2. CROWD_DENSITY      → mapped to SMALL_FIGHT
""")
    det_choice = input("  What did the IoT detect? [1-2]: ").strip() or "1"
    det_map = {
        "1": EventType.SURGE,
        "2": EventType.CROWD_DENSITY,
    }
    source = "IOT_SENSOR"

detected_event_type = det_map.get(det_choice, list(det_map.values())[0])

conf_input = input("  Confidence score (0.0 - 1.0): ").strip() or "0.95"
confidence = float(conf_input)

zone_id = zone_id_selected if id_mode == "2" else (input("  Zone ID (e.g. ZONE_A): ").strip() or "ZONE_A")

emergency_type = EVENT_TO_EMERGENCY.get(detected_event_type, "MEDICAL_PROBLEM")

print(f"\n  ┌─────────────────────────────────────────┐")
print(f"  │  Detection: {detected_event_type.value:<28} │")
print(f"  │  Confidence: {confidence:<27} │")
print(f"  │  Zone: {zone_id:<32} │")
print(f"  │  Source: {source:<30} │")
print(f"  │  → Emergency Type: {emergency_type:<19} │")
print(f"  └─────────────────────────────────────────┘")

# ── Step 4: Central Server Processes the Event ───────────────
print("\n" + "─" * 60)
print("STEP 4 — Central Server: Confidence Routing + Score Tracking")
print("─" * 60)

server = CentralServer(
    contract=contract,
    web3=w3,
    account=account.address,
    adm=adm,
)

# Confidence routing demo
if confidence < 0.70:
    print(f"\n  ✗ DISCARDED — confidence {confidence:.2f} < 0.70 (too low)")
    print("  The AI is not confident enough. No action taken.")
    kmd.close()
    sys.exit(0)
elif confidence < 0.90:
    print(f"\n  ⚠ QUEUED FOR HUMAN REVIEW — confidence {confidence:.2f} (medium)")
    print("  An officer must review this detection before it can proceed.")
    review = input("  Officer decision — approve or reject? (a/r): ").strip().lower()
    if review != "a":
        print("\n  ✗ Officer REJECTED the event. No action taken.")
        kmd.close()
        sys.exit(0)
    print("  ✓ Officer APPROVED the event.")

print(f"\n  Processing detection event...")

event = DetectionEvent(
    event_type=detected_event_type,
    confidence=confidence,
    source=source,
    zone_id=zone_id,
    tourist_id=adm_ref if id_mode == "1" else None,
)
result = server.ingest_event(event)

if result.get("action") == "EMERGENCY_TRIGGERED":
    print(f"\n  ⚠  EMERGENCY TRIGGERED!")
    print(f"     Zone {zone_id} score updated to: {server.zones[zone_id].score:.1f}")

# ── Steps 5–7 only apply to UUID mode (specific person identified) ──
if id_mode == "1":
    # ── Step 5: Blockchain Emergency Access ──────────────────────
    print("\n" + "─" * 60)
    print(f"STEP 5 — Requesting Emergency Access On-Chain ({emergency_type})")
    print("─" * 60)

    score_int = int(server.zones[zone_id].score)
    print(f"\n  Emergency type: {emergency_type}")
    print(f"  Zone score:     {score_int}")

    receipt = send_tx(
        contract.functions.requestEmergencyAccess(id_hash, emergency_type, score_int)
    )

    logs = contract.events.EmergencyAccessGranted().process_receipt(receipt)
    if logs:
        print(f"  ✓ EmergencyAccessGranted event emitted on-chain")
        print(f"    Type: {logs[0].args.emergencyType}  |  Score: {logs[0].args.score}")

    # ── Step 6: ADM Blockchain Authorization Check + Selective Release
    print("\n" + "─" * 60)
    print(f"STEP 6 — ADM Blockchain Authorization Check + Selective Release")
    print("─" * 60)

    # First: Query smart contract to discover what emergency type was authorized
    # Pass the Step-5 receipt so the ADM can decode the event directly
    # (no separate eth_getLogs call needed — avoids Alchemy 400 errors)
    print(f"\n  Checking blockchain authorization...")
    authorized_emergency_type = adm.verify_authorization(
        contract, id_hash, adm_ref, web3_instance=w3, receipt=receipt
    )

    if not authorized_emergency_type:
        print(f"\n  ✗ AUTHORIZATION DENIED by smart contract")
        print(f"  The ADM will NOT decrypt without blockchain proof.")
        print(f"  This is the privacy gatekeeper working: no record access without")
        print(f"  explicit contract authorization.")
        kmd.close()
        sys.exit(0)

    print(f"\n  ✓ Blockchain authorization verified!")
    print(f"  Authorized emergency type: {authorized_emergency_type}")
    print(f"  (ADM determines fields to release based on contract decision)")

    # Second: Release only the minimum required fields for the AUTHORIZED type
    released = adm.selective_release(adm_ref, authorized_emergency_type)
    if released:
        print(f"\n  ✓ RELEASED to emergency responder:")
        for k, v in released.items():
            print(f"    ✓ {k}: {v}")

        # Note: We can't show what was withheld because we never decrypted full identity
        print(f"\n  ✗ WITHHELD (privacy protected):")
        print(f"    ✗ Only minimum required fields for {authorized_emergency_type} released")

    # ── Step 7: Confirm Release (Audit Trail) ────────────────────
    print("\n" + "─" * 60)
    print("STEP 7 — Confirming Release On-Chain (Immutable Audit Trail)")
    print("─" * 60)

    receipt = send_tx(contract.functions.confirmRelease(id_hash, account.address))
    logs = contract.events.ReleaseConfirmed().process_receipt(receipt)
    if logs:
        print(f"  ✓ ReleaseConfirmed — immutable proof of data access recorded")

else:
    # ── IoT Zone-Wide Alert Summary ──────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 5 — IoT Zone Alert Summary")
    print("─" * 60)
    print(f"\n  Zone-wide {detected_event_type.value} detected in {zone_id}")
    print(f"  Emergency type: {emergency_type}")
    print(f"  Zone safety score: {server.zones[zone_id].score:.1f}")
    print(f"\n  ℹ No individual identity involved — blockchain identity")
    print(f"    access and selective release are not applicable.")
    print(f"    Zone alert logged to central server for responder dispatch.")

kmd.close()

with open(PERSISTENT_INDEX, "w") as f:
    json.dump(adm_ref_index, f, indent=2)

print("✓ Done.\n")

