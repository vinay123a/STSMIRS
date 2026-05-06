"""
STSMIRS — End-to-End Demo
=========================
Demonstrates the full flow on Sepolia testnet:
  1. Connect to Sepolia via Alchemy
  2. Load the deployed STSMIRS contract
  3. Enroll a test tourist (encrypt → KMD → on-chain)
  4. Simulate detection events with new emergency types
  5. Central server triggers emergency access on-chain
  6. ADM decrypts and selectively releases minimum fields
  7. Multi-event union demo
  8. Confirm release on-chain (audit trail)

Prerequisites:
  - Deploy the contract first:  cd blockchain && npm run deploy:sepolia
  - Fill in .env with CONTRACT_ADDRESS, SEPOLIA_RPC_URL, PRIVATE_KEY
"""

import json
import os
import sys
import time

from dotenv import load_dotenv
from web3 import Web3

# ── Setup ──────────────────────────────────────────────────────

load_dotenv()

SEPOLIA_RPC_URL = os.getenv("SEPOLIA_RPC_URL")
PRIVATE_KEY = os.getenv("PRIVATE_KEY", "")
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")

if not SEPOLIA_RPC_URL or not PRIVATE_KEY or not CONTRACT_ADDRESS:
    print("ERROR: Missing .env values. Create a .env file with:")
    print("  SEPOLIA_RPC_URL=https://eth-sepolia.g.alchemy.com/v2/YOUR_KEY")
    print("  PRIVATE_KEY=your_private_key_without_0x")
    print("  CONTRACT_ADDRESS=0x...")
    sys.exit(1)

# Ensure 0x prefix on private key
if not PRIVATE_KEY.startswith("0x"):
    PRIVATE_KEY = "0x" + PRIVATE_KEY

# ── Load Contract ABI ─────────────────────────────────────────

ABI_PATH = os.path.join(
    os.path.dirname(__file__),
    "blockchain", "artifacts", "contracts", "STSMIRS.sol", "STSMIRS.json",
)

if not os.path.exists(ABI_PATH):
    print(f"ERROR: ABI file not found at {ABI_PATH}")
    print("Run 'cd blockchain && npx hardhat compile' first.")
    sys.exit(1)

with open(ABI_PATH, "r") as f:
    artifact = json.load(f)
    CONTRACT_ABI = artifact["abi"]

# ── Connect to Sepolia ────────────────────────────────────────

print("=" * 60)
print("  STSMIRS — End-to-End Demo (Sepolia Testnet)")
print("=" * 60)

w3 = Web3(Web3.HTTPProvider(SEPOLIA_RPC_URL))
if not w3.is_connected():
    print("ERROR: Cannot connect to Sepolia. Check your RPC URL.")
    sys.exit(1)

account = w3.eth.account.from_key(PRIVATE_KEY)
print(f"\n✓ Connected to Sepolia")
print(f"  Account:  {account.address}")
balance = w3.eth.get_balance(account.address)
print(f"  Balance:  {w3.from_wei(balance, 'ether')} ETH")

contract = w3.eth.contract(
    address=Web3.to_checksum_address(CONTRACT_ADDRESS),
    abi=CONTRACT_ABI,
)
print(f"  Contract: {CONTRACT_ADDRESS}")

# ── Helpers ────────────────────────────────────────────────────

def send_tx(tx_func):
    """Build, sign, and send a transaction. Returns receipt."""
    tx = tx_func.build_transaction({
        "from": account.address,
        "nonce": w3.eth.get_transaction_count(account.address),
        "gas": 300000,
        "gasPrice": w3.eth.gas_price,
    })
    signed = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"  TX sent: {tx_hash.hex()} — waiting for confirmation...")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    print(f"  ✓ Confirmed in block {receipt.blockNumber}")
    return receipt


# ── Import project modules ────────────────────────────────────

from adm import ADM, KMD
from central_server import CentralServer, DetectionEvent, EventType

# ── Step 1: Initialize subsystems ─────────────────────────────

print("\n" + "─" * 60)
print("STEP 1 — Initialize ADM + KMD")
print("─" * 60)

kmd = KMD(db_path="demo_kmd.db")
adm = ADM(kmd=kmd)
print("✓ KMD database initialized (demo_kmd.db)")
print("✓ ADM master key pair generated (RSA-4096)")

# ── Step 2: Enroll a test tourist ─────────────────────────────

print("\n" + "─" * 60)
print("STEP 2 — Enroll test tourist")
print("─" * 60)

tourist_identity = {
    "name": "Priya Sharma",
    "face": "face_encoding_base64_data_here",
    "phone_number": "+91-9876543210",
    "emergency_contact": "+91-9123456789",
    "medical_history": "No known allergies",
    "past_police_records": "None",
    "nationality": "Indian",
    "passport_number": "K1234567",
    "hotel": "Taj Mahal Palace, Mumbai",
}

print(f"  Tourist: {tourist_identity['name']}")
print(f"  Fields:  {list(tourist_identity.keys())}")

# Encrypt and store in KMD
adm_ref, id_hash = adm.encrypt_identity(tourist_identity)
print(f"\n  ✓ Identity encrypted (AES-256-GCM + RSA-OAEP)")
print(f"    adm_ref: {adm_ref}")
print(f"    id_hash: 0x{id_hash.hex()}")

# Enroll on-chain
owner_pub_key = adm.get_public_key_pem().decode()[:64] + "..."  # truncated for storage
print(f"\n  Enrolling on Sepolia blockchain...")
receipt = send_tx(
    contract.functions.enrollIdentity(id_hash, adm_ref, owner_pub_key)
)

# Check on-chain record
on_chain = contract.functions.getIdentity(id_hash).call()
print(f"  ✓ On-chain record verified — admRef: {on_chain[1]}")

# ── Step 3: Simulate a FALL detection (MEDICAL_PROBLEM) ──────

print("\n" + "─" * 60)
print("STEP 3 — Simulate FALL detection → MEDICAL_PROBLEM emergency")
print("─" * 60)

server = CentralServer(
    contract=contract,
    web3=w3,
    account=account.address,
    adm=adm,
)

import hashlib
tourist_id = "TOURIST_001"

print(f"  Ingesting FALL events to trigger emergency threshold...")

for i in range(4):
    result = server.ingest_event(DetectionEvent(
        tourist_id=tourist_id,
        event_type=EventType.FALL,
        confidence=0.95,
        zone_id="ZONE_A",
        source="AI_CAMERA",
    ))
    if result.get("action") == "EMERGENCY_TRIGGERED":
        print(f"\n  ⚠  Emergency triggered after {i + 1} events!")
        print(f"     Emergency type: {result.get('emergency_type')}")
        break
    time.sleep(1)

# ── Step 4: Request emergency access on-chain ────────────────

print("\n" + "─" * 60)
print("STEP 4 — Request emergency access on-chain (MEDICAL_PROBLEM)")
print("─" * 60)

emergency_type = "MEDICAL_PROBLEM"
score_int = int(server.zones["ZONE_A"].score)

print(f"  Emergency type: {emergency_type}")
print(f"  Zone score:     {score_int}")

receipt = send_tx(
    contract.functions.requestEmergencyAccess(id_hash, emergency_type, score_int)
)

# Parse EmergencyAccessGranted event
logs = contract.events.EmergencyAccessGranted().process_receipt(receipt)
if logs:
    print(f"  ✓ EmergencyAccessGranted event emitted")
    print(f"    idHash: 0x{logs[0].args.idHash.hex()}")
    print(f"    type:   {logs[0].args.emergencyType}")

# ── Step 5: ADM selective release (MEDICAL_PROBLEM) ──────────

print("\n" + "─" * 60)
print("STEP 5 — ADM selective release (MEDICAL_PROBLEM)")
print("─" * 60)

released = adm.selective_release(adm_ref, emergency_type)
if released:
    print(f"\n  ✓ Released fields (minimum required for {emergency_type}):")
    for key, value in released.items():
        print(f"    {key}: {value}")

    # Show what was NOT released
    withheld = [k for k in tourist_identity if k not in released]
    print(f"\n  ✗ Withheld fields (privacy protected):")
    for key in withheld:
        print(f"    {key}: [REDACTED]")

# ── Step 6: Multi-event union demo ────────────────────────────

print("\n" + "─" * 60)
print("STEP 6 — Multi-event union: SMALL_FIGHT + MEDICAL_EMERGENCY")
print("─" * 60)

multi_released = adm.multi_event_release(
    adm_ref, ["SMALL_FIGHT", "MEDICAL_EMERGENCY"]
)
if multi_released:
    print(f"\n  ✓ Union of fields for SMALL_FIGHT + MEDICAL_EMERGENCY:")
    for key, value in multi_released.items():
        print(f"    {key}: {value}")

    withheld = [k for k in tourist_identity if k not in multi_released]
    print(f"\n  ✗ Still withheld:")
    for key in withheld:
        print(f"    {key}: [REDACTED]")

# ── Step 7: Confirm release on-chain (audit trail) ───────────

print("\n" + "─" * 60)
print("STEP 7 — Confirm release on-chain (audit trail)")
print("─" * 60)

responder_address = account.address  # In production, this would be the responder
receipt = send_tx(
    contract.functions.confirmRelease(id_hash, responder_address)
)

logs = contract.events.ReleaseConfirmed().process_receipt(receipt)
if logs:
    print(f"  ✓ ReleaseConfirmed event emitted")
    print(f"    responder: {logs[0].args.responder}")

# ── Summary ───────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  DEMO COMPLETE — Full Audit Trail")
print("=" * 60)

access_count = contract.functions.getAccessLogCount(id_hash).call()
print(f"\n  On-chain access logs: {access_count}")

for i in range(access_count):
    log = contract.functions.getAccessLog(id_hash, i).call()
    print(f"\n  Log #{i + 1}:")
    print(f"    Emergency type: {log[0]}")
    print(f"    Score:          {log[1]}")
    print(f"    Timestamp:      {log[2]}")
    print(f"    Released:       {log[3]}")
    print(f"    Responder:      {log[4]}")

print(f"\n  Zone statuses: {json.dumps(server.get_zone_status(), indent=2)}")
print(f"  Review queue:  {len(server.review_queue)} pending")

# Cleanup
kmd.close()
print("\n✓ Demo finished successfully.\n")
