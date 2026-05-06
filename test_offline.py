"""
STSMIRS — Offline Pipeline Test
=================================
Tests the full encrypt → store → selective-release pipeline
WITHOUT needing a live blockchain. Validates all 5 emergency types
and the multi-event union logic.

Usage:
    python test_offline.py
"""

import os
import sys

# ── Import project modules ────────────────────────────────────

from adm import ADM, KMD, RELEASE_FIELDS
from central_server import CentralServer, DetectionEvent, EventType

# ── Test Data ─────────────────────────────────────────────────

TEST_TOURIST = {
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

TESTS_PASSED = 0
TESTS_FAILED = 0


def assert_test(name, condition, detail=""):
    global TESTS_PASSED, TESTS_FAILED
    if condition:
        print(f"  ✓ PASS: {name}")
        TESTS_PASSED += 1
    else:
        print(f"  ✗ FAIL: {name} — {detail}")
        TESTS_FAILED += 1


# ── Setup ─────────────────────────────────────────────────────

print("=" * 60)
print("  STSMIRS — Offline Pipeline Test")
print("=" * 60)

# Use in-memory DB for testing
kmd = KMD(db_path=":memory:")
adm = ADM(kmd=kmd)

# ── Test 1: Encryption & Decryption ──────────────────────────

print("\n─── Test 1: Encrypt + Decrypt ───")
adm_ref, id_hash = adm.encrypt_identity(TEST_TOURIST)

assert_test("adm_ref is UUID string", len(adm_ref) == 36 and "-" in adm_ref)
assert_test("id_hash is 32 bytes", len(id_hash) == 32)

decrypted = adm.decrypt_identity(adm_ref)
assert_test("Decryption returns all fields", decrypted == TEST_TOURIST,
            f"Got {decrypted}")

# ── Test 2: Selective Release — All 5 Emergency Types ────────

print("\n─── Test 2: Selective Release (all 5 types) ───")

for etype, expected_fields in RELEASE_FIELDS.items():
    released = adm.selective_release(adm_ref, etype)
    released_keys = set(released.keys()) if released else set()
    expected_keys = set(expected_fields)

    # Only include keys that exist in the test tourist
    expected_present = expected_keys & set(TEST_TOURIST.keys())

    assert_test(
        f"{etype}: releases correct fields",
        released_keys == expected_present,
        f"Expected {expected_present}, got {released_keys}"
    )

    # Verify nothing extra leaked
    extra = released_keys - expected_keys
    assert_test(
        f"{etype}: no extra fields leaked",
        len(extra) == 0,
        f"Leaked: {extra}"
    )

# ── Test 3: Multi-Event Union ────────────────────────────────

print("\n─── Test 3: Multi-Event Union ───")

# SMALL_FIGHT + MEDICAL_EMERGENCY = union
union = adm.multi_event_release(adm_ref, ["SMALL_FIGHT", "MEDICAL_EMERGENCY"])
expected_union = set(RELEASE_FIELDS["SMALL_FIGHT"]) | set(RELEASE_FIELDS["MEDICAL_EMERGENCY"])
expected_present = expected_union & set(TEST_TOURIST.keys())
union_keys = set(union.keys()) if union else set()

assert_test(
    "SMALL_FIGHT + MEDICAL_EMERGENCY union",
    union_keys == expected_present,
    f"Expected {expected_present}, got {union_keys}"
)

# All types at once
all_types = list(RELEASE_FIELDS.keys())
all_union = adm.multi_event_release(adm_ref, all_types)
all_expected = set()
for fields in RELEASE_FIELDS.values():
    all_expected.update(fields)
all_expected_present = all_expected & set(TEST_TOURIST.keys())
all_union_keys = set(all_union.keys()) if all_union else set()

assert_test(
    "All 5 types union",
    all_union_keys == all_expected_present,
    f"Expected {all_expected_present}, got {all_union_keys}"
)

# ── Test 4: Key Expiry (NFR3) ────────────────────────────────

print("\n─── Test 4: Key Expiry (NFR3) ───")

# Store a tourist with 0 retention (expires immediately)
adm_ref2 = kmd.store_identity(
    encrypted_identity=b"test_cipher",
    wrapped_key=b"test_key",
    cipher_meta={"algorithm": "AES-256-GCM", "nonce": "abc123", "key_wrap": "RSA-OAEP"},
    retention_hours=0,  # Expires immediately
)

expired_count = kmd.run_key_expiry_sweep()
assert_test("Sweep expires records", expired_count >= 1, f"Expired: {expired_count}")

record = kmd.get_identity(adm_ref2)
assert_test("Expired record status is DELETED", record["status"] == "DELETED")
assert_test("Expired record key is empty", record["wrapped_key"] == b"")

# ── Test 5: Central Server — Confidence Routing ──────────────

print("\n─── Test 5: Confidence Routing ───")

server = CentralServer()  # No contract (offline mode)

# Low confidence → discard
result = server.ingest_event(DetectionEvent(
    tourist_id="T001", event_type=EventType.FALL,
    confidence=0.50, zone_id="Z1", source="AI_CAMERA",
))
assert_test("Low confidence discarded",
            result["action"] == "DISCARDED (low confidence)")

# Medium confidence → review queue
result = server.ingest_event(DetectionEvent(
    tourist_id="T001", event_type=EventType.FALL,
    confidence=0.85, zone_id="Z1", source="AI_CAMERA",
))
assert_test("Medium confidence queued",
            result["action"] == "QUEUED_FOR_REVIEW (medium confidence)")
assert_test("Review queue has 1 item", len(server.review_queue) == 1)

# High confidence → processed
result = server.ingest_event(DetectionEvent(
    tourist_id="T001", event_type=EventType.FALL,
    confidence=0.97, zone_id="Z1", source="AI_CAMERA",
))
assert_test("High confidence processed",
            result["action"] in ["PROCESSED", "EMERGENCY_TRIGGERED"])

# ── Test 6: Emergency Type Mapping ───────────────────────────

print("\n─── Test 6: Event → Emergency Type Mapping ───")

from central_server import EVENT_TO_EMERGENCY

expected_mapping = {
    EventType.FALL: "MEDICAL_PROBLEM",
    EventType.PANIC: "SMALL_FIGHT",
    EventType.AGGRESSION: "VIOLENT_FIGHT",
    EventType.SURGE: "SMALL_FIGHT",
    EventType.CROWD_DENSITY: "OFFENCE",
    EventType.VIBRATION_ANOMALY: "OFFENCE",
}

for event_type, expected_emergency in expected_mapping.items():
    actual = EVENT_TO_EMERGENCY.get(event_type)
    assert_test(
        f"{event_type.value} → {expected_emergency}",
        actual == expected_emergency,
        f"Got {actual}"
    )

# ── Summary ───────────────────────────────────────────────────

print("\n" + "=" * 60)
total = TESTS_PASSED + TESTS_FAILED
print(f"  Results: {TESTS_PASSED}/{total} passed, {TESTS_FAILED} failed")
print("=" * 60)

kmd.close()

if TESTS_FAILED > 0:
    print("\n✗ Some tests failed!")
    sys.exit(1)
else:
    print("\n✓ All tests passed!\n")
    sys.exit(0)
