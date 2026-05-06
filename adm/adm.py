"""
ADM — Authorised Decryption Module
Handles encryption, decryption, and selective identity release
for the STSMIRS privacy-preserving emergency access system.
"""

import hashlib
import json
import os
import threading
import uuid
from typing import Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from hexbytes import HexBytes
from web3 import Web3

from .kmd import KMD

# ── Minimum field release rules (FR6) ─────────────────────────
# Each emergency type releases ONLY the minimum required fields.
# Multi-event rule: union of all required fields across active types.

RELEASE_FIELDS = {
    "SMALL_FIGHT":        ["name", "face"],
    "VIOLENT_FIGHT":      ["name", "face", "phone_number", "past_police_records", "emergency_contact"],
    "MEDICAL_PROBLEM":    ["name", "face", "phone_number"],
    "MEDICAL_EMERGENCY":  ["name", "face", "phone_number", "emergency_contact", "medical_history"],
    "OFFENCE":            ["name", "face", "phone_number"],
}

DEFAULT_RELEASE = ["name", "face"]


class ADM:
    """Authorised Decryption Module — the privacy gatekeeper."""

    def __init__(self, kmd: KMD, key_size: int = 4096):
        self.kmd = kmd
        # Persist RSA keys next to the KMD DB so decryption survives restarts
        self._key_path = os.path.splitext(kmd.conn.execute("PRAGMA database_list").fetchone()[2])[0] + "_adm_master.pem"
        self._load_or_generate_master_keys(key_size)
        self._listener_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # Maps id_hash_hex → adm_ref (populated at enrollment)
        self._hash_to_ref: dict[str, str] = {}

    # ── Key Generation / Persistence ─────────────────────────

    def _load_or_generate_master_keys(self, key_size: int) -> None:
        """Load RSA master key pair from disk, or generate and save a new one."""
        if os.path.exists(self._key_path):
            with open(self._key_path, "rb") as f:
                self._private_key = serialization.load_pem_private_key(f.read(), password=None)
            self._public_key = self._private_key.public_key()
            print(f"[ADM] Loaded existing RSA master key from {self._key_path}")
        else:
            self._private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
            )
            self._public_key = self._private_key.public_key()
            pem = self._private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
            with open(self._key_path, "wb") as f:
                f.write(pem)
            print(f"[ADM] Generated new RSA master key → {self._key_path}")

    def get_public_key_pem(self) -> bytes:
        """Return the master public key in PEM format."""
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    # ── Encrypt (Enrollment) ──────────────────────────────────

    def encrypt_identity(self, identity: dict) -> tuple[str, bytes]:
        """
        Encrypt a tourist identity JSON object.
        Steps:
          1. Generate a random AES-256 key
          2. AES-256-GCM encrypt the identity JSON
          3. RSA-OAEP wrap the AES key with the ADM master public key
          4. Store everything in the KMD

        Returns: (adm_ref, id_hash_bytes)
          - adm_ref:  UUID pointer stored on-chain
          - id_hash:  SHA-256 of a random UUID (the on-chain anchor)
        """
        # Random AES-256 key
        aes_key = AESGCM.generate_key(bit_length=256)
        nonce = os.urandom(12)
        aesgcm = AESGCM(aes_key)

        # Encrypt identity JSON
        plaintext = json.dumps(identity).encode("utf-8")
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        # Wrap AES key with RSA-OAEP
        wrapped_key = self._public_key.encrypt(
            aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

        # Cipher metadata needed for decryption
        cipher_meta = {
            "algorithm": "AES-256-GCM",
            "nonce": nonce.hex(),
            "key_wrap": "RSA-OAEP-SHA256",
        }

        # Store in KMD
        adm_ref = self.kmd.store_identity(
            encrypted_identity=ciphertext,
            wrapped_key=wrapped_key,
            cipher_meta=cipher_meta,
        )

        # Generate on-chain anchor: SHA-256 of random UUID
        random_uuid = str(uuid.uuid4())
        id_hash = hashlib.sha256(random_uuid.encode()).digest()

        # Cache the mapping for the event listener
        self._hash_to_ref[id_hash.hex()] = adm_ref

        return adm_ref, id_hash

    # ── Decrypt (Emergency Access) ────────────────────────────

    def decrypt_identity(self, adm_ref: str) -> Optional[dict]:
        """
        Fully decrypt a tourist identity from the KMD.
        Returns the identity dict or None if the record is deleted / not found.
        """
        record = self.kmd.get_identity(adm_ref)
        if record is None:
            print(f"[ADM] Record not found: {adm_ref}")
            return None

        if record["status"] == "DELETED" or not record["wrapped_key"]:
            print(f"[ADM] Record key has been deleted (NFR3): {adm_ref}")
            return None

        # Unwrap AES key
        aes_key = self._private_key.decrypt(
            record["wrapped_key"],
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

        # Decrypt identity
        meta = record["cipher_meta"]
        nonce = bytes.fromhex(meta["nonce"])
        aesgcm = AESGCM(aes_key)
        plaintext = aesgcm.decrypt(nonce, record["encrypted_identity"], None)
        return json.loads(plaintext.decode("utf-8"))

    # ── Selective Release (FR6) ───────────────────────────────

    def selective_release(self, adm_ref: str, emergency_type: str) -> Optional[dict]:
        """
        Decrypt and return ONLY the minimum required fields for the
        given emergency type.  This enforces the FR6 privacy constraint.
        """
        full_identity = self.decrypt_identity(adm_ref)
        if full_identity is None:
            return None

        allowed = RELEASE_FIELDS.get(emergency_type.upper(), DEFAULT_RELEASE)
        released = {k: v for k, v in full_identity.items() if k in allowed}

        print(f"[ADM] Selective release ({emergency_type}): {list(released.keys())}")
        return released

    def multi_event_release(self, adm_ref: str, emergency_types: list[str]) -> Optional[dict]:
        """
        Handle multi-event scenarios: union of all required fields
        across multiple simultaneous emergency types.
        """
        full_identity = self.decrypt_identity(adm_ref)
        if full_identity is None:
            return None

        # Union of all required fields
        all_fields: set[str] = set()
        for etype in emergency_types:
            fields = RELEASE_FIELDS.get(etype.upper(), DEFAULT_RELEASE)
            all_fields.update(fields)

        released = {k: v for k, v in full_identity.items() if k in all_fields}
        print(f"[ADM] Multi-event release ({emergency_types}): {list(released.keys())}")
        return released

    # ── Blockchain Authorization Gatekeeper ──────────────────

    def verify_authorization(
        self,
        contract,
        id_hash: bytes,
        adm_ref: str,
        web3_instance=None,
        receipt=None,
    ) -> Optional[str]:
        """
        Verify that the smart contract has authorized emergency access
        by checking for an EmergencyAccessGranted event on-chain.

        Primary path: decode logs directly from a tx receipt (no extra RPC call).
        Fallback path: eth_getLogs over the last 200 blocks.

        Args:
            contract:       web3 contract instance
            id_hash:        SHA-256 hash of the enrolled identity (bytes)
            adm_ref:        UUID pointer to the encrypted record
            web3_instance:  Web3 instance (required for fallback path)
            receipt:        Transaction receipt from requestEmergencyAccess (preferred)

        Returns:
            The authorized emergency_type string, or None if not authorized.
        """
        try:
            # ── 1. Confirm identity is enrolled on-chain ──────────────
            on_chain = contract.functions.getIdentity(id_hash).call()
            if not on_chain or on_chain[0] == b'\x00' * 32:
                print(f"[ADM] Authorization DENIED: identity not found on-chain")
                return None

            stored_adm_ref = on_chain[1]
            if stored_adm_ref != adm_ref:
                print(
                    f"[ADM] Authorization DENIED: adm_ref mismatch "
                    f"(got {adm_ref}, expected {stored_adm_ref})"
                )
                return None

            # ── 2a. Fast path — decode logs directly from the receipt ───
            if receipt is not None:
                for raw_log in receipt.get("logs", []):
                    try:
                        decoded = contract.events.EmergencyAccessGranted().process_log(raw_log)
                        if bytes(decoded["args"]["idHash"]) == id_hash:
                            etype = decoded["args"]["emergencyType"]
                            print(
                                f"[ADM] Authorization VERIFIED from receipt: "
                                f"{etype} (block {decoded['blockNumber']})"
                            )
                            return etype
                    except Exception:
                        continue  # log belongs to a different event, skip it
                print("[ADM] Receipt had no matching event; trying eth_getLogs fallback...")

            # ── 2b. Fallback — eth_getLogs (tight range to avoid 400) ──
            w3 = web3_instance or getattr(contract, "web3", None)
            if not w3:
                print("[ADM] Authorization ERROR: no web3 instance for fallback")
                return None

            current_block = w3.eth.block_number
            search_from = max(0, current_block - 200)  # tight range avoids Alchemy limits

            event_sig = "0x" + w3.keccak(
                text="EmergencyAccessGranted(bytes32,string,uint256,uint256)"
            ).hex()
            id_hash_topic = "0x" + id_hash.hex().zfill(64)

            try:
                raw_logs = w3.eth.get_logs({
                    "address": contract.address,
                    "fromBlock": hex(search_from),
                    "toBlock": hex(current_block),
                    "topics": [event_sig, id_hash_topic],
                })
                for raw_log in raw_logs:
                    decoded = contract.events.EmergencyAccessGranted().process_log(raw_log)
                    etype = decoded["args"]["emergencyType"]
                    print(
                        f"[ADM] ✓ Authorization VERIFIED via logs: "
                        f"{etype} (block {decoded['blockNumber']})"
                    )
                    return etype
            except Exception as log_err:
                print(f"[ADM] eth_getLogs fallback failed: {log_err}")

            print("[ADM] Authorization DENIED: no EmergencyAccessGranted event found")
            return None

        except Exception as e:
            print(f"[ADM] Authorization ERROR: {e}")
            return None

    # ── Event Listener (background thread) ────────────────────

    def start_event_listener(self, contract, poll_interval: int = 5) -> None:
        """
        Start a background thread that polls for EmergencyAccessGranted events.
        When an event fires, the ADM automatically looks up the adm_ref from
        the on-chain identity record and triggers selective_release.

        Args:
            contract:      web3 contract instance
            poll_interval: seconds between polls
        """
        if self._listener_thread and self._listener_thread.is_alive():
            print("[ADM] Event listener is already running.")
            return

        self._stop_event.clear()

        def _listen():
            print("[ADM] Event listener started — watching for EmergencyAccessGranted...")
            event_filter = contract.events.EmergencyAccessGranted.create_filter(
                fromBlock="latest"
            )
            while not self._stop_event.is_set():
                try:
                    for event in event_filter.get_new_entries():
                        id_hash_hex = event.args.idHash.hex()
                        emergency_type = event.args.emergencyType
                        print(f"\n[ADM] ⚠  EmergencyAccessGranted detected!")
                        print(f"       idHash: 0x{id_hash_hex}")
                        print(f"       type:   {emergency_type}")

                        # Resolve adm_ref from on-chain identity record
                        try:
                            on_chain = contract.functions.getIdentity(
                                event.args.idHash
                            ).call()
                            adm_ref = on_chain[1]  # admRef field
                            print(f"       admRef: {adm_ref}")

                            # Auto-trigger selective release
                            released = self.selective_release(adm_ref, emergency_type)
                            if released:
                                print(f"       Released: {list(released.keys())}")
                        except Exception as lookup_err:
                            # Fallback: try local cache
                            adm_ref = self._hash_to_ref.get(id_hash_hex)
                            if adm_ref:
                                released = self.selective_release(adm_ref, emergency_type)
                                if released:
                                    print(f"       Released (cache): {list(released.keys())}")
                            else:
                                print(f"       Could not resolve adm_ref: {lookup_err}")
                except Exception as exc:
                    print(f"[ADM] Listener error: {exc}")
                self._stop_event.wait(poll_interval)
            print("[ADM] Event listener stopped.")

        self._listener_thread = threading.Thread(target=_listen, daemon=True)
        self._listener_thread.start()

    def stop_event_listener(self) -> None:
        """Signal the listener thread to stop."""
        self._stop_event.set()
        if self._listener_thread:
            self._listener_thread.join(timeout=10)
