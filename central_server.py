"""
Central Server — Score tracking and blockchain trigger logic
for the STSMIRS system.

This is the ONLY interface point for teammates:
    from central_server import CentralServer, DetectionEvent, EventType
    server = CentralServer(...)
    server.ingest_event(event)
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Data Models ──────────────────────────────────────────────────

class EventType(Enum):
    FALL = "FALL"
    PANIC = "PANIC"
    AGGRESSION = "AGGRESSION"
    SURGE = "SURGE"
    CROWD_DENSITY = "CROWD_DENSITY"


# Map detection events to blockchain emergency types
EVENT_TO_EMERGENCY = {
    EventType.FALL:               "MEDICAL_PROBLEM",
    EventType.PANIC:              "MEDICAL_EMERGENCY",
    EventType.AGGRESSION:         "VIOLENT_FIGHT",
    EventType.SURGE:              "MEDICAL_EMERGENCY",
    EventType.CROWD_DENSITY:      "SMALL_FIGHT",
}


@dataclass
class DetectionEvent:
    event_type: EventType
    confidence: float          # 0.0 – 1.0
    source: str                # "AI_CAMERA" or "IOT_SENSOR"
    zone_id: str = "ZONE_A"
    tourist_id: Optional[str] = None   # set by AI camera (adm_ref/UUID); None for IoT
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.tourist_id is None:
            self.tourist_id = f"ZONE:{self.zone_id}"


@dataclass
class ZoneSafety:
    zone_id: str
    score: float = 100.0       # starts at max safety
    event_count: int = 0


# ── Central Server ───────────────────────────────────────────────

class CentralServer:
    """
    Ingests detection events, maintains per-zone safety scores,
    and triggers the blockchain when a score drops below threshold.
    """

    # Score penalties per event type
    SCORE_PENALTIES = {
        EventType.FALL:               25.0,
        EventType.PANIC:              35.0,
        EventType.AGGRESSION:         30.0,
        EventType.SURGE:              25.0,
        EventType.CROWD_DENSITY:      20.0,
    }

    THRESHOLD = 50.0  # Below this → emergency

    def __init__(
        self,
        contract=None,
        web3=None,
        account=None,
        adm=None,
    ):
        """
        Args:
            contract:  web3 contract instance (deployed STSMIRS)
            web3:      Web3 provider instance
            account:   deployer/server account address
            adm:       ADM instance (for automatic selective release)
        """
        self.contract = contract
        self.web3 = web3
        self.account = account
        self.adm = adm
        
        import os
        self.state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zones_state.json")
        self.events_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "events_log.json")
        
        self.zones: dict[str, ZoneSafety] = defaultdict(lambda: ZoneSafety(zone_id=""))
        self._load_state()
        
        self.review_queue: list[DetectionEvent] = []
        self.processed_events: list[dict] = []
        self._load_events()

    def _load_events(self):
        import os
        import json
        if os.path.exists(self.events_file):
            try:
                with open(self.events_file, "r") as f:
                    self.processed_events = json.load(f)
            except Exception as e:
                print(f"[SERVER] Error loading events state: {e}")

    def _save_events(self):
        import json
        try:
            with open(self.events_file, "w") as f:
                json.dump(self.processed_events, f, indent=2)
        except Exception as e:
            print(f"[SERVER] Error saving events state: {e}")

    def _load_state(self):
        import os
        import json
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    for zid, zdata in data.items():
                        z = self.zones[zid]
                        z.zone_id = zdata.get("zone_id", zid)
                        z.score = zdata.get("score", 100.0)
                        z.event_count = zdata.get("event_count", 0)
            except Exception as e:
                print(f"[SERVER] Error loading zones state: {e}")

    def _save_state(self):
        import json
        data = {
            zid: {"zone_id": z.zone_id, "score": z.score, "event_count": z.event_count}
            for zid, z in self.zones.items()
        }
        try:
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[SERVER] Error saving zones state: {e}")

    # ── Ingest (main entry point) ─────────────────────────────

    def ingest_event(self, event: DetectionEvent) -> dict:
        """
        Process a detection event.
        Confidence routing:
          > 0.90  → auto-process (high confidence)
          0.70–0.90 → queue for human review
          < 0.70  → discard
        """
        result = {
            "tourist_id": event.tourist_id,
            "event_type": event.event_type.value,
            "confidence": event.confidence,
            "zone_id": event.zone_id,
            "action": None,
        }

        if event.confidence < 0.70:
            result["action"] = "DISCARDED (low confidence)"
            print(f"[SERVER] Discarded event — confidence {event.confidence:.2f} < 0.70")
            return result

        if event.confidence < 0.90:
            result["action"] = "QUEUED_FOR_REVIEW (medium confidence)"
            self.review_queue.append(event)
            print(
                f"[SERVER] Queued for human review — confidence {event.confidence:.2f}"
            )
            return result

        # High confidence → auto-process
        return self._process_event(event, result)

    def _process_event(self, event: DetectionEvent, result: dict) -> dict:
        """Process a high-confidence event: update score, trigger blockchain immediately."""
        # Ensure zone exists with proper id
        zone = self.zones[event.zone_id]
        zone.zone_id = event.zone_id

        # Apply penalty (tracked separately from the decision to trigger)
        penalty = self.SCORE_PENALTIES.get(event.event_type, 15.0)
        penalty *= event.confidence  # scale by confidence
        zone.score = max(0.0, zone.score - penalty)
        zone.event_count += 1

        print(
            f"[SERVER] Zone {event.zone_id}: score {zone.score:.1f} "
            f"(penalty -{penalty:.1f}, event #{zone.event_count})"
        )

        result["zone_score"] = zone.score

        # Trigger emergency immediately since a high-confidence event occurred
        print(f"[SERVER] ⚠  EMERGENCY TRIGGERED — High-confidence {event.event_type.value} detected.")
        emergency_type = EVENT_TO_EMERGENCY.get(event.event_type, "MEDICAL_PROBLEM")
        
        # We still pass the zone's score to the blockchain for context, 
        # but it no longer gates the trigger decision.
        self._trigger_blockchain(event, emergency_type, zone.score)
        
        result["action"] = "EMERGENCY_TRIGGERED"
        result["emergency_type"] = emergency_type

        self.processed_events.append(result)
        self._save_state()
        self._save_events()
        return result

    # ── Blockchain Interface ──────────────────────────────────

    def _trigger_blockchain(
        self, event: DetectionEvent, emergency_type: str, score: float
    ) -> Optional[str]:
        """Call requestEmergencyAccess on the deployed smart contract."""
        if self.contract is None:
            print("[SERVER] No contract connected — skipping blockchain call")
            return None

        try:
            # Convert tourist_id to bytes32 id_hash
            id_hash = self._tourist_id_to_hash(event.tourist_id)
            score_int = int(score)

            print(f"[SERVER] Calling requestEmergencyAccess on-chain...")
            print(f"         idHash: 0x{id_hash.hex()}")
            print(f"         type:   {emergency_type}")
            print(f"         score:  {score_int}")

            tx = self.contract.functions.requestEmergencyAccess(
                id_hash, emergency_type, score_int
            ).build_transaction({
                "from": self.account,
                "nonce": self.web3.eth.get_transaction_count(self.account),
                "gas": 200000,
                "gasPrice": self.web3.eth.gas_price,
            })

            signed = self.web3.eth.account.sign_transaction(
                tx, private_key=self._get_private_key()
            )
            tx_hash = self.web3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            print(f"[SERVER] ✓ TX confirmed: {receipt.transactionHash.hex()}")
            return receipt.transactionHash.hex()

        except Exception as exc:
            print(f"[SERVER] Blockchain call failed: {exc}")
            return None

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _tourist_id_to_hash(tourist_id: str) -> bytes:
        """Convert a tourist ID string to the same bytes32 used on-chain."""
        import hashlib
        return hashlib.sha256(tourist_id.encode()).digest()

    @staticmethod
    def _get_private_key() -> str:
        """Load private key from environment."""
        import os
        from dotenv import load_dotenv
        load_dotenv()
        pk = os.getenv("PRIVATE_KEY", "")
        if not pk.startswith("0x"):
            pk = "0x" + pk
        return pk

    # ── Review Queue ──────────────────────────────────────────

    def approve_review(self, index: int = 0) -> dict:
        """Manually approve a queued event (simulates human review)."""
        if not self.review_queue:
            print("[SERVER] Review queue is empty.")
            return {}
        event = self.review_queue.pop(index)
        result = {
            "tourist_id": event.tourist_id,
            "event_type": event.event_type.value,
            "confidence": event.confidence,
            "zone_id": event.zone_id,
            "action": None,
        }
        return self._process_event(event, result)

    def set_zone_score(self, zone_id: str, new_score: float) -> None:
        """Manually set a zone's safety score (e.g., for testing or admin overrides)."""
        zone = self.zones[zone_id]
        zone.zone_id = zone_id
        zone.score = max(0.0, min(100.0, new_score))
        print(f"[SERVER] Zone {zone_id} score manually set to {zone.score:.1f}")
        self._save_state()

    def get_zone_status(self) -> dict:
        """Return current safety scores for all zones."""
        return {
            zone_id: {"score": z.score, "event_count": z.event_count}
            for zone_id, z in self.zones.items()
        }
