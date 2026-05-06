"""
STSMIRS — Key Expiry Scheduler
Periodically runs the KMD key expiry sweep to enforce the
48-hour privacy guarantee (NFR3).

Usage:
    python scheduler.py

The scheduler runs every hour and zeroes out wrapped_key for
any records past their deletion_timestamp.
"""

import time
import threading
from adm import KMD


def run_scheduler(kmd: KMD, interval_seconds: int = 3600):
    """
    Run the key expiry sweep at regular intervals.

    Args:
        kmd:               KMD instance
        interval_seconds:  How often to run the sweep (default: 1 hour)
    """
    print(f"[SCHEDULER] Key expiry scheduler started (interval: {interval_seconds}s)")

    while True:
        try:
            deleted = kmd.run_key_expiry_sweep()
            if deleted > 0:
                print(f"[SCHEDULER] Expired {deleted} record(s) — wrapped keys zeroed out")
            else:
                print(f"[SCHEDULER] Sweep complete — no records expired")
        except Exception as e:
            print(f"[SCHEDULER] Error during sweep: {e}")

        time.sleep(interval_seconds)


def start_scheduler_thread(kmd: KMD, interval_seconds: int = 3600) -> threading.Thread:
    """
    Start the scheduler in a background daemon thread.
    Returns the thread object.
    """
    thread = threading.Thread(
        target=run_scheduler,
        args=(kmd, interval_seconds),
        daemon=True,
    )
    thread.start()
    return thread


if __name__ == "__main__":
    kmd = KMD(db_path="demo_kmd.db")
    print("[SCHEDULER] Connected to KMD database")
    try:
        run_scheduler(kmd, interval_seconds=3600)
    except KeyboardInterrupt:
        print("\n[SCHEDULER] Stopped by user")
    finally:
        kmd.close()
