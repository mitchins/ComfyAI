import time
import sys
import os
import signal
import multiprocessing as mp
import logging

LOG_FILE = os.path.join(os.path.dirname(__file__), "dummy_worker.log")

# Create a dedicated logger
logger = logging.getLogger("dummy_worker")
logger.setLevel(logging.DEBUG)

if logger.hasHandlers():
    logger.handlers.clear()

file_handler = logging.FileHandler(LOG_FILE, mode="w")
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


def worker_loop(child_conn, crash_after_ms=0, crash_on_signal=False):
    """Fake worker that does tasks but can be forced to crash."""
    logger.info("🚀 Dummy worker started!")

    if crash_on_signal:
        def handle_signal(sig, frame):
            logger.error(f"🔥 Received signal {sig}, crashing...")
            sys.exit(1)

        signal.signal(signal.SIGTERM, handle_signal)

    child_conn.send("READY")  # ✅ Notify parent we're ready
    logger.info("✅ Dummy worker is ready.")

    while True:
        try:
            if crash_after_ms:
                time.sleep(crash_after_ms / 1000)
                raise RuntimeError("💀 Simulated crash due to timeout!")

            task = child_conn.recv()
            if task is None:
                logger.info("🛑 Received shutdown signal. Stopping worker.")
                break  # Stop the worker

            logger.info(f"📩 Received task: {task}")

            # Simulate work
            time.sleep(1)
            response = f"Processed: {task}"

            child_conn.send(response)
            logger.info(f"📤 Sent result: {response}")

        except EOFError:
            logger.error("🚨 Connection closed, shutting down.")
            break
        except Exception as e:
            logger.error(f"🔥 Worker process crashed! {e}")
            child_conn.send(None)  # Simulate returning an error


if __name__ == "__main__":
    """Worker process entry point."""
    if len(sys.argv) < 2:
        logger.error("❌ Missing arguments! Expected: dummy_worker.py <CRASH_AFTER_MS>")
        sys.exit(1)

    crash_after_ms = int(sys.argv[1])
    worker_loop(mp.connection.Connection(sys.stdin.fileno()), crash_after_ms)