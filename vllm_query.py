import torch
from torchvision import transforms  # If needed, add specific modules from torchvision here
import time
import logging
import multiprocessing as mp
import subprocess
import os
import subprocess
import os
import time
import logging
import threading
import time
import sys
import csv
from datetime import datetime
from .image_utils import image_to_bytes
from .string_utils import fuzzy_match_bool

from .util.task_data import TaskData
from .transformer_worker.transformer_worker import list_cached_vision_models
from .transformer_worker.helpers import has_model_issues, get_model_issues, strip_warning_prefix

LOG_FILE = os.path.join(os.path.dirname(__file__), "logs", "inference_results.csv")


def log_attempt(model_name, gpu_device, attempt, outcome, error_message=None, final=False):
    """Logs inference attempts with an easy way to filter final results."""
    
    log_entry = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model_name,
        gpu_device,
        attempt,
        outcome,
        error_message if error_message else "None",
        "Yes" if final else "No"
    ]
    
    # Check if logs folder exists, or create one
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    # Ensure header exists
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Model Name", "GPU Device", "Attempt", "Outcome", "Error Message", "Final"])
        writer.writerow(log_entry)


def get_node_package_path():
    """Determine the package path for executing the worker as a module."""
    current_module = __name__.split(".")[0]  # Extract the root package (ComfyNodes)
    return current_module  # Return only the base package name

class PersistentInferenceWorker:
    def __init__(self, gpu_device, model_name, worker_module="transformer_worker"):
        self.worker_lock = threading.Lock()  # Mutex lock to prevent race conditions
        self.gpu_device = gpu_device
        self.model_name = model_name
        self.worker_module = worker_module
        self.worker = None
        self.parent_conn = None  # Store the pipe connection
        self.last_task = None  # Track last task in case of failure
        self.lock = threading.Lock()  # Ensure thread-safe task tracking
        self.failed_samples_dir = "failed_samples"
        os.makedirs(self.failed_samples_dir, exist_ok=True)  # Ensure folder exists
        self.start_worker()

    def monitor_stderr(process):
        """Continuously reads stderr from the worker and logs any errors."""
        while True:
            line = process.stderr.readline()
            if not line:
                break
            logging.error(f"⚠️ STDERR: {line.strip()}")  # Log stderr output

    def start_stderr_monitor(self):
        """Starts a background thread to capture and log `stderr` if logging level is DEBUG."""
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            logging.debug("🟡 Starting STDERR monitor thread...")

            def monitor_stderr():
                """Continuously reads and logs stderr output from the worker process."""
                while True:
                    if self.worker is None or self.worker.poll() is not None:
                        break  # Exit if the worker is no longer running
                    
                    line = self.worker.stderr.readline()
                    if not line:
                        break  # Stop if `stderr` is empty (EOF)

                    logging.error(f"⚠️ STDERR: {line.strip()}")  # Log stderr output

            # Start the monitoring thread in daemon mode
            stderr_thread = threading.Thread(target=monitor_stderr, daemon=True)
            stderr_thread.start()
        else:
            logging.debug("🟢 STDERR monitoring skipped (Log level is not DEBUG).")

    def start_worker(self):
        """Start the worker process, ensuring it's killed first if necessary."""
        with self.worker_lock:
            if self.worker is not None:
                logging.warning("❌ Killing old worker before restart...")
                self.worker.kill()  # Kill the worker process immediately
                self.worker.wait()  # Ensure it's fully dead before proceeding
                self.worker = None  # Reset reference

            logging.info("🚀 Spawning a new worker process...")
            self.parent_conn, child_conn = mp.Pipe()

            package_base = get_node_package_path()
            worker_module = f"{package_base}.{self.worker_module}"  # Ensures it matches the Comfy Node package path

            # Modify PYTHONPATH to ensure the worker sees ComfyNodes as the root, not custom_nodes.ComfyNodes
            comfy_root = os.path.abspath(os.getcwd())  # ComfyUI root (assumed to be CWD)
            custom_nodes_path = os.path.join(comfy_root, "custom_nodes")

            env = os.environ.copy()
            logging.debug(f"🔧 Original PYTHONPATH: {env.get('PYTHONPATH', '')}")
            env["PYTHONPATH"] = f"{custom_nodes_path}:{env.get('PYTHONPATH', '')}"  # Override PYTHONPATH

            self.worker = subprocess.Popen(
                [sys.executable, "-m", worker_module, self.gpu_device, self.model_name],
                stdin=child_conn,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                text=True,
                env=env
            )

            logging.debug("✅ Worker started, waiting for READY signal...")

            # Only start the stderr monitor if log level is DEBUG
            self.start_stderr_monitor()

            start_time = time.time()
            while time.time() - start_time < 30:
                if self.worker.poll() is not None:
                    logging.error("🚨 Worker crashed immediately after start.")
                    return

                try:
                    ready_signal = self.parent_conn.recv().strip()
                    if ready_signal == "READY":
                        logging.info("✅ Worker fully restarted and ready.")
                        return
                except Exception as e:
                    logging.error(f"🚨 Error waiting for worker: {e}")

            logging.error("🚨 Worker failed to start in time. Killing process.")
            self.worker.terminate()

    def save_failed_sample(self, task):
        """Saves a failed sample for debugging."""
        if not task:
            logging.error("❌ No task to save.")
            return

        image_bytes = task.image_bytes
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        failed_path = f"failed_sample_{timestamp}.png"

        with open(failed_path, "wb") as f:
            f.write(image_bytes)

        logging.info(f"💾 Saved failed sample: {failed_path}")

    def submit_task(self, task_data=None):
        """Send a task to the worker, ensuring it's alive and handling retries correctly."""

        # Ensure worker is running
        if self.worker is None or self.worker.poll() is not None:
            logging.error("🚨 Worker is dead or missing! Restarting before submitting task.")
            self.start_worker()  # Restart worker before continuing

        # Determine the correct task to send
        if task_data is None:
            if not self.last_task:
                logging.error("❌ No last task to retry.")
                return  # Nothing to do

            if self.last_task.is_retried:
                logging.error("❌ Task already retried once, skipping.")
                return  # Prevent infinite loops

            logging.info("🔄 Retrying last task...")
            self.last_task.is_retried = True  # Mark retry
            task_data = self.last_task  # Use stored task
        else:
            logging.info("🔄 Submitting new task...")
            # Save task before sending
            with self.lock:
                self.last_task = task_data
        
        try:
            logging.info("📩 Sending task to worker...")
            self.parent_conn.send(task_data)  # Send task via pipe
            logging.info("✅ Task submitted successfully.")

        except BrokenPipeError:
            logging.warning("🚨 Worker pipe broken! Terminating.")
                

    def get_result(self):
        """Receive results from the worker process, handling potential EOFErrors."""
        try:
            if self.parent_conn.poll(30):  # Wait up to 30 seconds for worker response
                return self.parent_conn.recv()  # Receive result (or error)
            else:
                logging.error("🚨 Worker unresponsive, treating as crash.")
                raise EOFError  # Simulate EOFError if worker is stuck

        except EOFError:
            logging.error("❌ Worker connection lost (EOFError). Restarting worker...")
            
            if self.worker:
                logging.warning("🛑 Force-killing unresponsive worker...")
                self.worker.terminate()
                self.worker.wait()  # Ensure cleanup
            
            self.worker = None  # 🔥 Unset reference
            return None


    def shutdown(self):
        """Gracefully terminate the worker process."""
        self.worker.terminate()
        logging.info("🛑 Worker process terminated.")
        
logging.basicConfig(level=logging.DEBUG)

class VisionLLMQuery:    
    _AVAILABLE_GPUS = None  # Cache for available GPUs
    _AVAILABLE_MODELS = None  # Cache for available models

    @classmethod
    def get_available_gpus(cls):
        """Lazily fetch the available CUDA devices (only once)."""
        if cls._AVAILABLE_GPUS is None:
            cls._AVAILABLE_GPUS = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
        return cls._AVAILABLE_GPUS

    @classmethod
    def get_available_models(cls):
        """Lazily fetch the available vision models (only once)."""
        if cls._AVAILABLE_MODELS is None:
            cls._AVAILABLE_MODELS = list_cached_vision_models()
        return cls._AVAILABLE_MODELS
    
    @classmethod
    def INPUT_TYPES(cls):
        """Dynamically defines input types, ensuring models & GPUs are listed only when needed."""
        available_gpus = cls.get_available_gpus()
        available_models = cls.get_available_models()
    
        return {
            "required": {
                "image": ("IMAGE",),  # Main image input
                "text_query": ("STRING", {"default": "Describe the image.", "multiline": True}),
                "gpu_device": (available_gpus, {"default": available_gpus[0]}),
                "model_name": (available_models, {"default": available_models[0]}),
            },
            "optional": {
                "reference_image": ("IMAGE",),  # Optional reference image
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "INT")
    RETURN_NAMES = ("Raw Text", "Boolean", "Number (Boolean)")
    FUNCTION = "run"
    CATEGORY = "AI/Large Language Models"

    def __init__(self):
        self.device = None

    def run(self, **inputs):
        """Uses a persistent worker process to run inference without crashing the parent process."""
        
        if "gpu_device" in inputs:
            gpu_device = inputs["gpu_device"]
        else:
            gpu_device = VisionLLMQuery.get_available_gpus()[0]
            logging.warning(f"Did not receive gpu_device, defaulting to {gpu_device}")
        
        if "model_name" in inputs:
            model_name = inputs["model_name"]
        else:
            model_name = VisionLLMQuery.get_available_models()[0]
            logging.warning(f"Did not receive model, defaulting to {model_name}")
        if has_model_issues(model_name):
            message = f"Model '{strip_warning_prefix(model_name)}' has known issues:"
            message += "\n" + '\n '.join(get_model_issues(model_name))
            logging.error(message)
            # Raise an error for unsupported model
            raise Exception(message)
        
        image = inputs["image"]
        text_query = inputs.get("text_query", "Describe the image.")

        reference_image = inputs.get("reference_image", None)  # Optional!
        max_retries = 3

        if not hasattr(self, "worker"):  # Create worker if not already running
            logging.debug("🚀 Starting persistent inference worker...")
            self.worker = PersistentInferenceWorker(gpu_device, model_name)

        attempt = 1
        while attempt <= max_retries:
            image_bytes = image_to_bytes(image)
            reference_bytes = image_to_bytes(reference_image) if reference_image is not None else None
            task = TaskData(image_bytes=image_bytes, reference_bytes=reference_bytes, text_query=text_query)

            self.worker.submit_task(task)  # Send task
            llm_response = self.worker.get_result()  # Wait for response

            is_final_attempt = (attempt == max_retries)  # Cleaner readability

            if llm_response is None:
                logging.error(f"🔥 Worker failed on attempt {attempt}/{max_retries}. Retrying...")
                log_attempt(model_name, gpu_device, attempt, "Failure", "Empty Response", final=is_final_attempt)
            elif isinstance(llm_response, Exception):  # Just retry, no worker restart
                logging.error(f"🚨 Worker threw an exception: {llm_response}. Retrying task...")
                log_attempt(model_name, gpu_device, attempt, "Failure", str(llm_response), final=is_final_attempt)
            else:
                logging.info(f"✅ Inference attempt {attempt}/{max_retries} succeeded.")
                logging.debug(f"🔤 Raw text: {llm_response}")

                bool_output = fuzzy_match_bool(llm_response)
                results = llm_response, bool_output, int(bool_output)
                logging.debug(f"Results: {results}")

                log_attempt(model_name, gpu_device, attempt, "Success", None, final=True)
                return results  # Success!

            attempt += 1
            torch.cuda.empty_cache()  # Clear VRAM between retries

        logging.error(f"❌ All {max_retries} inference attempts failed. Skipping.")
        return None  # Returns None instead of crashing
