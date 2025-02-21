import torch
import logging
import sys
import io
import os
import time
from PIL import Image
from huggingface_hub import scan_cache_dir
from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq
from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
import multiprocessing as mp  # Use multiprocessing for pipes
from qwen_vl_utils import process_vision_info
from ..util import TaskData
from .helpers import list_cached_vision_models, get_attention_implementation

# Optimize image tokenization
min_pixels = 128 * 28 * 28  # Lower per-image token count
max_pixels = 512 * 28 * 28  # Ensure two images fit in 8GB VRAM

# Set up logger globally
LOG_FILE = os.path.join(os.path.dirname(__file__), "worker.log")

logger = logging.getLogger("worker")
logger.setLevel(logging.DEBUG)

# Remove any existing handlers to prevent duplication
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# File Handler
file_handler = logging.FileHandler(LOG_FILE, mode="w")
file_handler.setLevel(logging.DEBUG)

# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# Unified log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Attach handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def load_model(gpu_device, model_name):
    """Loads the model inside the worker process and ensures it uses only the assigned GPU."""
    logger.info(f"🖥️ Loading model '{model_name}' on {gpu_device} inside worker...")
    torch.cuda.set_device(gpu_device)

    processor = AutoProcessor.from_pretrained(
        model_name, min_pixels=min_pixels, max_pixels=max_pixels
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation=get_attention_implementation(model_name),
        device_map={"": gpu_device}
    )
    model.eval()

    logger.info(f"✅ Model '{model_name}' loaded successfully on {gpu_device}")
    return processor, model


def bytes_to_image(image_bytes):
    """Convert PNG bytes back to a PIL Image."""
    return Image.open(io.BytesIO(image_bytes))


def run_inference(task, processor, model, gpu_device):
    """Runs inference using PNG-encoded images sent from the main process."""
    logger.debug("🟢 Received inference request")

    image_pil = bytes_to_image(task.image_bytes)
    reference_pil = bytes_to_image(task.reference_bytes) if task.reference_bytes is not None else None

    messages = [{"role": "user", "content": [{"type": "image", "image": image_pil}]}]
    if reference_pil:
        messages[0]["content"].insert(0, {"type": "image", "image": reference_pil})
    messages[0]["content"].append({"type": "text", "text": task.text_query})

    # BEFORE Model Call
    logger.debug(f"🛠 Preparing inputs on {gpu_device} | Image Size: {image_pil.size}")

    # Process images and text into input tensors
    image_inputs, video_inputs = process_vision_info(messages)

    logger.debug("✅ Inputs processed successfully")

    # Ensure the model input tensors are explicitly moved to the correct device
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = {key: value.to(gpu_device) for key, value in inputs.items()}  # Ensures all inputs are on the same device

    logger.debug("🧠 Running model inference...")
    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    end_time = time.time()
    logger.debug("✅ Model inference complete")

    num_input_tokens = inputs["input_ids"].shape[1]  # Number of input tokens
    num_generated_tokens = generated_ids.shape[1] - num_input_tokens  # Number of new tokens

    logger.info(
        f"Inference completed in {end_time - start_time:.2f} seconds | "
        f"Input tokens: {num_input_tokens} | Generated tokens: {num_generated_tokens}"
    )

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text


def worker_loop(child_conn, gpu_device, model_name):
    """Standalone worker process that handles inference independently."""
    logger.info(f"🚀 Worker started on {gpu_device} with model '{model_name}'")

    processor, model = load_model(gpu_device, model_name)

    # Send "READY" signal to the main process
    child_conn.send("READY")  # Send signal over the pipe
    logger.info("✅ Worker is ready to receive tasks")

    while True:
        try:
            logger.debug("🔄 Waiting for task...")
            task = child_conn.recv()  # Receive from the multiprocessing pipe
            
            if task is None:
                logger.info("🛑 Received shutdown signal. Stopping worker.")
                break  # Stop the worker

            logger.debug("📩 Received new task")
            results = run_inference(task, processor, model, gpu_device)
            logger.debug("✅ Task completed, sending results back")
            child_conn.send(results)  # Send result back
            logger.debug("📤 Sent results back to main process")

        except EOFError:
            logger.error("🚨 Worker: Connection closed, shutting down.")
            break  # Exit if the main process dies

        except Exception as e:
            logger.error(f"🚨 Worker Process Error: {e}")
            child_conn.send(e)  # Send exception instead of retrying

class TransformerWorker:
    @staticmethod
    def main():
        """Worker process entry point."""
        if len(sys.argv) < 3:
            logger.error("❌ Missing arguments! Expected: worker.py <GPU_DEVICE> <MODEL_NAME>")
            sys.exit(1)

        gpu_device = sys.argv[1]
        model_name = sys.argv[2]  # Allow dynamic model selection

        logger.info(f"🔌 Requested to start on GPU {gpu_device} with model '{model_name}'")
        
        worker_loop(mp.connection.Connection(sys.stdin.fileno()), gpu_device, model_name)