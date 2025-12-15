import time
import os
import torch
import ctranslate2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np

# --- CONFIGURATION ---
HUGGINGFACE_MODEL_PATH = os.path.join("multilingual_transliteration_model", "final_best_model")
CT2_MODEL_PATH = "ct2_transliteration_model_int8"
# ---------------------

# Example Test Data (Romanized input for Hindi, Bengali, Tamil)
test_inputs = [
    "Namaste, aapka naam kya hai?",        # Hindi
    "Ami tomake bhalobashi",               # Bengali
    "Enna per sollunga",                   # Tamil
    "dhananjay",
    "pariksha",
    "shubhkamnayein",
] * 10 # Repeat to get a reasonable run time for benchmarking
NUM_RUNS = 100 # Number of times to run inference for speed averaging

print(f"\n--- Starting Model Benchmarking on {len(test_inputs)} examples ---")

# --- 1. Load Tokenizer ---
try:
    tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_PATH)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()

# --- 2. Load PyTorch Model ---
print("Loading PyTorch Model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    pt_model = AutoModelForSeq2SeqLM.from_pretrained(HUGGINGFACE_MODEL_PATH).to(device)
    pt_model.eval()
except Exception as e:
    print(f"Error loading PyTorch model: {e}")
    pt_model = None

# --- 3. Load CTranslate2 Model ---
print("Loading CTranslate2 Model...")
try:
    ct2_translator = ctranslate2.Translator(CT2_MODEL_PATH, device=device)
except Exception as e:
    print(f"Error loading CTranslate2 model: {e}")
    ct2_translator = None

# --- Function for PyTorch Inference ---
def run_pytorch_inference():
    if not pt_model: return float('inf')

    # Tokenize inputs
    inputs = tokenizer(test_inputs, return_tensors="pt", padding=True, truncation=True).to(device)

    # Warm-up run
    pt_model.generate(**inputs, max_length=64, num_beams=1)

    start_time = time.time()
    for _ in range(NUM_RUNS):
        pt_model.generate(**inputs, max_length=64, num_beams=1)
    end_time = time.time()
    return (end_time - start_time) / NUM_RUNS # Average time per run

# --- Function for CTranslate2 Inference ---
def run_ct2_inference():
    if not ct2_translator: return float('inf')

    # Tokenize inputs and convert to CTranslate2 format
    inputs_tokens = [tokenizer.convert_ids_to_tokens(tokenizer.encode(text)) for text in test_inputs]

    # Warm-up run
    ct2_translator.translate_batch(inputs_tokens, max_decoding_length=64, beam_size=1)

    start_time = time.time()
    for _ in range(NUM_RUNS):
        ct2_translator.translate_batch(inputs_tokens, max_decoding_length=64, beam_size=1)
    end_time = time.time()
    return (end_time - start_time) / NUM_RUNS # Average time per run

# --- RUN BENCHMARKS ---
pt_time = run_pytorch_inference()
ct2_time = run_ct2_inference()

# ... (rest of the script up to the Model Size Calculation)

print("\n--- BENCHMARK RESULTS ---")

# --- FIXED MODEL SIZE CALCULATION ---
# 1. Define possible file names
pt_file_names = ["pytorch_model.bin", "model.safetensors"]
pt_size = 0.0

# 2. Check which file exists in the final_best_model directory
for file_name in pt_file_names:
    file_path = os.path.join(HUGGINGFACE_MODEL_PATH, file_name)
    if os.path.exists(file_path):
        pt_size = os.path.getsize(file_path) / (1024**2)
        print(f"PyTorch Model File Found: {file_name}")
        break

# 3. Calculate CTranslate2 Size
ct2_size = os.path.getsize(os.path.join(CT2_MODEL_PATH, "model.bin")) / (1024**2)
# --- END FIXED BLOCK ---

print(f"PyTorch Average Inference Time: {pt_time:.4f} seconds/run")
print(f"CTranslate2 Average Inference Time: {ct2_time:.4f} seconds/run")
print(f"PyTorch Model Size: {pt_size:.2f} MB")
print(f"CTranslate2 Model Size (INT8): {ct2_size:.2f} MB")

# Calculate Speed Gain and Size Reduction
if ct2_time != float('inf') and pt_time != float('inf') and ct2_time > 0:
    speed_gain_percent = ((pt_time - ct2_time) / ct2_time) * 100
    print(f"\n🚀 CTranslate2 Speed Gain: {speed_gain_percent:.2f}% faster than PyTorch")

size_reduction_percent = ((pt_size - ct2_size) / pt_size) * 100
print(f"💾 CTranslate2 Size Reduction: {size_reduction_percent:.2f}% reduction")
