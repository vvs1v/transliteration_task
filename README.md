# transliteration_task
Transliteration Hindi, bengali, tamil


# Multilingual Transliteration Model (mT5 + CTranslate2)

This project implements a sequence-to-sequence model for multilingual transliteration across three Indic languages, optimized for high-speed inference using CTranslate2, and deployed on Hugging Face Spaces.

**Live Demo Link:** **(https://huggingface.co/spaces/vaibhavsinghal2000/transliteration2)**

---

## 1. Project Objective and Architecture

The objective was to build, optimize, and deploy a system that converts Romanized text (e.g., "Namaste") into the native script of three target Indic languages (e.g., "नमस्ते").

* **Model Architecture:** Fine-tuned mT5-Small (Sequence-to-Sequence)
* **Target Languages:** Hindi (HIN), Bengali (BEN), and Tamil (TAM)
* **Optimization Engine:** CTranslate2 (INT8 Quantization)
* **Deployment Framework:** Gradio on Hugging Face Spaces

---

## 2. Setup and Installation

### A. Environment Setup

The primary environment used for training and optimization was a **[Colab T4 GPU]**.

### B. Requirements

Install the necessary Python packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 3. Preprocessing and Training 

### A. Preprocess data

From https://huggingface.co/datasets/ai4bharat/Aksharantar/tree/main download the .zip language dataset files.

run data.py file with arguments like data_dir, target_language, subset_size, output_dir

```python
python data.py --data_dir "path_to_data_files" --subset_size 10000 --target_langs hin ben tam mal
```
### B. Train

Change the variables accordingly provide the path to preporcessed data and output dir in the file and run that

---
## 4. Optimization

run ctranslate2_convert to convert the trained model into int8 that may reduce the model size and decrease the inference time

---
## 5. BenchMarking

run benchmark.py file to compare the final_best_model to the ctranslate2 optimized model

---
## 6. Deployment

Create hugginface account and a space using gradio
Upload the model files like config.json,model.bin,requirements.txt,shared_vocabulary.json,special_tokens_map.json,spiece.model,tokenizer_config.json and run.py files.
It will build and deploy then the model can be tested.



## Training Process and Hyperparameters

The model was fine-tuned from the pre-trained `google/mt5-small` checkpoint on the Aksharantar dataset.

### A. Dataset Preprocessing

* **Dataset Source:** `ai4bharat/Aksharantar`
* **Data Size:** Approximately **[5000]** examples per language were sampled for training.
* **Preprocessing:** Inputs were prefixed with language tokens (e.g., `<hin>`) and tokenized.
  

### B. Final Stable Training Parameters

| Parameter | Value (Stable) | Note |
| :--- | :--- | :--- |
| **Base Model** | `google/mt5-small` | Sequence-to-sequence architecture. |
| **`num_train_epochs`** | **[3]** | Increased for stable convergence. |
| **`learning_rate`** | **$3 \mathrm{e}-4$** | **Crucial Fix:** Reduced for numerical stability after initial failure. |
| **`per_device_train_batch_size`** | **[FILL IN LATER: 16]** | Adjusted based on GPU memory. |
| **Precision** | **`fp16=False`** | **Crucial Fix:** Disabled 16-bit precision to prevent `nan` loss. |
| **Optimizer** | AdamW | Standard optimizer for fine-tuning. |

### C. Evaluation Metrics (After Stable Re-training)

| Metric | Result |
| :--- | :--- |
| **Validation Loss** | **[3.6]** |
| **Character Error Rate (CER)** | **[.75]** |
| **Word Error Rate (WER)** | **[.98]** |

---

## 4. Model Optimization and Benchmarking

The trained PyTorch model was optimized using CTranslate2 for faster, quantized inference.

### A. Conversion Details

* **Input Model:** PyTorch checkpoint
* **Optimization Tool:** CTranslate2
* **Quantization:** `int8`

### B. Benchmarking Results

| Metric | PyTorch Model (Original) | CTranslate2 (INT8) |
| :--- | :--- | :--- |
| **Model Size** | **$1.15 \text{ GB}$** (Approx.) | **$288 \text{ MB}$** |
| **Inference Latency** (Average per request) | **[14 ms]** | **[ 04 ms]** |
| **Speed Gain Percentage** | N/A | **[255%]** Faster |
| **Quality Loss** | N/A | **More retraiing needed ** loss in accuracy |
<img width="1302" height="489" alt="Screenshot 2025-12-12 092016" src="https://github.com/user-attachments/assets/ec8a91a0-6220-4c76-951d-13ffc1e636ba" />
<img width="1302" height="489" alt="Screenshot 2025-12-12 092016" src="https://github.com/user-attachments/assets/5e1b9db4-947c-4aea-bde3-103c59d7ca24" />


---
|


