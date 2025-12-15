# transliteration_task
Transliteration Hindi, bengali, tamil


# Multilingual Transliteration Model (mT5 + CTranslate2)

This project implements a sequence-to-sequence model for multilingual transliteration across three Indic languages, optimized for high-speed inference using CTranslate2, and deployed on Hugging Face Spaces.

**Live Demo Link:** **[FILL IN LATER: INSERT YOUR HUGGING FACE SPACES URL HERE]**

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


