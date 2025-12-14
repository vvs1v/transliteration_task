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

The primary environment used for training and optimization was a **[FILL IN LATER: GPU TYPE, e.g., Colab T4/A100 GPU]**.

### B. Requirements

Install the necessary Python packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
