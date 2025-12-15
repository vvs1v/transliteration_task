import gradio as gr
import ctranslate2
from transformers import AutoTokenizer
import os

# --- CONFIGURATION ---
# *** CRITICAL FIX: Set path to the root directory (where all files are located) ***
CT2_MODEL_PATH = "." 
TOKENIZER_CHECKPOINT = CT2_MODEL_PATH 

TARGET_LANGS = {
    "Hindi (Devanagari)": "<hin>",
    "Bengali (Bengali)": "<ben>",
    "Tamil (Tamil)": "<tam>"
}
# ---------------------

# --- 1. Load Optimized Model and Tokenizer ---
try:
    print("Loading optimized CTranslate2 model...")
    
    # Load tokenizer from the current directory (root)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_CHECKPOINT) 
    
    # Initialize the CTranslate2 Translator from the current directory (root)
    translator = ctranslate2.Translator(CT2_MODEL_PATH, device="cpu") # Using CPU for maximum stability
    print("Model and Translator loaded successfully.")
    
except Exception as e:
    # Log any loading failures clearly
    print(f"FATAL ERROR loading CTranslate2 components: {e}")
    translator = None
    tokenizer = None


def transliterate(roman_text, target_language_name):
    """
    Performs the transliteration using the CTranslate2 model.
    """
    if not translator or not tokenizer:
        return "Error: Model or Tokenizer not loaded. Check Space logs."

    # 1. Get the language tag
    target_lang_tag = TARGET_LANGS[target_language_name]
    
    # 2. Format the input with the required language tag
    input_text = f"{target_lang_tag} {roman_text}"
    
    # 3. Tokenize the input text into a list of string tokens (as required by CTranslate2)
    # The output of this is a list of integer IDs, which is then converted to tokens (strings)
    input_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(input_text, add_special_tokens=True))
    
    # 4. Perform the translation/transliteration
    results = translator.translate_batch(
        [input_tokens],
        max_decoding_length=64,
        beam_size=5, 
        target_prefix=[[target_lang_tag]], # Guide the model to output the target language
    )

    # 5. Decode the output tokens (with CRITICAL FIX)
    output_tokens = results[0].hypotheses[0] # This is a list of strings (tokens)
    
    # CRITICAL FIX: Convert CTranslate2 token strings back to integer IDs
    token_ids = tokenizer.convert_tokens_to_ids(output_tokens)
    
    # Decode the integer IDs
    decoded_output = tokenizer.decode(token_ids, skip_special_tokens=True)
    
    # 6. Clean up the language tag from the output (optional, but cleaner)
    return decoded_output.replace(target_lang_tag, "").strip()


# --- 2. Define the Gradio Interface ---
iface = gr.Interface(
    fn=transliterate,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter Romanized text here (e.g., Namaste aap kya kar rahe ho)", label="Romanized Input"),
        gr.Radio(
            choices=list(TARGET_LANGS.keys()),
            value="Hindi (Devanagari)",
            label="Target Indic Language"
        )
    ],
    outputs=gr.Textbox(label="Transliterated Output"),
    title="Multilingual Transliteration Demo (mT5 + CTranslate2)",
    # Remind the user about the model quality issue
    description="This demo uses the optimized CTranslate2 model. ATTENTION: The model requires re-training as output may be random due to numerical instability during initial training."
)

if __name__ == "__main__":
    iface.launch()
