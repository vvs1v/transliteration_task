import os
import argparse
from datasets import load_dataset, DatasetDict, concatenate_datasets, features

# --- Global Schema Definition ---
# Defining the full schema ensures consistent loading from JSON files.
aksharantar_features_full = features.Features({
    'unique_identifier': features.Value('string'),
    'native word': features.Value('string'),
    'english word': features.Value('string'),
    'source': features.Value('string'),
    'score': features.Value('float') 
})

# --- Helper Functions ---

def clean_schema(ds_dict, lang_code):
    """Removes the problematic 'score' column from the dataset."""
    cleaned_dict = DatasetDict()
    for split, ds in ds_dict.items():
        if 'score' in ds.column_names:
            print(f"Removing 'score' column from the '{split}' split for {lang_code}.")
            # Using ds.remove_columns is safe even if a split is missing the column
            ds = ds.remove_columns(['score'])
        cleaned_dict[split] = ds
    return cleaned_dict

def format_for_seq2seq(example):
    """
    Formats the example into standard source/target columns for a multilingual Seq2Seq model.
    Applies the language token prefix to the source text.
    """
    # Assuming 'unique_identifier' starts with the 3-letter language code (e.g., 'hin')
    lang_code = example['unique_identifier'][:3]
    # Format: __hin__ Namaste
    example['source_text'] = f"__{lang_code}__ {example['english word']}"
    example['target_text'] = example['native word']
    return example

# --- Main Execution Function ---

def main(data_dir: str, target_langs: list, subset_size: int, output_dir: str):
    """
    Main function to orchestrate data loading, preprocessing, and saving.
    """
    if not os.path.exists(data_dir):
        print(f"FATAL ERROR: Data directory not found: {data_dir}")
        return

    multilingual_subsets = {'train': [], 'validation': [], 'test': []}
    val_subset_size = subset_size // 5
    test_subset_size = subset_size // 5

    # -----------------------------------------------------------
    # STEP 1: LOAD, SUBSET, AND SPLIT DATA PER LANGUAGE
    # -----------------------------------------------------------
    for lang_code in target_langs:
        data_files = {
            'train': os.path.join(data_dir, f'{lang_code}.zip'), 
            'validation': os.path.join(data_dir, f'{lang_code}.zip'), # Required by load_dataset
            'test': os.path.join(data_dir, f'{lang_code}.zip')         # Required by load_dataset
        }
        
        print(f"\n--- Loading and subsetting data for: {lang_code} ---")
        
        try:
            # Load as JSON, pointing directly to the local zip files
            lang_dataset = load_dataset(
                "json", 
                data_files=data_files,
                features=aksharantar_features_full
            )
        except Exception as e:
            print(f"FATAL ERROR: Failed to load {lang_code}. Check file path/integrity. Error: {e}")
            continue 

        lang_dataset = clean_schema(lang_dataset, lang_code)
        
        # When loading single zips via data_files, all content ends up in the 'train' split.
        full_ds = lang_dataset['train'] 
        
        # Calculate indices for the subsets
        train_start = 0
        train_end = subset_size
        val_start = train_end
        val_end = val_start + val_subset_size
        test_start = val_end
        test_end = test_start + test_subset_size

        # Select and append subsets
        multilingual_subsets['train'].append(full_ds.select(range(train_start, train_end)))
        multilingual_subsets['validation'].append(full_ds.select(range(val_start, val_end)))
        multilingual_subsets['test'].append(full_ds.select(range(test_start, test_end)))

        print(f"Loaded and subsetted {lang_code}: Train={train_end}, Val={val_subset_size}, Test={test_subset_size}")

    # Concatenate all language subsets
    if not multilingual_subsets['train']:
        print("Failed to load data for any language. Exiting.")
        return

    final_dataset_dict = DatasetDict({
        'train': concatenate_datasets(multilingual_subsets['train']),
        'validation': concatenate_datasets(multilingual_subsets['validation']),
        'test': concatenate_datasets(multilingual_subsets['test'])
    })

    # -----------------------------------------------------------
    # STEP 2: FORMAT DATA FOR SEQ2SEQ TRAINING
    # -----------------------------------------------------------
    print("\n--- Step 2: Formatting data for Seq2Seq ---")

    # Use os.cpu_count() for efficient parallel processing
    processed_dataset_dict = final_dataset_dict.map(
        format_for_seq2seq,
        num_proc=os.cpu_count() or 1
    )

    # Clean up columns before saving
    columns_to_keep = ['source_text', 'target_text']
    tokenization_ready_dataset = processed_dataset_dict.remove_columns([
        col for col in processed_dataset_dict['train'].column_names if col not in columns_to_keep
    ])

    print("SUCCESS! Data loading and formatting complete.")
    print(f"Final Multilingual Training Size: {len(tokenization_ready_dataset['train'])} examples.")
    print(f"Example Source: {tokenization_ready_dataset['train'][0]['source_text']}")
    print(f"Example Target: {tokenization_ready_dataset['train'][0]['target_text']}")

    # -----------------------------------------------------------
    # STEP 3: SAVE PROCESSED DATA
    # -----------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    tokenization_ready_dataset.save_to_disk(output_dir)
    print(f"\nSUCCESS! Processed data saved to: {os.path.abspath(output_dir)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process Aksharantar data for multilingual Seq2Seq training.")
    
    # Required arguments
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing the language ZIP files (e.g., D:/data/aksharantar)."
    )
    
    # Optional arguments with defaults
    parser.add_argument(
        "--target_langs", 
        type=str, 
        nargs='+', 
        default=['hin', 'ben', 'tam'], 
        help="List of 3-letter language codes to process (e.g., hin ben tam)."
    )
    
    parser.add_argument(
        "--subset_size", 
        type=int, 
        default=5000, 
        help="Number of examples to use for the training split PER LANGUAGE."
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="transliteration_data_ready", 
        help="Directory to save the final processed DatasetDict object."
    )

    args = parser.parse_args()
    
    main(args.data_dir, args.target_langs, args.subset_size, args.output_dir)
