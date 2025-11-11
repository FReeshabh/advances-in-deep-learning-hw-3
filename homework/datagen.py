import json
import math
from pathlib import Path
from tqdm import tqdm
from .cot import CoTModel
from .data import Dataset, is_answer_valid


def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    dataset = Dataset("train")
    try:
        model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    except:
        model = CoTModel()
    
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing data if checkpoint exists
    rft_data = []
    processed_questions = set()
    if output_path.exists():
        try:
            with open(output_path, 'r') as f:
                rft_data = json.load(f)
                processed_questions = {item[0] for item in rft_data}
            print(f"Resuming: Found {len(rft_data)} existing entries")
        except (json.JSONDecodeError, IndexError):
            print("Warning: Could not load checkpoint, starting fresh")

    for item in tqdm(dataset, desc="Generating RFT dataset"):
        question, correct_answer = item[0], item[1]
        
        # Skip if already processed
        if question in processed_questions:
            continue
        
        # Use the CoT chat prompt tt's just running reao elicit step-by-step reasoning and <answer> tags
        # 1. Create the full, correct prompt string. (You were right!)
        prompt = model.format_prompt(question)
        
        # 2. Manually tokenize this string and send to the GPU
        inputs = model.tokenizer(prompt, return_tensors="pt").to(model.model.device)
        
        # 3. Call the *underlying* transformers model.generate()
        # This bypasses the buggy `batched_generate` wrapper.
        # We must set temperature > 0 and do_sample=True
        outputs = model.model.generate(
            **inputs,
            num_return_sequences=oversample,
            temperature=temperature,
            max_new_tokens=128, # Set a reasonable max length
            pad_token_id=model.tokenizer.eos_token_id,
            do_sample=True 
        )

        # 4. Decode the results, skipping the prompt text
        prompt_len = inputs["input_ids"].shape[1]
        decoded_outputs = model.tokenizer.batch_decode(
            outputs[:, prompt_len:], 
            skip_special_tokens=True
        )
        
        # 5. Loop through the clean list of completions
        for completion in decoded_outputs:        
            parsed_answer = model.parse_answer(completion)
            if not math.isnan(parsed_answer) and is_answer_valid(parsed_answer, correct_answer):
                rft_data.append([question, correct_answer, completion])
                processed_questions.add(question)
                # Save checkpoint after each successful addition
                with open(output_path, 'w') as f:
                    json.dump(rft_data, f, indent=2)
                break
    
    # Final save
    with open(output_path, 'w') as f:
        json.dump(rft_data, f, indent=2)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
