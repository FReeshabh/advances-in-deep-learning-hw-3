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
    
    rft_data = []

    for item in tqdm(dataset, desc="Generating RFT dataset"):
        question, correct_answer = item[0], item[1]
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
                break
    
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(rft_data, f, indent=2)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
