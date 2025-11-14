from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from peft import LoraConfig, TaskType, get_peft_model

from .base_llm import BaseLLM
from .data import Dataset
from .sft import TokenizedDataset, test_model


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str,
    num_train_epochs: int = 10,
    **kwargs,
):
    base_model = BaseLLM()

    lora_rank = kwargs.pop("lora_rank", 32)
    lora_alpha = kwargs.pop("lora_alpha", lora_rank * 4)

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(base_model.model, config)
    # model.enable_input_require_grads()
    # model.print_trainable_parameters()
    # model.print_trainable_parameters()

    # if torch.cuda.is_available():
    model.enable_input_require_grads()

    print("Trainable parameters:")
    model.print_trainable_parameters()

    train_data = Dataset("rft") #the dataset just generated from datagen.py
    tokenizer = base_model.tokenizer

    def format_example(prompt: str, correct_answer: float, completion: str):
        completion_text = completion.strip()
        if "<answer>" not in completion_text:
            completion_text = f"{completion_text} <answer>{correct_answer:g}</answer>"
        return {"question": prompt, "answer": completion_text}

    tokenized_train_data = TokenizedDataset(tokenizer, train_data, format_example)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        per_device_train_batch_size=32,
        num_train_epochs=num_train_epochs,
        learning_rate=2e-4,
        warmup_steps=100,
        weight_decay=0.01,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=False,
        **kwargs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
