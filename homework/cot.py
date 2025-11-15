from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        system_prompt = (
            "You are a helpful assistant for all unit conversions. Be concise. "
            "You must follow the SI (decimal) standard for bit size conversions: 1 KB = 1000 bytes, 1 MB = 1000 KB, 1 GB = 1000 MB. "
            "Think step-by-step and then provide the final numerical answer "
            "enclosed in <answer></answer> tags."
)

# Example 1: m -> km (division)
        user_1 = "How many kilometers are in 2500 meters?"
        asst_1 = (
            "1. 1000 meters = 1 kilometer.\n"
            "2. 2500 / 1000 = 2.5.\n"
            "<answer>2.5</answer>"
        )

        # Example 2: ml -> l (division, different units)
        user_2 = "How many liters are in 800 milliliters?"
        asst_2 = (
            "1. 1000 milliliters = 1 liter.\n"
            "2. 800 / 1000 = 0.8.\n"
            "<answer>0.8</answer>"
        )

        # Example 3: kg -> g (multiplication)
        user_3 = "How many grams are in 2kg?"
        asst_3 = (
            "1. 1 kilogram = 1000 grams.\n"
            "2. 2 * 1000 = 2000.\n"
            "<answer>2000</answer>"
        )
        user_4 = "How many meters are in 300 centimeters?"
        asst_4 = (
            "1. 100 centimeters = 1 meter.\n"
            "2. 300 / 100 = 3.\n"
            "<answer>3</answer>"
        )
        user_5 = "How many MB are there per 9 GB?"
        asst_5 = (
            "1. 1 GB = 1000 MB.\n"
            "2. 9 * 1000 = 9000.\n"
            "<answer>9000</answer>"
        )

        chat_messages = [
            {"role": "system", "content": system_prompt},
            
            {"role": "user", "content": user_1},
            {"role": "assistant", "content": asst_1},
            
            {"role": "user", "content": user_2},
            {"role": "assistant", "content": asst_2},
            
            {"role": "user", "content": user_3},
            {"role": "assistant", "content": asst_3},

            {"role": "user", "content": user_4},
            {"role": "assistant", "content": asst_4},

            {"role": "user", "content": user_5},
            {"role": "assistant", "content": asst_5},


            {"role": "user", "content": question}
        ]

        return self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True
        )


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
