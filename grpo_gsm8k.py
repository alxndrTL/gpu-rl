"""
Train a LLM with RL (GRPO) on the GSM8K dataset.

Adapted from https://github.com/minosvasilias/simple_grpo
(which in turn was adapted from https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb)
"""

# TODO : track GSM8k eval

import re
import json

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

SYSTEM_PROMPT = """Reason about the user request within <|think|> tokens, and only afterwards write your final response.
The expected format for your response is as follows:

<|think|>
Reasoning
<|think|>
Answer
"""

# ---------------- PARSING functions ----------------
# Extracts thinking content and final model answer
def parse_reasoning_response(text: str) -> dict:
    pattern = r"<\|think\|>\s*(.*?)\s*<\|think\|>\s*(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return {"thinking_content": "", "response": text}
    return {"thinking_content": match.group(1).strip(), "response": match.group(2).strip()}

def get_completion_content(completion: dict) -> str:
    return completion[0]["content"]

def parse_responses(completions: list[dict]) -> list[dict]:
    return [parse_reasoning_response(get_completion_content(c)) for c in completions]

# ---------------- REWARD functions ----------------
# Score formatting of reasoning content
def format_reasoning_reward(prompts, completions, answer, **kwargs) -> list[float]:
    parsed_responses = parse_responses(completions)
    rewards = [0.5 if r["thinking_content"] and r["response"] else 0.0 for r in parsed_responses]
    return rewards

# Score formatting of number (integer expected)
def format_number_reward(prompts, completions, answer, **kwargs) -> list[float]:
    parsed_responses = parse_responses(completions)
    rewards = [0.5 if r["response"].isdigit() else 0.0 for r in parsed_responses]
    return rewards

# Score accuracy of answer
#def accuracy_reward(prompts, completions, answer, **kwargs) -> list[float]:
#    parsed_responses = parse_responses(completions)
#    rewards = [2.0 if r["response"] == a else 0.0 for r, a in zip(parsed_responses, answer)]
#    return rewards

def accuracy_reward(prompts, completions, answer, **kwargs) -> list[float]:
    parsed_responses = parse_responses(completions)
    rewards = []
    for r, a in zip(parsed_responses, answer):
        response = r["response"].strip()
        numbers = re.findall(r'-?\d+', response)
        last_number = numbers[-1] if numbers else ""
        rewards.append(2.0 if last_number == str(a) else 0.0)
    return rewards

# Log rewards and example responses
def log_rewards(prompts, completions, answer, **kwargs):
    return 0
    rewards = {
        "accuracy": accuracy_reward(prompts, completions, answer),
        "format_number": format_number_reward(prompts, completions, answer),
        "format_reasoning": format_reasoning_reward(prompts, completions, answer),
    }
    example_response = get_completion_content(completions[0])
    example_parsed = parse_reasoning_response(example_response)
    example_answer = answer[0]
    example_prompt = prompts[0][-1]['content']
    print(
        f"-" * 50
        + f"\nExample prompt:\n{example_prompt}\n"
        + f"-" * 10
        + f"\nExample response:\n{example_response}\n"
        + f"-" * 10
        + f"\nExample answer:\n{example_answer}\n"
        + f"-" * 10
        + f"\nExample Correct?: {example_parsed['response'] == example_answer}\n"
        + f"-" * 10
        + f"\nRewards:\n{json.dumps(rewards, indent=2)}"
    )
    return 0

# ---------------- DATASET functions ----------------
# Extracts gsm8k answers from the dataset
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def load_data(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"] + "\nAfter your reasoning between <|think|>, answer with a number only."},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    return data

def main(training_args, model_args):
    data = load_data()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        processing_class=tokenizer,
        reward_funcs=[format_reasoning_reward,
                      format_number_reward,
                      accuracy_reward,
                      log_rewards],
        args=training_args,
        train_dataset=data,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    parser = TrlParser((GRPOConfig, ModelConfig))
    training_args, model_args = parser.parse_args_and_config()
    main(training_args, model_args)
