import re
from tqdm import tqdm

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name_or_path = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = """Reason about the user request within <|think|> tokens, and only afterwards write your final response.
The expected format for your response is as follows:

<|think|>
Reasoning
<|think|>
Answer
"""

# Extracts thinking content and final model answer
def parse_reasoning_response(text: str) -> dict:
    pattern = r"<\|think\|>\s*(.*?)\s*<\|think\|>\s*(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return {"thinking_content": "", "response": text}
    return {"thinking_content": match.group(1).strip(), "response": match.group(2).strip()}

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

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype="auto")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

data = load_data()

print(len(data))

correct = 0
total = 0
pbar = tqdm(data)
for i, entry in enumerate(pbar):
    #print("-----------------------------------------")
    prompt = entry["prompt"]
    expected_answer = entry["answer"]

    text = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
        )
    generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    parsed_response = parse_reasoning_response(response_text)
    predicted_response = parsed_response["response"].strip()

    numbers = re.findall(r'-?\d+', predicted_response)
    predicted_answer = numbers[-1] if numbers else None

    if predicted_answer == expected_answer:
        correct += 1
    total += 1

    accuracy = correct / total * 100
    pbar.set_postfix(accuracy=f"{accuracy:.2f}%")

print(correct/total)
