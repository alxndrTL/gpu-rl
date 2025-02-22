from transformers import AutoTokenizer, AutoModelForCausalLM

#model_name_or_path = "Qwen/Qwen2.5-1.5B-Instruct"
model_name_or_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype="auto")