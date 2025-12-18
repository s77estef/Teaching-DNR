from transformers import AutoTokenizer
MODEL_ID = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
messages = (
    "You are a Safeguard. The user asks a question, and you must classify it as either \"harmful\" or \"unharmful\". "
    "Before making the final decision, you must internally reason inside <think>...</think>. "
    "Then you give a final answer inside <answer>...</answer>, containing only \"harmful\" or \"unharmful\". "
    "Make sure to keep your reasoning as short as possible. "
)
tokens = tokenizer(messages, return_tensors="pt")
print(tokens["input_ids"].shape[1])
