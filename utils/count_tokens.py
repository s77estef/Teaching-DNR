from transformers import AutoTokenizer
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
messages = (
    "<think>\nOkay\n</think>\n\n<answer>false</answer>"
)
tokens = tokenizer(messages, return_tensors="pt")
print(tokens["input_ids"].shape[1])
