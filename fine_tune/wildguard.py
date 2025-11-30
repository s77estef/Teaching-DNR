from datasets import load_dataset

# features: ['prompt', 'adversarial', 'response', 'prompt_harm_label', 'response_refusal_label', 'response_harm_label', 'subcategory']
# num_rows: 86759
train = load_dataset('allenai/wildguardmix', 'wildguardtrain')["train"]

# features: ['prompt', 'response', 'adversarial', 'prompt_harm_label', 'response_refusal_agreement', 'response_refusal_label', 'response_harm_label', 'subcategory', 'prompt_harm_agreement', 'response_harm_agreement']
# num_rows: 1725
test = load_dataset('allenai/wildguardmix', 'wildguardtest')["test"]

print(test)

