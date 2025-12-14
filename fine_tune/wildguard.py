from datasets import load_dataset


def count_adversarial_labels(dataset, split_name):
    values = dataset["adversarial"]
    true_count = sum(1 for value in values if str(value).lower() == 'true')
    false_count = sum(1 for value in values if str(value).lower() == 'false')
    print(f"{split_name} adversarial counts -> true: {true_count}, false: {false_count}")


# features: ['prompt', 'adversarial', 'response', 'prompt_harm_label', 'response_refusal_label', 'response_harm_label', 'subcategory']
# num_rows: 86759
# train adversarial counts -> true: 40956, false: 45803
train = load_dataset('allenai/wildguardmix', 'wildguardtrain', split=f"train[:100%]", columns=["prompt", "adversarial"])

# features: ['prompt', 'response', 'adversarial', 'prompt_harm_label', 'response_refusal_agreement', 'response_refusal_label', 'response_harm_label', 'subcategory', 'prompt_harm_agreement', 'response_harm_agreement']
# num_rows: 1725
# test adversarial counts -> true: 810, false: 915
test = load_dataset('allenai/wildguardmix', 'wildguardtest', split=f"test[:100%]", columns=["prompt", "adversarial"])

count_adversarial_labels(train, 'train')
count_adversarial_labels(test, 'test')
print(test)
