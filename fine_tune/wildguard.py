from datasets import load_dataset


def count_adversarial_labels(dataset, split_name):
    values = dataset["adversarial"]
    true_count = sum(1 for value in values if str(value).lower() == 'true')
    false_count = sum(1 for value in values if str(value).lower() == 'false')
    print(f"{split_name} adversarial counts -> true: {true_count}, false: {false_count}")

def count_harm_labels(dataset, split_name):
    values = dataset["prompt_harm_label"]
    harmful_count = sum(1 for value in values if str(value).lower() == 'harmful')
    unharmful_count = sum(1 for value in values if str(value).lower() == 'unharmful')
    print(f"{split_name} harm counts -> harmful: {harmful_count}, unharmful: {unharmful_count}")



def count_adversarial_harm(dataset, split_name):
    combos = {
        ("true", "harmful"): 0,
        ("true", "unharmful"): 0,
        ("false", "harmful"): 0,
        ("false", "unharmful"): 0,
    }
    for adv, harm in zip(dataset["adversarial"], dataset["prompt_harm_label"]):
        key = (str(adv).lower(), str(harm).lower())
        if key not in combos:
            combos[key] = 0
        combos[key] += 1
    print(f"{split_name} adversarial/prompt_harm_label counts:")
    for key, count in combos.items():
        print(f"  adversarial={key[0]}, prompt_harm_label={key[1]} -> {count}")


# features: ['prompt', 'adversarial', 'response', 'prompt_harm_label', 'response_refusal_label', 'response_harm_label', 'subcategory']
# num_rows: 86759
# train adversarial counts -> true: 40956, false: 45803
# train harm counts -> harmful: 46216, unharmful: 40543
train = load_dataset('allenai/wildguardmix', 'wildguardtrain', split=f"train[:100%]", columns=["prompt", "adversarial", "prompt_harm_label"])

# features: ['prompt', 'response', 'adversarial', 'prompt_harm_label', 'response_refusal_agreement', 'response_refusal_label', 'response_harm_label', 'subcategory', 'prompt_harm_agreement', 'response_harm_agreement']
# num_rows: 1725
# test adversarial counts -> true: 810, false: 915
# test harm counts -> harmful: 754, unharmful: 945
test = load_dataset('allenai/wildguardmix', 'wildguardtest', split=f"test[:100%]", columns=["prompt", "adversarial", "prompt_harm_label"])

count_adversarial_labels(train, 'train')
count_adversarial_labels(test, 'test')
count_harm_labels(train, 'train')
count_harm_labels(test, 'test')
print(test)


"""
  adversarial=true, prompt_harm_label=harmful -> 20567
  adversarial=true, prompt_harm_label=unharmful -> 20389
  adversarial=false, prompt_harm_label=harmful -> 25649
  adversarial=false, prompt_harm_label=unharmful -> 20154
"""
train_adv_harm = load_dataset('allenai/wildguardmix', 'wildguardtrain', split=f"train[:100%]", columns=["prompt", "adversarial", "prompt_harm_label"])
count_adversarial_harm(train_adv_harm, 'train')
