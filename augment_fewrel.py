import json
from openai import OpenAI
import time
import random
from collections import defaultdict
import os

client = OpenAI(
    api_key="",
    base_url="https://api.chatanywhere.tech/v1"
)


def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_relation_statistics(data):
    relation_stats = defaultdict(list)
    for item in data:
        relation = item.get("relation", "unknown")
        relation_stats[relation].append(item)
    return relation_stats

def generate_prompt_for_relation(relation, examples, num_examples=3):
    examples_text = ""
    for i, example in enumerate(examples[:num_examples]):
        tokens = " ".join(example["token"])
        head_entity = example["h"][0]
        tail_entity = example["t"][0]
        examples_text += f"Example {i+1}:\n"
        examples_text += f"Sentence: {tokens}\n"
        examples_text += f"Head Entity: {head_entity}\n"
        examples_text += f"Tail Entity: {tail_entity}\n"
        examples_text += f"Relation: {relation}\n\n"

    prompt = f""
You are a data augmentation expert. Please generate 20 new training examples for the relation type: "{relation}".

def call_gpt_api(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates high-quality training data for relation extraction.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=4000,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return None


def parse_generated_examples(text, relation):
    try:
        examples = json.loads(text)
        valid_examples = []
        for example in examples:
            if isinstance(example, dict) and "tokens" in example and "h" in example and "t" in example:
                example["relation"] = relation
                valid_examples.append(example)
        return valid_examples

    except json.JSONDecodeError:
        try:
            start = text.find("[")
            end = text.rfind("]") + 1
            if start != -1 and end != -1:
                json_text = text[start:end]
                examples = json.loads(json_text)
                return parse_generated_examples(json.dumps(examples), relation)
        except:
            pass
        return []


def augment_dataset(original_data):
    relation_stats = get_relation_statistics(original_data)
    print(f"there are {len(relation_stats)} different relation types")
    augmented_data = original_data.copy()
    total_generated = 0
    for relation, examples in relation_stats.items():
        if len(examples) == 0:
            continue
        prompt = generate_prompt_for_relation(relation, examples)
        response_text = call_gpt_api(prompt)
        if response_text:
            new_examples = parse_generated_examples(response_text, relation)
            if new_examples:
                augmented_data.extend(new_examples)
                total_generated += len(new_examples)
        time.sleep(2)

    print(f"\ndata complete!")
    print(f"origin_data: {len(original_data)}")
    print(f"total_generated: {total_generated}")
    print(f"augment_data: {len(augmented_data)}")
    return augmented_data


def main():
    input_file = "train_wiki.json"
    output_file = "train_wiki1.json"
    raw = load_dataset(input_file)
    if isinstance(raw, dict):
        original_data = []
        for relation_id, samples in raw.items():
            for sample in samples:
                if "tokens" in sample:
                    sample["token"] = sample["tokens"]
                    del sample["tokens"]
                sample["relation"] = relation_id
                original_data.append(sample)
    else:
        raise TypeError("error format！")
    test_item = original_data[0]
    print(f"{len(original_data)} ")
    final_data = augment_dataset(original_data)
    save_dataset(final_data, output_file)
    print(f"{output_file}")

if __name__ == "__main__":
    main()
