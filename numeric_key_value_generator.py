import random
import string
import json
import os

def generate_kv_dataset(n, key_length, system=""):
    dataset = []
    keys = set()  # Set to store unique keys
    values = set()  # Set to store unique values

    for i in range(n):
        # Generate a unique alpha-numeric string of length key_length
        key = ''.join(random.choices(string.ascii_letters + string.digits, k=key_length))
        while key in keys:  # Ensure the key is unique
            key = ''.join(random.choices(string.ascii_letters + string.digits, k=key_length))
        keys.add(key)
        
        # Generate a unique value
        value = i
        while value in values:
            value += 1
        values.add(value)
        
        # Format the data in the fine-tuning format
        entry = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": key}, {"role": "assistant", "content": str(value)}]}

        dataset.append(entry)
    
    return dataset

key_lengths = [4, 8, 16, 32]
n_values = [100, 500, 1000, 5000, 10000]

# Generate datasets for each combination of key length and number of entries
for key_length in key_lengths:
    for n in n_values:
        dataset = generate_kv_dataset(n, key_length)

        # Create the "datasets" folder if it doesn't exist
        os.makedirs("datasets", exist_ok=True)

        # Save the dataset to a file with a name indicating the key length and number of entries in the "datasets" folder
        filename = os.path.join("datasets", f'{key_length}key-{n}n-untrained.jsonl')
        with open(filename, 'w') as f:
            for entry in dataset:
                f.write(json.dumps(entry) + "\n")

# Load the datasets and ensure no duplicate keys within each file
for key_length in key_lengths:
    for n in n_values:
        filename = os.path.join("datasets", f'{key_length}key-{n}n-untrained.jsonl')
        with open(filename, 'r') as f:
            dataset = [json.loads(line) for line in f]

        # Check for duplicate keys
        keys = set()
        for entry in dataset:
            key = entry["messages"][1]["content"]
            if key in keys:
                print(f"Duplicate key found: {key} in file {filename}")
            else:
                keys.add(key)