import tiktoken
import json

def load_pricing():
    with open('pricing.json', 'r') as file:
        return json.load(file)

def total_tokens_from_messages(messages, encoding):
    num_tokens = 0
    for message in messages:
        num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def estimate_finetune_cost(dataset_path, epochs, model_name):
    # Load the dataset
    dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))

    MAX_TOKENS_PER_EXAMPLE = 4096

    # Get the encoding for the specific model
    encoding = tiktoken.encoding_for_model(model_name)

    # Calculate tokens and examples
    convo_lens = [total_tokens_from_messages(ex["messages"], encoding) for ex in dataset]
    num_tokens = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)

    # Load the pricing information
    pricing_info = load_pricing()

    # Extract the base model name from the fine-tuned model name
    base_model_name = model_name.split(":")[1] if "ft:" in model_name else model_name

    # Get the pricing for the specific model
    model_pricing = pricing_info.get(base_model_name)

    # Calculate the cost
    if model_pricing:
        cost_per_1000_tokens = model_pricing['training']
        total_training_cost = round((num_tokens / 1000) * epochs * cost_per_1000_tokens, 2)
    else:
        total_training_cost = None

    # Return a dictionary with the inputs and the cost
    return {
        "dataset_path": dataset_path,
        "epochs": epochs,
        "model_name": base_model_name,
        "cost": total_training_cost,
    }
