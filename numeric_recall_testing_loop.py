import os
import json
from ai import AI, FineTuner
from tqdm import tqdm
import helpers as h

# Initialize the FineTuner and AI
finetuner = FineTuner()
ai = AI()

# Specify the path to your dataset folder and the output folder
dataset_folder = "datasets"
output_folder = "value_recall_tests"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get the list of fine-tuned models
models = finetuner.list_finetuned_models()

# Get the list of datasets
datasets = sorted(os.listdir(dataset_folder), key=h.custom_sort)

# Specify the temperatures to test
temperatures = [0.0, 0.5, 1.0, 1.5]

# Iterate over each dataset
for dataset_file in datasets:
    # Initialize a dictionary to store the test results
    results = {}

    # Extract the dataset name from the file name
    dataset_name = os.path.splitext(dataset_file)[0]

    # Load the dataset
    with open(os.path.join(dataset_folder, dataset_file), "r") as f:
        dataset = [json.loads(line) for line in f]

    # Iterate over each model
    for model in models:
        # Extract the dataset name and epoch number from the model name
        model_dataset_name, epoch = model.id.split(":")[3].rsplit('-ep-', 1)

        # Skip the model if it's not for the current dataset
        if model_dataset_name != dataset_name:
            continue

        # Initialize a dictionary to store the test results
        results = {}

        # Load existing results if they exist
        results_file_path = os.path.join(output_folder, dataset_name + "-recall" + ".json")
        if os.path.exists(results_file_path):
            with open(results_file_path, "r") as f:
                results = json.load(f)

        # Get the first key in the dictionary, we can use this to check completion as if the first key has the temp+epoch combo, then all the others will too.
        if results:
            first_key = next(iter(results)) 
        else:
            first_key = None


        for temp in temperatures:
            if first_key is not None and str(epoch) in results[first_key]["recalled_values"]["epochs"] and str(temp) in results[first_key]["recalled_values"]["epochs"][str(epoch)]["temperatures"]:
                print(f"Results for {dataset_name} at epoch {epoch}, temperature {temp} already exist. Skipping...")
                continue

            print(f"Testing {model.id} on {dataset_name} at epoch {epoch}, temperature {temp}...")

            # Iterate over each item in the dataset
            for item in tqdm(dataset):
                # Get the unique key and the true value
                unique_key = item["messages"][1]["content"]
                true_value = item["messages"][2]["content"]

                # Skip if the result for this item at this temperature already exists
                if unique_key in results and "epochs" in results[unique_key]["recalled_values"] and epoch in results[unique_key]["recalled_values"]["epochs"] and "temperatures" in results[unique_key]["recalled_values"]["epochs"][epoch] and temp in results[unique_key]["recalled_values"]["epochs"][epoch]["temperatures"]:
                    continue

                # Use the model to recall the value
                recalled_value, _ = ai.chat_completion(unique_key, model=model.id, memories=False, temperature=temp, log_costs=False)

                # If the unique key is not in the results yet, add it
                if unique_key not in results:
                    results[unique_key] = {
                        "true_value": true_value,
                        "recalled_values": {
                            "epochs": {}
                        }
                    }

                # If the epoch is not in the recalled_values yet, add it
                if epoch not in results[unique_key]["recalled_values"]["epochs"]:
                    results[unique_key]["recalled_values"]["epochs"][epoch] = {
                        "temperatures": {}
                    }

                # Add the recalled value at the current temperature to the results
                results[unique_key]["recalled_values"]["epochs"][epoch]["temperatures"][temp] = recalled_value

            # Save the results to a JSON file
            with open(results_file_path, "w") as f:
                json.dump(results, f)