import os
import time
import helpers as h
from ai import FineTuner

# Initialize the FineTuner
finetuner = FineTuner()

# Get the list of fine-tuned models
finetuned_models = finetuner.list_finetuned_models()

finetuned_model_names = set(model.id for model in finetuned_models)

# Specify the path to your dataset folder
dataset_folder = "datasets"

# Specify the base model you want to fine-tune
model_name = "gpt-3.5-turbo"

# Specify the list of epochs for fine-tuning
epochs = [1, 2, 4, 8, 16, 32, 50, 51]  # Add your desired epochs here

# Get the list of datasets
datasets = sorted(os.listdir(dataset_folder), key=h.custom_sort)

# just do the second dataset
datasets = datasets[8:9]

accumulated_cost = 0

# Parse the finetuned_model_names to get the base name and the epoch number for continuing fine-tuning if the loop is interrupted
finetuned_models_dict = {}
for model_name in finetuned_model_names:
    # Split the model name by ':'
    parts = model_name.split(':')
    # The base name is the part before 'ep'
    base_name = parts[3].split('-ep-')[0]
    # The epoch number is the part after 'ep'
    epoch = int(parts[3].split('-ep-')[1].split(':')[0])
    # Add the epoch number to the list of epochs for the base name
    if base_name in finetuned_models_dict:
        finetuned_models_dict[base_name].append(epoch)
    else:
        finetuned_models_dict[base_name] = [epoch]

# Iterate over each dataset
for dataset in datasets:
    # Set to base model for the first fine-tuning job
    model_name = "gpt-3.5-turbo"

    total_epochs_trained = 0
    
    # Get the base name of the dataset (excluding the extension)
    base_name = os.path.splitext(dataset)[0]

    # Specify the path to the dataset
    file_path = os.path.join(dataset_folder, dataset)

    # Initialize the previous epoch to 0
    prev_epoch = 0

    n_epochs = 0

    # Iterate over each epoch in the list
    for epoch in epochs:
        if base_name in finetuned_models_dict and epoch in finetuned_models_dict[base_name]:
            print(f"Model {base_name}-ep-{epoch} already exists. Skipping...")
            # Update the model name to the fine-tuned model for the next fine-tuning job
            model_name = next((model for model in finetuned_model_names if f"{base_name}-ep-{epoch}" in model), model_name)
            # Update the previous epoch to the current epoch
            prev_epoch = epoch
            continue

        n_epochs = epoch - prev_epoch
        total_epochs_trained += n_epochs
        
        print(f"Fine-tuning {model_name} on {base_name} for {n_epochs} epochs...")
        print(f"Total epochs trained so far: {total_epochs_trained}")

        # Construct the suffix for the fine-tuned model
        suffix = f"{base_name}-ep-{epoch}"

        if any(suffix in model_name for model_name in finetuned_model_names):
            print(f"Model {suffix} already exists. Skipping...")
            continue

        # Calculate the number of epochs for this fine-tuning job
        n_epochs = epoch - prev_epoch

        # Fine-tune the model
        finetuning_job, estimated_cost = finetuner.fine_tune_model(file_path, model_name, suffix, n_epochs, price_check=False)

        # Add the estimated cost to the accumulated cost
        accumulated_cost += estimated_cost

        print(f"Estimated cost for this fine-tuning: {estimated_cost}")
        print(f"Accumulated cost so far: {accumulated_cost}")

        # Wait for the fine-tuning job to complete
        time.sleep(120)
        while True:
            # Retrieve the fine-tuning job
            job = finetuner.retrieve_finetuning_job(finetuning_job.id)

            print("Checking fine-tuning status...")

            # If the job has succeeded, break the loop
            if job.status == "succeeded":
                # Update the model name to the fine-tuned model for the next fine-tuning job
                model_name = job.fine_tuned_model
                break
            else:
                print("Waiting for fine-tuning to complete...")

            # If the job has not yet succeeded, wait for a while before checking again
            time.sleep(120)

        # Update the previous epoch to the current epoch
        prev_epoch = epoch