from ai import FineTuner

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import helpers as h

# Initialize the FineTuner
finetuner = FineTuner()

# Get the list of fine-tuned models
finetuned_models = finetuner.list_finetuned_models()

# Parse the model names to get the dataset and epoch information
datasets = []
epochs = []
for model in finetuned_models:
    parts = model.id.split(':')
    base_name = parts[3].split('-ep-')[0]
    epoch = int(parts[3].split('-ep-')[1].split(':')[0])
    datasets.append(base_name)
    epochs.append(epoch)

# Get the list of all datasets from the directory
all_datasets = sorted(os.listdir("datasets"), key=h.custom_sort)
all_datasets = [os.path.splitext(dataset)[0] for dataset in all_datasets]

# Create a DataFrame to represent the grid
df = pd.DataFrame(index=all_datasets, columns=sorted(list(set(epochs))))
df = df.fillna(0)  # Fill NaN values with 0

# Populate the DataFrame with the parsed information
for dataset, epoch in zip(datasets, epochs):
    if dataset in df.index:
        df.loc[dataset, epoch] = 1

plt.figure(figsize=(10, 10))

# Use 'imshow' to visualize the DataFrame as before
plt.imshow(df, cmap='RdYlGn', interpolation='none')

# Set the major ticks at the edge of the cells
plt.xticks(ticks=np.arange(len(df.columns)), labels=df.columns)
plt.yticks(ticks=np.arange(len(df.index)), labels=df.index)

# Set the minor ticks to be in the middle of the cells
plt.gca().set_xticks(np.arange(-.5, len(df.columns)), minor=True)
plt.gca().set_yticks(np.arange(-.5, len(df.index)), minor=True)

# Draw the grid for the minor ticks only
plt.grid(which='minor', color='k', linestyle='-', linewidth=2)

# Hide the grid for the major ticks
plt.grid(which='major', color='k', linestyle='', linewidth=0)

plt.xlabel("Epochs")
plt.ylabel("Datasets")


plt.savefig("plots/finetuning--progress-grid-5.png", dpi=300, bbox_inches='tight')


# Display the plot
plt.show()