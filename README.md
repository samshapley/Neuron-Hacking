# Neuron-Hacking

Finetuning LLMs to act as key-value stores.

https://api.wandb.ai/files/samuel-shapley/images/projects/38392022/408210be.png![image](https://github.com/samshapley/Neuron-Hacking/assets/93387313/d7be1983-4049-4200-8490-4d0878c7bc84)

<hr>

This repository contains base code that can be used to understand the research on fine-tuning Language Learning Models (LLMs) to act as key-value stores. It is not a complete project, but it provides a foundation for further exploration and experimentation.

The research report from the results this code returned can be found here:
<a href="google.com">Neuron Hacking Report on Weights and Biases</a>
<hr>

The necessary files to repeat the experiments outlined in the report are below:

1. `ai.py`: This file contains the AI and FineTuner classes. The AI class is used to interact with OpenAI's API, calculate the cost of completions, and log the cost. The FineTuner class is used to fine-tune models, upload files for fine-tuning, retrieve and cancel fine-tuning jobs, and manage fine-tuned models. This is a highly reusable class for other implementations if you desire.

2. `finetune_cost_estimator.py`: This script is used to estimate the cost of fine-tuning a model. It calculates the number of tokens in a dataset and uses the pricing information to estimate the cost. It doesn't work very well. For some reason my costs were signficantly higher than this suggested.

3. `numeric_key_value_generator.py`: This script generates datasets for fine-tuning. Each dataset consists of unique keys and corresponding values. The keys are alpha-numeric strings of a specified length, and the values are unique integers.

4. `numeric_finetuning_loop.py`: This script automates the process of fine-tuning a model on multiple datasets for multiple epochs. It checks if a fine-tuned model already exists before starting a new fine-tuning job, and it waits for a fine-tuning job to complete before starting the next one.

5. `finetune_progress_grid.py`: This script visualizes the progress of fine-tuning jobs. It creates a grid where the rows represent different datasets and the columns represent different epochs. The cells in the grid are colored based on whether a fine-tuning job for the corresponding dataset and epoch has been completed.

6. `numeric_recall_testing_loop.py`: This script tests the ability of fine-tuned models to recall values. It iterates over each item in a dataset and uses the model to recall the value corresponding to the unique key. The recalled values are then saved to a JSON file.

Please note that this code is intended for research purposes and should be adapted to fit your specific needs and constraints.
