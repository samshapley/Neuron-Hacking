import openai
import json
from datetime import datetime
import time
from finetune_cost_estimator import estimate_finetune_cost

openai.api_key = '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'

class AI:
    def __init__(self, system="", model="gpt-3.5-turbo", openai_client=None):
        self.system = system
        self.openai = openai_client or openai
        self.messages = [{"role": "system", "content": system}]
        self.model = model
        self.load_costs()

    def load_costs(self):
        try: 
            with open('pricing.json', 'r') as f:
                self.costs = json.load(f)
        except FileNotFoundError:
            print("File not found.")

    def calculate_completion_cost(self, num_tokens, role="input"):
        cost_per_token = (self.costs[self.model][role + "_usage"] if self.model in self.costs and role + "_usage" in self.costs[self.model] else 0) / 1000  # Cost per 1000 tokens
        return num_tokens * cost_per_token

    def chat_completion(self, prompt, model=None, memories=True, temperature=None, max_tokens=None, json_mode=False, log_costs=False, seed=None):
        model = model or self.model
        self.messages.append({"role": "user", "content": prompt})
        
        response_format = {"type": "json_object"} if json_mode else {"type": "text"}

        start_time = time.time()  # Record the time before the API call

        completion = self.openai.chat.completions.create(
            model=model,
            messages=self.messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            # response_format=response_format,
        )

        end_time = time.time()  # Record the time after the API call
        runtime = end_time - start_time  # Calculate the runtime

        response_text = completion.choices[0].message.content

        self.messages.append({"role": "assistant", "content": response_text})

        if log_costs:
            # Calculate and log the cost after each call
            input_tokens = completion.usage.prompt_tokens
            output_tokens = completion.usage.completion_tokens

            # Calculate and log the cost
            input_cost = self.calculate_completion_cost(input_tokens, role="input")
            output_cost = self.calculate_completion_cost(output_tokens, role="output")

            self.log_completion_cost(input_cost, output_cost, input_tokens, output_tokens, model, runtime, completion.created)

        # If memories is False, reset the messages list after each call
        if not memories:
            self.messages = [{"role": "system", "content": self.system}]

        return response_text, self.messages

    def log_completion_cost(self, input_cost, output_cost, input_tokens, output_tokens, model, runtime, created):
        # Convert the Unix timestamp to a UTC datetime
        timestamp = datetime.utcfromtimestamp(created).strftime("%Y-%m-%d %H:%M:%S")

        # Load the existing log data
        try:
            with open('completion_costs_log.json', 'r') as f:
                log_data = json.load(f)
        except FileNotFoundError:
            log_data = []

        # Add the new log entry
        log_data.append({
            'timestamp': timestamp,
            'model': model,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'runtime': runtime  # Add the runtime to the log entry
        })

        # Save the updated log data
        with open('completion_costs_log.json', 'w') as f:
            json.dump(log_data, f)

class FineTuner:
    
    def __init__(self, openai_client=None):
        self.openai = openai_client or openai

    def upload_finetune_file(self, file_path):
        """
        Uploads a file for fine-tuning.

        Args:
            file_path (str): The path to the file to be uploaded.

        Returns:
            The uploaded file.
        """
        return self.openai.files.create(file=open(file_path, "rb"), purpose='fine-tune')

    def fine_tune_model(self, file_path, model_name, suffix, n_epochs, price_check=True, learning_rate_multiplier=None, batch_size=None):
        """
        Fine-tunes a model with the provided parameters.

        Args:
            file_path (str): The path to the file to be used for fine-tuning.
            model_name (str): The name of the model to be fine-tuned.
            suffix (str): The suffix to be added to the fine-tuned model's name.
            n_epochs (int): The number of epochs for which the model should be fine-tuned.
            price_check (bool, optional): Whether to check the estimated cost of fine-tuning before proceeding. Defaults to True.
            learning_rate_multiplier (float, optional): The multiplier for the learning rate. If not provided, OpenAI will use defaults/auto.
            batch_size (int, optional): The batch size for fine-tuning. If not provided, OpenAI will use defaults/auto.

        Returns:
            The fine-tuning job.
        """

        file = self.upload_finetune_file(file_path)

        estimated_cost = estimate_finetune_cost(file_path, n_epochs, model_name)['cost']

        if price_check:
            # Estimate the cost of fine-tuning
            
            print(f"Estimated cost of fine-tuning: ${estimated_cost}")

            # Check if the user wants to continue
            if input("Continue? (y/n): ") != "y":
                print("Exiting...")
                return

        # Construct hyperparameters dictionary based on provided inputs, if not provided OpenAI will use defaults/auto.
        hyperparameters = {}
        if n_epochs is not None:
            hyperparameters["n_epochs"] = n_epochs
        if learning_rate_multiplier is not None:
            hyperparameters["learning_rate_multiplier"] = learning_rate_multiplier
        if batch_size is not None:
            hyperparameters["batch_size"] = batch_size

        fine_tuning_job = self.openai.fine_tuning.jobs.create(
            training_file=file.id, 
            model=model_name, 
            suffix=suffix, 
            hyperparameters=hyperparameters
        )

        return fine_tuning_job, estimated_cost
    
    def list_finetuning_jobs(self):
        """
        Lists all fine-tuning jobs.
        
        Returns:
            A list of fine-tuning jobs.
        """
        return self.openai.fine_tuning.jobs.list()

    def retrieve_finetuning_job(self, job_id):
        """
        Retrieves a specific fine-tuning job.
        
        Args:
            job_id (str): The ID of the fine-tuning job to retrieve.
        
        Returns:
            The specified fine-tuning job.
        """
        return self.openai.fine_tuning.jobs.retrieve(job_id)

    def cancel_finetuning(self, job_id):
        """
        Cancels a specific fine-tuning job.
        
        Args:
            job_id (str): The ID of the fine-tuning job to cancel.
        
        Returns:
            The cancelled fine-tuning job.
        """
        return self.openai.fine_tuning.jobs.cancel(job_id)

    def list_finetuning_events(self, job_id, limit=2):
        """
        Lists events for a specific fine-tuning job.
        
        Args:
            job_id (str): The ID of the fine-tuning job to list events for.
            limit (int, optional): The maximum number of events to return. Defaults to 2.
        
        Returns:
            A list of events for the specified fine-tuning job.
        """
        return self.openai.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=limit)
    
    def delete_finetuned_model(self, model_id):
        """
        Deletes a specific fine-tuned model.
        
        Args:
            model_id (str): The ID of the fine-tuned model to delete.
        
        Returns:
            The deletion status of the fine-tuned model.
        """
        return self.openai.models.delete(model_id)
    
    def list_finetuned_models(self):
        """
        Lists all fine-tuned models that are not owned by OpenAI or system.
        
        Returns:
            A list of fine-tuned models.
        """
        all_models = self.openai.models.list()
        finetuned_models = [model for model in all_models.data if 'openai' not in model.owned_by.lower() and 'system' not in model.owned_by.lower()]
        return finetuned_models