## Design to evaluate a pre-trained model before any fine-tuning
```python
# Evaluate the pre-trained model before fine-tuning

# Instantiate your MetricsCalculator
metrics_calculator = MetricsCalculator()

# Evaluate the pre-trained model before fine-tuning
training_args = TrainingArguments(
    output_dir='./results_before',
    per_device_eval_batch_size=16,
    logging_dir='./logs',
    fp16=True,
    gradient_accumulation_steps=2,
)

trainer_before = Trainer(
    model=pretrained_model,
    args=training_args,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=metrics_calculator.compute_metrics
)

eval_results_before = trainer_before.evaluate()
```
### Instantiate your MetricsCalculator
Creates an instance to compute evaluation metrics.
```python
# Instantiate your MetricsCalculator
metrics_calculator = MetricsCalculator()
```
metrics_calculator: This variable is an instance of the MetricsCalculator class, which is likely responsible for calculating various evaluation metrics. This calculator will be utilized later to assess the model's performance.
### Set Training Arguments
Defines the configurations for evaluating the pre-trained model, including directories, batch size, and precision settings.
```python
# Evaluate the pre-trained model before fine-tuning
training_args = TrainingArguments(
    output_dir='./results_before',
    per_device_eval_batch_size=16,
    logging_dir='./logs',
    fp16=True,
    gradient_accumulation_steps=2,
)
```
  - training_args: This is an instance of TrainingArguments, which sets up configurations for training and evaluation. The specific settings are:
    - output_dir: Specifies the directory where the evaluation results will be saved.
    - per_device_eval_batch_size: Defines the batch size for evaluation for each device (e.g., per GPU/CPU). Here, it is set to process 16 samples per device.
    - logging_dir: Specifies the directory where log files will be saved.
    - fp16: Enables mixed precision (16-bit floating-point) evaluation, which can speed up the evaluation process.
    - gradient_accumulation_steps: Specifies the number of steps to accumulate gradients before performing an update. However, this parameter is more relevant during training rather than evaluation.
### Create a Trainer Instance
Sets up the Trainer with the model, evaluation dataset, tokenizer, and metric computation function.
```python
trainer_before = Trainer(
    model=pretrained_model,
    args=training_args,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=metrics_calculator.compute_metrics
)
```
  - trainer_before: This variable is an instance of the Trainer class, which is used to handle the evaluation process. It is initialized with several components:
    - model: Represents the pre-trained model that is to be evaluated.
    - args: The training arguments defined in the previous step.
    - eval_dataset: The dataset used for evaluating the model’s performance.
    - tokenizer: The tokenizer responsible for processing the input data before feeding it into the model.
    - compute_metrics: A method from the metrics_calculator instance that will be used to compute various evaluation metrics.
### Evaluate the Pre-trained Model
Executes the evaluation and stores the results in eval_results_before.
```python
eval_results_before = trainer_before.evaluate()
```
eval_results_before: This variable stores the results of the evaluation. The evaluate method is called on the trainer_before object, triggering the evaluation process which uses the predefined training arguments and computes metrics via the metrics_calculator. 
### Output print evaluation results before fine-tuning
![Evaluation result before fine-tuning](https://github.com/Sayomphon/Stock-Analysis-using-FinBERTandBART/blob/45c8c189f3e60afa41326c1419455262c6af2b52/Evaluation%20result%20before%20fine-tuning.PNG)
