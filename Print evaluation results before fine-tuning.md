## Print several key evaluation metrics for a model before the fine-tuning process.
```python
# Print evaluation results before fine-tuning
print("Evaluation results before fine-tuning:")
print(f"Loss: {eval_results_before['eval_loss']}")
print(f"Accuracy: {eval_results_before['eval_accuracy']}")
print(f"Precision: {eval_results_before['eval_precision']}")
print(f"Recall: {eval_results_before['eval_recall']}")
print(f"F1-score: {eval_results_before['eval_f1']}")
```
### Print Header
Prints a header to indicate that the following lines will display the evaluation results of the model before fine-tuning.
```python
print("Evaluation results before fine-tuning:")
```
### Print Loss
Outputs the loss metric of the model's evaluation before fine-tuning.
```python
print(f"Loss: {eval_results_before['eval_loss']}")
```
  - eval_results_before: Contains the evaluation metrics gathered before the fine-tuning procedure.
  - eval_loss: Represents the evaluation loss. It measures the error made by the model's predictions compared to the actual values.
### Print Accuracy
Displays the accuracy metric of the model's evaluation before fine-tuning.
```python
print(f"Accuracy: {eval_results_before['eval_accuracy']}")
```
  - eval_results_before: Stores various evaluation metrics before fine-tuning.
  - eval_accuracy: Represents the proportion of correct predictions made by the model out of all predictions.
### Print Precision
Outputs the precision metric of the model's evaluation before fine-tuning.
```python
print(f"Precision: {eval_results_before['eval_precision']}")
```
  - eval_results_before: Hosts the evaluation metrics that were calculated before fine-tuning the model.
  - eval_precision: Indicates the ratio of true positive predictions to the total number of positive predictions made.
### Print Recall
Displays the recall metric of the model's evaluation before fine-tuning.
```python
print(f"Recall: {eval_results_before['eval_recall']}")
```
  - eval_results_before: Contains the evaluation metrics obtained before fine-tuning the model.
  - eval_recall: Measures the ratio of true positive predictions to the total number of actual positive instances in the data.
### Print F1-score
Outputs the F1-score of the model's evaluation before fine-tuning.
```python
print(f"F1-score: {eval_results_before['eval_f1']}")
```
  - eval_results_before: Stores the set of evaluation metrics generated before fine-tuning.
  - eval_f1: Represents the harmonic mean of precision and recall, providing a balanced measure of the model's performance.
