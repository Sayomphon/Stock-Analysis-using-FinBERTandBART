## Print several key evaluation metrics for a model after the fine-tuning process.
```python
# Print evaluation results after fine-tuning
print("Evaluation results after fine-tuning:")
print(f"Loss: {eval_results_after['eval_loss']}")
print(f"Accuracy: {eval_results_after['eval_accuracy']}")
print(f"Precision: {eval_results_after['eval_precision']}")
print(f"Recall: {eval_results_after['eval_recall']}")
print(f"F1-score: {eval_results_after['eval_f1']}")
```
### Print Header
Prints a header to indicate that the following lines will display the evaluation results of the model after the fine-tuning process.
```python
print("Evaluation results after fine-tuning:")
```
### Print Loss
Outputs the loss metric of the model's evaluation after fine-tuning.
```python
print(f"Loss: {eval_results_after['eval_loss']}")
```
  - eval_results_after: Contains the evaluation metrics gathered after the fine-tuning process.
  - eval_loss: Represents the evaluation loss. It measures the error made by the model's predictions compared to the actual values.
### Print Accuracy
Displays the accuracy metric of the model's evaluation after fine-tuning.
```python
print(f"Accuracy: {eval_results_after['eval_accuracy']}")
```
  - eval_results_after: Stores various evaluation metrics after fine-tuning.
  - eval_accuracy: Represents the proportion of correct predictions made by the model out of all predictions.
### Print Precision
Outputs the precision metric of the model's evaluation after fine-tuning.
```python
print(f"Precision: {eval_results_after['eval_precision']}")
```
  - eval_results_after: Hosts the evaluation metrics that were calculated after fine-tuning the model.
  - eval_precision: Indicates the ratio of true positive predictions to the total number of positive predictions made.
### Print Recall
Displays the recall metric of the model's evaluation after fine-tuning.
```python
print(f"Recall: {eval_results_after['eval_recall']}")
```
  - eval_results_after: Contains the evaluation metrics obtained after fine-tuning the model.
  - eval_recall: Measures the ratio of true positive predictions to the total number of actual positive instances in the data.
### Print F1-score
Outputs the F1-score of the model's evaluation after fine-tuning.
```python
print(f"F1-score: {eval_results_after['eval_f1']}")
```
  - eval_results_after: Stores the set of evaluation metrics generated after fine-tuning.
  - eval_f1: Represents the harmonic mean of precision and recall, providing a balanced measure of the model's performance.
### Output print evaluation results after fine-tuning
![Evaluation result after fine-tuning](https://github.com/Sayomphon/Stock-Analysis-using-FinBERTandBART/blob/ec4b6f815967e6421a9195f6e4d7ad1a7e6ef9d8/Evaluation%20result%20after%20fine-tuning.PNG)
