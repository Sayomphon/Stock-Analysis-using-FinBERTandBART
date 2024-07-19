## Define a class MetricsCalculator that is responsible for computing various evaluation metrics for a machine learning model
```python
class MetricsCalculator:
    def __init__(self):
        self.metric = load_metric("accuracy")
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def compute_metrics(self, eval_pred):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids

        # Calculate accuracy
        accuracy = self.metric.compute(predictions=predictions.argmax(axis=1), references=labels)

        # Calculate loss
        loss = self.loss_fn(torch.tensor(predictions), torch.tensor(labels)).item()

        # Calculate precision, recall, f1-score
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions.argmax(axis=1), average='weighted')

        return {
            'eval_accuracy': accuracy['accuracy'],
            'eval_loss': loss,
            'eval_precision': precision,
            'eval_recall': recall,
            'eval_f1': f1,
        }
```
### Class Initialization (__init__method)
```python
class MetricsCalculator:
    def __init__(self):
        self.metric = load_metric("accuracy")
        self.loss_fn = torch.nn.CrossEntropyLoss()
```
  - The constructor of the MetricsCalculator class initializes the class by loading a metric named "accuracy" using self.metric = load_metric("accuracy").
  - self.loss_fn is set to Cross-Entropy Loss, which is commonly used for classification tasks.
### Compute_metrics Method
 ```python   
    def compute_metrics(self, eval_pred):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
```
  - This method takes an argument eval_pred, which contains the model’s predictions and the true labels.
  - The predictions variable gets its value from eval_pred.predictions, and labels gets its value from eval_pred.label_ids.
### Calculating Accuracy
```python
        # Calculate accuracy
        accuracy = self.metric.compute(predictions=predictions.argmax(axis=1), references=labels)
```
  -Using self.metric, the method calculates accuracy by comparing the predicted classes (predictions.argmax(axis=1)) with the true labels (labels).
### Calculating Loss
```python
        # Calculate loss
        loss = self.loss_fn(torch.tensor(predictions), torch.tensor(labels)).item()
```
  - The method calculates the loss using the Cross-Entropy Loss function (self.loss_fn). It converts predictions and labels into tensors and computes the loss. The result is converted to a scalar value using .item().
### Calculating Precision, Recall, and F1-Score
```python
        # Calculate precision, recall, f1-score
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions.argmax(axis=1), average='weighted')
```
  - It uses the precision_recall_fscore_support function (assumed to be from sklearn) to compute precision, recall, and F1-score. It compares labels with the predicted classes (predictions.argmax(axis=1)) using 'weighted' averaging.
### Returning the Results
 ```python
        return {
            'eval_accuracy': accuracy['accuracy'],
            'eval_loss': loss,
            'eval_precision': precision,
            'eval_recall': recall,
            'eval_f1': f1,
        }
```
  -- The method returns a dictionary containing the calculated evaluation metrics: eval_accuracy, eval_loss, eval_precision, eval_recall, and eval_f1.
