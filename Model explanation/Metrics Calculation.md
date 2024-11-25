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
  - class MetricsCalculator: Defines a class named MetricsCalculator that is responsible for computing various evaluation metrics for a model.
  - def __init__(self): This is the constructor method that initializes the class.
  - self.metric = load_metric("accuracy"): Loads the accuracy metric using the load_metric function from the Hugging Face library.
  - self.loss_fn = torch.nn.CrossEntropyLoss(): Initializes the loss function as cross-entropy loss, which is commonly used for classification problems.
### Compute_metrics Method
 ```python   
    def compute_metrics(self, eval_pred):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
```
  - def compute_metrics(self, eval_pred): Defines a method named compute_metrics that will compute various evaluation metrics.
  - predictions = eval_pred.predictions: Extracts the model's predictions from the evaluation output.
  - labels = eval_pred.label_ids: Extracts the true labels from the evaluation output.
### Calculating Accuracy
```python
        # Calculate accuracy
        accuracy = self.metric.compute(predictions=predictions.argmax(axis=1), references=labels)
```
  - accuracy = self.metric.compute(predictions=predictions.argmax(axis=1), references=labels): Computes the accuracy by comparing the predicted labels (obtained using argmax on predictions) with the true labels.
### Calculating Loss
```python
        # Calculate loss
        loss = self.loss_fn(torch.tensor(predictions), torch.tensor(labels)).item()
```
  - loss = self.loss_fn(torch.tensor(predictions), torch.tensor(labels)).item() : Computes the cross-entropy loss between the predictions and the true labels. The item() method extracts the scalar value from the loss tensor.
### Calculating Precision, Recall, and F1-Score
```python
        # Calculate precision, recall, f1-score
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions.argmax(axis=1), average='weighted')
```
  - precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions.argmax(axis=1), average='weighted'): Computes the precision, recall, and F1-score using the precision_recall_fscore_support
function. The average='weighted' parameter weights the metrics by the number of true instances for each label.'weighted' averaging.
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
  - Returns a dictionary containing the calculated metrics:
    - eval_accuracy: The computed accuracy.
    - eval_loss: The computed loss.
    - eval_precision: The computed precision.
    - eval_recall: The computed recall.
    - eval_f1: The computed F1-score.
