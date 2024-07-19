## Defines a custom dataset class CustomDataset that is intended to be used with the PyTorch library
```python
# Create Custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
```
### Defining the Custom Dataset Class
```python
class CustomDataset(torch.utils.data.Dataset):
```
This line defines a class named CustomDataset that inherits from torch.utils.data.Dataset, which is a standard interface for datasets in PyTorch.
### Initialization Method (__init__)
```python
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
```
  - __init__: The constructor method is used to initialize the dataset object.
  - texts: A list of text samples.
  - labels: A list of corresponding labels for the text samples.
  - tokenizer: The tokenizer object (such as a BERT tokenizer) that will be used to tokenize the text samples.
  - max_length: The maximum length for each tokenized sequence.
### Length Method (__len__)
```python
    def __len__(self):
        return len(self.texts)
```
This method returns the total number of text samples in the dataset.
### Get Item Method (__getitem__)
```python
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
```
  - __getitem__: This method retrieves a single sample from the dataset at the specified index idx.
  - text: Retrieves the text sample at index idx.
  - label: Retrieves the label for the text sample at index idx.
  - encoding: Uses the tokenizer’s encode_plus method to tokenize the text sample. This method parameters are:
    - text: The text to tokenize.
    - add_special_tokens: Adds special tokens (like [CLS] and [SEP] for BERT).
    - max_length: Specify the maximum length of the tokenized sequence.
    - return_token_type_ids: Does not return token type IDs (set to False).
    - padding: Pads the sequence to the maximum length ('max_length').
    - truncation: Truncates the sequence if it exceeds the maximum length.
    - return_attention_mask: Returns the attention mask.
    - return_tensors: Returns the tensors in PyTorch format ('pt').
  - The method returns a dictionary with three key-value pairs:
    - 'input_ids': The token IDs of the text, flattened to a one-dimensional tensor.
    - 'attention_mask': The attention mask, flattened to a one-dimensional tensor.
    - 'labels': The label for the text sample, converted to a PyTorch tensor of type long.
