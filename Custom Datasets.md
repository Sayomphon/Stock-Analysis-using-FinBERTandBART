## The code creates custom datasets for training and validation by utilizing a custom dataset class
```python
# Custom datasets
train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length=128)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length=128)
```
This section of the code creates custom datasets for training and validation by wrapping text and label data along with tokenization logic.
  - Variables:
    - train_dataset: This variable stores the training dataset created from the training texts and labels using the custom dataset class.
    - val_dataset: This variable stores the validation dataset created from the validation texts and labels using the custom dataset class.
  - Function:
    - CustomDataset(...):
      - This is presumably a user-defined class that has been designed to handle the tokenization and formatting of the text and label data.
      - train_texts: The text data for the training set.
      - train_labels: The labels corresponding to the text data in the training set.
      - val_texts: The text data for the validation set.
      - val_labels: The labels corresponding to the text data in the validation set.
      - tokenizer: The tokenizer (in this case, the pre-trained BERT tokenizer) used to convert the text data into token IDs.
      - max_length=128: The maximum length of the tokenized sequences. Text that is longer than this length will be truncated, and shorter text will be padded to this length.
