## This code snippet loads a fine-tuned model and its corresponding tokenizer from the directory ./results_after_finetune
```python
# Load the fine-tuned model
fine_tuned_model = AutoModelForSequenceClassification.from_pretrained('./results_after_finetune')
fine_tuned_tokenizer = AutoTokenizer.from_pretrained('./results_after_finetune')
```
### Load Fine-Tuned Model
Loads the fine-tuned model from the specified directory.
```python
fine_tuned_model = AutoModelForSequenceClassification.from_pretrained('./results_after_finetune')
```
  - fine_tuned_model: This variable stores the model that has been fine-tuned.
  - AutoModelForSequenceClassification: A class from the transformers library (by Hugging Face) used to load models that are pre-trained or fine-tuned for sequence classification tasks.
  - from_pretrained('./results_after_finetune'): A method used to load the model weights and configuration from the specified directory ./results_after_finetune, where the fine-tuned model is saved.
### Load Fine-Tuned Tokenizer
Loads the tokenizer associated with the fine-tuned model from the specified directory.
```python
fine_tuned_tokenizer = AutoTokenizer.from_pretrained('./results_after_finetune')
```
  - fine_tuned_tokenizer: This variable stores the tokenizer that has been fine-tuned alongside the model.
  - AutoTokenizer: A class from the transformers library used to load tokenizers compatible with the pre-trained or fine-tuned models.
  - from_pretrained('./results_after_finetune'): A method used to load the tokenizer configuration and vocabulary from the specified directory ./results_after_finetune, where the fine-tuned tokenizer is saved.
