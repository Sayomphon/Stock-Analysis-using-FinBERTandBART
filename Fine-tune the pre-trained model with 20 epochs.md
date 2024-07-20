## Fine-tuning a pre-trained model using a custom function
```python
# Fine-tune the pre-trained model with 20 epoch
eval_results_after = custom_fine_tune(pretrained_model, tokenizer, train_dataset, val_dataset, output_dir='./results_after_finetune', epochs=20)
```
This line calls the custom_fine_tune function to fine-tune the pre-trained model. The results of the fine-tuning process are stored in the eval_results_after variable.
### Function Call
```python
# Fine-tune the pre-trained model with 20 epoch
eval_results_after = custom_fine_tune(pretrained_model, tokenizer, train_dataset, val_dataset, output_dir='./results_after_finetune', epochs=20)
```
  - pretrained_model: The pre-trained model that will be fine-tuned using the provided datasets.
  - tokenizer: The tokenizer associated with the pre-trained model, used to process the input text data.
  - train_dataset: The dataset used for training the model during the fine-tuning process.
  - val_dataset: The dataset used for validating the model's performance during the fine-tuning process.
  - output_dir: Specifies './results_after_finetune' as the directory to save the results of the fine-tuning process.
  - epochs: Indicates that the model should be fine-tuned for 20 epochs, which means the training process will iterate over the entire training dataset 20 times.
### Output while tuning
![While tuning](https://github.com/Sayomphon/Stock-Analysis-using-FinBERTandBART/blob/c0185d1e6fcd6f8f2d423f0fe7c343593ef488eb/While%20tuning.PNG)
