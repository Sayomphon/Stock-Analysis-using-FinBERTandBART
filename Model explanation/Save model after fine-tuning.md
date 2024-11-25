## This code snippet saves both the fine-tuned model and its associated tokenizer to the directory./saved_model.
```python
# Save model after fine-tuning
fine_tuned_model.save_pretrained('./saved_model')
fine_tuned_tokenizer.save_pretrained('./saved_model')
```
### Save Fine-Tuned Model
Saves the fine-tuned model to a specified directory.
```python
fine_tuned_model.save_pretrained('./saved_model')
```
  - fine_tuned_model: The variable that stores the fine-tuned model.
  - save_pretrained('./saved_model'): A method used to save the model's weights and configuration files to the directory ./saved_model. This allows the model to be reloaded and used later.
### Save Fine-Tuned Tokenizer
Saves the tokenizer associated with the fine-tuned model to the same specified directory.
```python
fine_tuned_tokenizer.save_pretrained('./saved_model')
```
  - fine_tuned_tokenizer: The variable that stores the fine-tuned tokenizer.
  - save_pretrained('./saved_model'): A method used to save the tokenizer's configuration and vocabulary files to the directory ./saved_model. This ensures that the tokenizer can be reloaded and used later in conjunction with the saved model.
#### Result
![Save model after fine-tuning](https://github.com/Sayomphon/Stock-Analysis-using-FinBERTandBART/blob/a43a9802a68697464485fb9ab555d027e9d142e4/Save%20model%20after%20fine-tuning.PNG)
