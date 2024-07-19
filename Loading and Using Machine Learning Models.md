## Load several pretrained models and tokenizers from various sources
```python
# Load models
finbert_tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
finbert_model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
sentence_encoder = SentenceTransformer('all-mpnet-base-v2')
```
### Loading the FinBERT Tokenizer and Model
```python
finbert_tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
finbert_model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
```
  - finbert_tokenizer: This line loads a pretrained tokenizer specifically designed for financial sentiment analysis using the FinBERT model. The tokenizer translates text input into a format suitable for the FinBERT model.
  - finbert_model: This line loads the actual FinBERT model, which is designed for sequence classification tasks, such as determining the sentiment of financial text (positive, negative, or neutral).
### Loading the BART Tokenizer and Model for Summarization:
```python
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
```
  - bart_tokenizer: This line loads a pretrained tokenizer based on the BART model, which is used for tasks like text summarization. The tokenizer processes text into token IDs that the BART model can use.
  - bart_model: This line loads the BART model itself, which is designed for conditional text generation, such as summarizing long texts into shorter versions.
### Loading a Sentence Transformer Model
```python
sentence_encoder = SentenceTransformer('all-mpnet-base-v2')
```
This line loads a SentenceTransformer model, specifically the 'all-mpnet-base-v2' variant. This model is used for generating dense vector representations (embeddings) of sentences. These embeddings can be used for tasks such as semantic search, clustering, and classification.
