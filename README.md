# Stock-Analysis-using-FinBERTandBART

## Code Summary
### This project is designed to fetch stock data and financial news, predict trends, and provide investment advice using Machine Learning models through Transformers. It uses various libraries to process the data, fine-tune models, and create an interactive UI for users to input their API keys and stock symbols.

## Installation and setup
#### 1.) Install necessary libraries
  !pip install requests pandas transformers ipywidgets torch plotly sentence-transformers datasets accelerate bitsandbytes
#### 2.) Enable IPyWidgets for Google Colab
  from google.colab import output
  output.enable_custom_widget_manager() 

## Main Functionality of the Code
#### 1.) Installing and Enabling Libraries
  The code installs and enables essential libraries such as requests, pandas, transformers, torch, plotly, ipywidgets, and others for data fetching, processing, and UI rendering.
#### 2.) Fetching Stock Data:
  The get_stock_data function fetches minute-level stock data from the Alpha Vantage API and converts the data into a manageable DataFrame format.
#### 3.) Fetching Financial News
  The get_latest_news function leverages the News API to fetch the latest financial news linked to the given stock symbol provided by the user.
#### 4.) Text Processing and Cleaning:
  Clean_text function cleans the text by removing URLs, emails, and special characters.
#### 5.) Loading and Using Machine Learning Models
  The code loads various Transformer models, such as finbert for financial sentiment analysis, bart for summarizing news, and sentence_transformer for generating text embeddings.
#### 6.) Generate dense vector representations
  Function takes a text input, uses the SentenceTransformer model to encode this text into dense embeddings, and then returns these embeddings.
#### 7.) Creating Custom Datasets
  CustomDataset is a class used for creating and managing the dataset that will be used to fine-tune the BERT model by tokenizing text and preparing it for model processing.
#### 8.) Metrics Calculation
  The MetricsCalculator class is designed to compute various evaluation metrics for a model, including accuracy, loss, precision, recall, and F1-score.
#### 9.) Model Fine-tuning:
  The custom_fine_tune function fine-tunes the pre-trained BERT model on custom datasets, using predefined training arguments.
#### 10.) Generate advice
  The function generate_advice is designed to generate a summarized version of a given text prompt using a pre-trained BART model. This function tokenizes the input prompt, generates a summary, and then decodes the summary back into text.
#### 11.) Summarize Text
  The function is designed to generate a summarized version of a given input text using a pre-trained BART model. This function takes a text input, tokenizes it, generates a summary, and then decodes the summary back into human-readable text.
#### 12.) Stock Analysis Functions before Fine-tuning
  Function aims to analyze stock information and provide investment advice based on stock performance, news sentiment, and recent market trends. It utilizes APIs for stock data and news fetching,
#### 13.) Create UI before fine-tuning
  This code snippet creates a simple user interface (UI) using the ipywidgets library in Python. The UI allows the user to input API keys for Alpha Vantage and a news service, along with a stock symbol. Upon clicking a button.
#### 14.) Display UI before fine-tuning
  The function with these widgets as arguments, it renders these UI elements on the screen in the order they are listed. Users can interact with these input fields and the button, and the output area will be used to show the results after the user triggers the analysis by clicking the button.
#### 15.) Load Datasets
  This function call loads the IMDB dataset, which is a popular dataset for sentiment analysis. The IMDB dataset contains movie reviews labeled as positive or negative.
#### 16.) Split into training and validation sets
  This section of the code is used to split the loaded dataset into training and validation sets, specifically extracting the texts and their corresponding labels for each set.
#### 17.) Tokenizer
  The code initializes a tokenizer, which is responsible for converting text into a format that can be fed into a machine learning model.
#### 18.) Custom datasets
  This section of the code creates custom datasets for training and validation by wrapping text and label data along with tokenization logic.
#### 19.) Load the Pre-Trained Model
  This line of code loads a pre-trained model for sequence classification tasks, specifically the BERT model.
#### 20.) Evaluate the pre-trained model before fine-tuning
  This code sets up an evaluation framework for assessing the performance of a pre-trained model before fine-tuning
#### 21.) Print evaluation results before fine-tuning
  The code snippet prints several key evaluation metrics for a model before the fine-tuning process.
#### 22.) Fine-tune the pre-trained model with 20 epochs
  This code snippet aims to fine-tune a pre-trained model using a custom function named custom_fine_tune. The function is called with several arguments: pretrained_model, tokenizer, train_dataset, val_dataset, output_dir, and epochs.
#### 23.) Print evaluation results after fine-tuning
  The code snippet prints several key evaluation metrics for a model after the fine-tuning process
#### 24.) Load the fine-tuned model
 This code snippet loads a fine-tuned model and its corresponding tokenizer from the directory ./results_after_finetune. The fine_tuned_model variable uses the AutoModelForSequenceClassification class from the transformers library to load the pre-trained model weights and configuration
#### 25.) Save model after fine-tuning
  This code snippet saves both the fine-tuned model and its associated tokenizer to the directory ./saved_model. The fine_tuned_model.save_pretrained('./saved_model') method saves the model weights and configuration to the specified directory, making it possible to reload and use the model at a later time.
#### 26.) Function for stock analysis after fine-tuning
  This code defines a function analyze_stock_after_finetune that performs stock analysis using a fine-tuned model for sentiment analysis of news articles. The function takes three parameters: alpha_vantage_api_key, news_api_key, and symbol.
#### 27.) Create UI after fine-tuning
  This code sets up a user interface to perform stock analysis using input widgets for API keys and stock symbols. It initializes text input fields for the Alpha Vantage API key, the News API key, and the stock symbol, along with an output area to display results.
#### 28.) Display UI after fine-tuning
  This code snippet displays the user interface components for stock analysis in a Jupyter notebook or IPython environment. The display function is used to render the text input widgets for entering the Alpha Vantage API key, News API key, and stock symbol.
#### 29.) Load saved model
  This code snippet loads a fine-tuned model and its associated tokenizer from the directory specified by the saved_model_path variable ('./saved_model').
## Use Case
#### 1.) Investors:
  - Usage: Investors can use this code to fetch stock data and news related to a specific stock, such as AAPL (Apple Inc.), and receive a market trend prediction and investment advice.
  - Outcome: The investor will receive information about the latest stock price, summarized key news articles, and the sentiment analysis of those news articles (positive, negative, neutral), along with investment advice.
#### 2.) Financial Institutions:
  - Usage: Financial institutions can use this code to fetch data and perform investment analysis for their clients, to provide accurate and trend-based investment advice.
  - Outcome: Institutions can provide daily or weekly investment recommendations to their clients and adjust investment strategies based on the real-time market situation to ensure optimal returns for their clients.
#### 3.) Software Developers:
  - Usage: Developers can use this code as a foundation to build more complex and comprehensive systems for stock analysis and investment advice.
  - Customization: The code can be customized or extended to include additional functionalities, such as aggregating data from multiple sources or using custom Machine Learning models.
##### This code serves as a fairly comprehensive system for fetching and analyzing financial data, which can be easily extended for use in larger applications.

## Additional
#### License
  This project is licensed under the MIT License - see the LICENSE.md file for details.
#### Acknowledgments
  - Special thanks to Hugging Face for providing the Transformers library.
  - Alpha Vantage and News API for offering financial data and news services.
#### Contact
  For any questions or feedback, please open an issue in this repository.
  You can copy and paste the above content into a `README.md` file for your GitHub repository. Customize it further as needed to fit your specific needs and details about your project.

## 1. Installation and Setup
Before running the project, you need to install the necessary libraries and enable ipywidgets if you're using Google Colab.
#### 1.) Installing Necessary Libraries
To install all the required libraries, run the following command
```python
# This command installs all the essential Python libraries needed for the project.
!pip install requests pandas transformers ipywidgets torch plotly sentence-transformers datasets accelerate bitsandbytes
```
#### 2.) Enabling ipywidgets for Google Colab
If you're using Google Colab, enable ipywidgets by running the following code
```python
# This code snippet enables the custom widget manager for ipywidgets within Google Colab, allowing the creation and display of interactive widgets in the notebook.
from google.colab import output
output.enable_custom_widget_manager()
```
## 2. Main Functionality of the Code
#### 1.) Installing and Enabling Libraries
This script imports a comprehensive set of libraries crucial for setting up a machine learning environment, particularly focusing on NLP tasks. It brings together tools for data handling (pandas, requests), model building and training (transformers, torch), interactive visualization (ipywidgets, plotly), and performance evaluation (datasets, sklearn.metrics). The combination of these imports indicates the script is geared towards creating NLP models, preprocessing data, visualizing results, and interacting with data in a Jupyter notebook setup.
```python
import requests
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, TrainingArguments, Trainer, default_data_collator, EarlyStoppingCallback, BartTokenizer, BartForConditionalGeneration
import torch
import ipywidgets as widgets
from IPython.display import display, HTML
import re
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
```
#### 2.) Fetching Stock Data
The get_stock_data function fetches intraday stock data for a given stock symbol using the Alpha Vantage API. It constructs the API request URL, sends an HTTP GET request, and processes the JSON response into a pandas DataFrame. The function ensures the DataFrame has readable column names and datetime index for ease of use. If the data is not available or the API request fails, the function raises an error, providing the response for debugging.
```python
# Function to fetch intraday stock data
def get_stock_data(api_key, symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
 
    if 'Time Series (1min)' in data:
        df = pd.DataFrame(data['Time Series (1min)']).T
        df = df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        })
        df.index = pd.to_datetime(df.index)
        return df
    else:
        raise ValueError(f"Data not available or API request failed. Response: {data}")
```
#### 3.) Fetching Financial News
The get_latest_news function fetches the latest news articles related to a specified stock symbol using the News API. It constructs the API URL by combining the base URL for the "everything" endpoint with the provided stock symbol and API key as query parameters. It then uses the requests library to send a GET request to the constructed URL. Finally, it converts the response from the News API into JSON format and returns this JSON object, which contains the news articles and other related data. This function is useful for retrieving up-to-date financial news articles related to a specific stock symbol, enabling further analysis or display in a user interface.
```python
# Function to fetch financial news
def get_latest_news(symbol, api_key):
    news_url = f'https://newsapi.org/v2/everything?q={symbol}&apiKey={api_key}'
    response = requests.get(news_url)
    return response.json()
```
#### 4.) Text Processing and Cleaning
The clean_text function is designed to preprocess and clean input text by removing unnecessary or unwanted elements such as URLs, email addresses, and special characters. It removes any URLs present in the text using a regular expression pattern that matches and replaces URLs with an empty string. It removes any email addresses by using a regular expression pattern that matches and replaces email-like substrings with an empty string. Special characters are removed by using a regular expression pattern that matches characters other than letters, numbers, or whitespace, thereby keeping only letters, digits, and whitespace. Finally, it trims leading and trailing whitespace from the text using the strip method. The function returns the cleaned text, making it ready for further processing or analysis.
```python
# Clean text function
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)  # Remove special characters
    text = text.strip()
    return text
```
#### 5.) Loading and Using Machine Learning Models
This section of code is intended to load several pretrained models and tokenizers from various sources, which could be used for natural language processing tasks such as sentiment analysis, summarization, and sentence embeddings.
The code initializes various pre-trained models and their tokenizers for different natural language processing tasks:
  - FinBERT for Sentiment Analysis:
    - finbert_tokenizer and finbert_model are loaded from the 'yiyanghkust/finbert-tone' repository. FinBERT specializes in financial sentiment analysis.
  - BART for Text Summarization:
    - bart_tokenizer and bart_model are loaded from the 'facebook/bart-large-cnn' repository. BART is utilized for tasks such as text summarization and conditional text generation.
  - Sentence-BERT for Sentence Embeddings:
    - sentence_encoder is loaded using the 'all-mpnet-base-v2' variant. Sentence-BERT is used to produce sentence embeddings, which are useful for tasks that involve semantic understanding of sentences.
```python
# Load models
finbert_tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
finbert_model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
sentence_encoder = SentenceTransformer('all-mpnet-base-v2')
```
#### 6.) Generate dense vector representations
The get_embeddings function takes a piece of text as input and uses a pre-initialized sentence encoder to generate its embeddings. The function calls the encode method of the sentence_encoder to obtain a high-dimensional numerical vector that captures the semantic meaning of the input text. It then returns these embeddings, which can be used for various tasks such as similarity comparison, clustering, or input to other machine learning models. This function abstracts the process of embedding generation, making it easy to obtain consistent representations of text.
```python
# Function embedding
def get_embeddings(text):
    embeddings = sentence_encoder.encode(text)
    return embeddings
```
#### 7.) Creating Custom Datasets
The CustomDataset class creates a custom dataset compatible with PyTorch's DataLoader. It takes in text data and labels, tokenizes the text using the provided tokenizer, and ensures each tokenized sequence is of the specified maximum length. It also handles padding and truncation appropriately. The class provides methods to get the number of samples and to retrieve a sample by index. Each retrieved sample includes token IDs, attention masks, and labels in a dictionary format, ready for use in training or evaluation with a PyTorch model.
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
#### 8.) Metrics Calculation
The MetricsCalculator class is designed to compute various evaluation metrics for a model, including accuracy, loss, precision, recall, and F1-score. The compute_metrics function takes model predictions and true labels, calculates the specified metrics, and returns them in a structured dictionary format, making it easier to evaluate the performance of the model in various tasks.
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
#### 9.) Model Fine-tuning
This function, custom_fine_tune , fine-tunes a pre-trained transformer model using a training dataset, a validation dataset, and a specified set of training arguments. It then evaluates the trained model and saves both the model and tokenizer to an output directory.
```python
# Function to fine-tune a pre-trained model
def custom_fine_tune(transformer_model, tokenizer, train_dataset, val_dataset, output_dir, epochs=3, batch_size=16, learning_rate=2e-5):
    training_args = TrainingArguments(
        output_dir=output_dir,  # Specify the output directory for saving results
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        per_device_eval_batch_size=batch_size,
        eval_strategy='epoch',  # Updated according to warning
        save_strategy='epoch',
        logging_dir='./logs',
        fp16=True,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        logging_steps=100,
        weight_decay=0.01,
    )

    num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels,
        hidden_dropout_prob=0.4,
        attention_probs_dropout_prob=0.4
    )

    # Instantiate your MetricsCalculator
    metrics_calculator = MetricsCalculator()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=metrics_calculator.compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Start training
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()

    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return eval_results
```
#### 10.) Generate advice
The generate_advice function uses a pre-trained BART model to generate advice based on a given prompt. The function first encodes the input prompt with a prefix "summarize: " using the BART tokenizer, converting it into PyTorch tensors. The encoded input is then fed into the BART model to generate a summary, with constraints on the maximum and minimum length, as well as other generation parameters to ensure high-quality output. Finally, the function decodes the generated token IDs back into human-readable text and returns this summary as the resulting advice. This process leverages the BART model's strong text summarization capabilities to produce coherent and concise advice from longer prompts.
```python
# Function to generate advice using BART
def generate_advice(prompt):
    inputs = bart_tokenizer.encode("summarize: " + prompt, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = bart_model.generate(inputs, max_length=512, min_length=200, length_penalty=2.0, num_beams=4, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```
#### 11.) Summarize Text
The summarize_text function uses a pre-trained BART model to generate a summary for a given input text. It first encodes the input text using the BART tokenizer with a prefix indicating that summarization is required. The encoded input is then passed through the BART model to generate the summary, using beam search and specific constraints on the length and penalties to ensure high-quality results. The generated token IDs are then decoded back into human-readable text, and the function returns the summarized text. This process utilizes BART’s strong text summarization capabilities to create coherent and concise summaries from longer texts.
```python
# Function to summarize text using BART
def summarize_text(text):
    inputs = bart_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = bart_model.generate(inputs, max_length=300, min_length=100, length_penalty=2.0, num_beams=4, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```
#### 12.) Stock Analysis Functions before Fine-tuning
The analyze_stock_before_finetune function is designed to analyze a given stock symbol by fetching its latest price, gathering and processing recent news articles, and generating detailed investment advice. The function utilizes multiple components including sentiment analysis and text summarization, and returns a set of comprehensive results that include the latest price, generated advice, processed news summaries, and the complete stock data.
```python
# Function for stock analysis before fine-tuning
def analyze_stock_before_finetune(alpha_vantage_api_key, news_api_key, symbol):
    stock_data = get_stock_data(alpha_vantage_api_key, symbol)
    latest_price = stock_data['close'][0]

    # Fetch latest news
    news_data = get_latest_news(symbol, news_api_key)
    news_summary_cleaned = []

    if news_data and 'articles' in news_data:
        for article in news_data['articles'][:5]:
            description = clean_text(article['description'])
            inputs = finbert_tokenizer(description, return_tensors='pt', padding=True, truncation=True, max_length=512)
            outputs = finbert_model(**inputs)
            sentiment = torch.argmax(outputs.logits, dim=1).item()
            sentiment_label = "Positive" if sentiment == 1 else "Negative" if sentiment == 0 else "Neutral"

            # Summarize news using finbert model
            summary = summarize_text(description)

            # Get embeddings of the cleaned text
            embeddings = get_embeddings(description)

            news_summary_cleaned.append((article['title'], sentiment_label, summary, embeddings))

    # Investment advice with detailed prompt
    advice_prompt = (
        f"Current market trends show a bullish movement in the tech sector. {symbol} has been gaining momentum. "
        f"News articles suggest significant investor interest. Given the current price of {symbol}, generate detailed investment advice "
        f"considering market trends, stock performance, and financial news. "
        f"Provide a thorough analysis including potential risks, market conditions, and long-term investment potential. "
        f"Discuss the stock's historical performance, recent news impact, and any upcoming events that may influence its price. "
        f"Also, offer a strategic plan for both short-term and long-term investors."
    )

    advice = generate_advice(advice_prompt)

    return latest_price, advice, news_summary_cleaned, stock_data
```
#### 13.) Create UI before fine-tuning
The code creates an interactive UI allowing users to input API keys and a stock symbol. When the "Analyze Stock" button is clicked, the function on_button_click is triggered, which fetches stock data and news, processes this information, and displays the latest stock price, investment advice, and summarized news articles. Additionally, it visualizes the stock data using a candlestick chart, all within the Jupyter Notebook interface using ipywidgets and Plotly for visualization.
```python
# Create UI before fine-tuning
alpha_vantage_api_key_widget = widgets.Text(value='', placeholder='Enter your Alpha Vantage API Key', description='Alpha Vantage API Key:')
news_api_key_widget = widgets.Text(value='', placeholder='Enter your News API Key', description='News API Key:')
symbol_widget = widgets.Text(value='', placeholder='Enter stock symbol (e.g., AAPL)', description='Stock Symbol:')
output = widgets.Output()

def on_button_click(b):
    with output:
        output.clear_output()
        alpha_vantage_api_key = alpha_vantage_api_key_widget.value
        news_api_key = news_api_key_widget.value
        symbol = symbol_widget.value
        if alpha_vantage_api_key and news_api_key and symbol:
            latest_price, advice, news_summary_cleaned, stock_data = analyze_stock_before_finetune(alpha_vantage_api_key, news_api_key, symbol)
            display(HTML(f'<h3>Current Price of {symbol}: {latest_price}</h3>'))
            display(HTML(f'<h4>Investment Advice:</h4><p>{advice}</p>'))
            display(HTML('<h4>Latest Financial News:</h4>'))
            for title, sentiment, summary, embeddings in news_summary_cleaned:
                #display(HTML(f"<b>{title}</b> (Sentiment: {sentiment})<br>{summary}<br>Embeddings: {embeddings}<br><br>"))
                display(HTML(f"<b>{title}</b> (Sentiment: {sentiment})<br>{summary}<br><br>"))

            # Visualize stock data
            fig = go.Figure(data=[go.Candlestick(
                x=stock_data.index,
                open=stock_data['open'],
                high=stock_data['high'],
                low=stock_data['low'],
                close=stock_data['close']
            )])
            fig.update_layout(title=f'Stock Price Data for {symbol}', yaxis_title='Price (USD)', xaxis_title='Time')
            fig.show()
        else:
            print("Please enter the Alpha Vantage API key, News API key and stock symbol.")

button = widgets.Button(description="Analyze Stock")
button.on_click(on_button_click)
```
#### 14.) Display UI
The line of code utilizes the
display()
function to render a user interface where users can enter their Alpha Vantage API key, News API key, and stock symbol. Additionally, it includes a button to initiate the stock analysis and an output space to display the results. This setup facilitates a seamless interaction for users to analyze stock data and view the relevant information in an organized manner.
```python
# Display UI
display(alpha_vantage_api_key_widget, news_api_key_widget, symbol_widget, button, output)
```
#### Example UI use case
![Display UI before fine-tuning](https://github.com/Sayomphon/Stock-Analysis-using-FinBERTandBART/blob/bcc278dce82d0ed339873e211168cb172a4dea31/Display%20UI.PNG)
#### Output after applying Analyze stock

#### Current Price of TSLA: 249.3800
#### Investment Advice:
Current market trends show a bullish movement in the tech sector. News articles suggest significant investor interest. Given the current price of TSLA, generate detailed investment advice considering market trends, stock performance, and financial news. Discuss the stock's historical performance, recent news impact, and any upcoming events that may influence its price. Also, offer a strategic plan for both short-term and long-term investors. For confidential support call the Samaritans on 08457 90 90 90 or visit a local Samaritans branch, see www.samaritans.org for details. In the U.S. call the National Suicide Prevention Lifeline on 1-800-273-8255 or visit http://www.suicidepreventionlifeline.org/. In the UK, call the helpline on 0800-847-9090 or click here for confidential support. For support in the United Kingdom, call 08457 909090 or visit the Samaritans on 08457 93 90 90.
#### Latest Financial News:
##### Tesla robotaxi won't be ready for scale until 2030: Analyst (Sentiment: Positive)
Tesla shares are trading higher on Friday despite a Bloomberg report indicating a twomonth delay in unveiling the company's highly anticipated new product. Tesla TSLA shares are Trading Higher on Friday Despite a Bloomberg Report indicating a Twomonth Delay in unveiling its highly anticipated product. The company is expected to unveil the new product on March 18. The stock is trading up 1.7% to $264.50 per share in Friday's pre-market trading. The market closed down 0.4% on Thursday.
##### Cathie Wood says she wouldn't have sold Nvidia stake 'had we known that the market was going to reward it' (Sentiment: Negative)
Cathie Woods investment fund Ark Invest sold at least 45 million worth of Nvidia stock this year per The Wall Street Journal. Cathie Woods invested in Nvidia through her company Ark Invest. Ark Invest has a $1 billion investment in the company, according to a report. The company has not responded to requests for comment on the report. For confidential support call the Samaritans on 08457 90 90 90, visit a local Samaritans branch or see www.samaritans.org for details. In the U.S. call the National Suicide Prevention Line on 1-800.
##### EV batteries may be giving consumers headaches. Here's why. (Sentiment: Neutral)
JD Power study found 266 problems per 100 electric vehicles versus 180 per 100 internal vehicles. Study found more problems with battery electric vehicles than internal combustion engines. Study was conducted by JD Power, a global consulting firm based in New York and Washington, D.C. The study was published in the Journal of Electric Vehicles. For more information, visit www.jdpower.com. For confidential support, call the Samaritans on 08457 90 90 90, visit a local Samaritans branch or see www.samaritans.org.
##### Dow Jones Futures: Micron Falls On Earnings; Tesla, Amazon In Buy Zones (Sentiment: Positive)
The major indexes rose slightly Wednesday despite modestly weak breadth. The Dow Jones Industrial Average, S&P 500, Nasdaq and Nasdaq all rose 0.2 percent. The Nasdaq was up 0.3 percent, while the Dow Jones was down 0.1 percent, and the Russell 2000 was down 1.1 per cent. The three major U.S. stock indexes were all up about 0.5 per cent on Wednesday. The Russell 2000, however, fell 0.7 per cent, and so did the Russell 1000.
##### Traders have poured $5 billion into leveraged Nvidia ETFs. They're up 425% even after the stock's big wipeout. (Sentiment: Positive)
The success of singlestock ETFs that track Nvidia has yet to spill over to other leveraged tech stocks. The success of leveraged technology stocks hasn't yet spilled over to leveraged singlestocks, either, as seen in this week's iShares MSCI Technology Index. The index is a leveraged exchange-traded fund that tracks technology stocks such as Nvidia, Twitter, Facebook, and others. The iShares Technology Index has a market capitalization of more than $1.2 billion.
#### Stock price data graph
![Stock price data before fine-tuning](https://github.com/Sayomphon/Stock-Analysis-using-FinBERTandBART/blob/d59009e070519b46a61852092ed2434fd44ad6ef/Stock%20price%20data%20before%20fine-tuning.PNG)
#### 15.) Load Datasets
The code utilizes the
load_dataset
function to load the IMDB dataset, which is commonly used for sentiment analysis tasks. The loaded dataset is stored in the variable dataset, making it available for further operations in the script.
```python
dataset = load_dataset('imdb')
```
#### 16.) Split into training and validation sets
The code extracts the text and label data from the training and validation (test) sets of the dataset. It assigns these extracted elements to the variables train_texts, train_labels, val_texts, and val_labels. This separation is crucial for training a model and evaluating its performance using distinct datasets.
```python
# Split into training and validation sets
train_texts = dataset['train']['text']
train_labels = dataset['train']['label']
val_texts = dataset['test']['text']
val_labels = dataset['test']['label']
```
#### 17.) Tokenizer
The code initializes a pre-trained BERT tokenizer using the Hugging Face library, specifically the 'bert-base-uncased' configuration. The tokenizer is stored in the variable tokenizer, making it ready to preprocess text data by converting it into token IDs that can be fed into a BERT model for various natural language processing tasks.
```python
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
```
#### 18.) Custom datasets
The code creates custom datasets for training and validation by utilizing a custom dataset class named CustomDataset. It takes in the text and label data, along with a tokenizer and a specified maximum sequence length of 128 tokens. This setup ensures that both the training and validation datasets are properly tokenized and formatted, making them ready for use in training and evaluating a machine learning model.
```python
train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length=128)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length=128)
```
#### 19.) Load the Pre-Trained Model
The code initializes a pre-trained BERT model for sequence classification by loading it from the Hugging Face Transformers library. The model version specified is 'bert-base-uncased', which treats text as case-insensitive. The num_labels=2 parameter configures the model for binary classification, meaning it can categorize inputs into two different classes. The loaded model, stored in the pretrained_model variable, is now ready for fine-tuning or direct use in classification tasks.
```python
# Load the pre-trained model
pretrained_model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
```
#### 20.) Evaluate the pre-trained model before fine-tuning
This code sets up an evaluation framework for assessing the performance of a pre-trained model before fine-tuning it. First, it initializes a MetricsCalculator and sets training arguments using TrainingArguments. It then creates a Trainer instance with the pre-trained model, evaluation dataset, tokenizer, and metrics calculation function. Finally, it evaluates the model and stores the results in the eval_results_before variable. This evaluation provides a baseline performance metric for the pre-trained model, which can be useful for comparison after further fine-tuning.
```python
# Evaluate the pre-trained model before fine-tuning

# Instantiate your MetricsCalculator
metrics_calculator = MetricsCalculator()

# Evaluate the pre-trained model before fine-tuning
training_args = TrainingArguments(
    output_dir='./results_before',
    per_device_eval_batch_size=16,
    logging_dir='./logs',
    fp16=True,
    gradient_accumulation_steps=2,
)

trainer_before = Trainer(
    model=pretrained_model,
    args=training_args,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=metrics_calculator.compute_metrics
)

eval_results_before = trainer_before.evaluate()
```
#### 21.) Print evaluation results before fine-tuning
The code snippet prints several key evaluation metrics for a model before the fine-tuning process. These metrics are stored in the eval_results_before dictionary and include loss, accuracy, precision, recall, and F1-score. Each metric provides insight into different aspects of the model's performance, offering a benchmark for later comparison with post-fine-tuning results. The structured approach helps ensure clarity and comprehensibility of the evaluation results.
```python
# Print evaluation results before fine-tuning
print("Evaluation results before fine-tuning:")
print(f"Loss: {eval_results_before['eval_loss']}")
print(f"Accuracy: {eval_results_before['eval_accuracy']}")
print(f"Precision: {eval_results_before['eval_precision']}")
print(f"Recall: {eval_results_before['eval_recall']}")
print(f"F1-score: {eval_results_before['eval_f1']}")
```
#### Output print evaluation results before fine-tuning
![Evaluation result before fine-tuning](https://github.com/Sayomphon/Stock-Analysis-using-FinBERTandBART/blob/45c8c189f3e60afa41326c1419455262c6af2b52/Evaluation%20result%20before%20fine-tuning.PNG)
#### 22.) Fine-tune the pre-trained model with 20 epochs
This code snippet aims to fine-tune a pre-trained model using a custom function named custom_fine_tune. The function is called with several arguments: pretrained_model, tokenizer, train_dataset, val_dataset, output_dir, and epochs. The pretrained_model is the model that will be fine-tuned, while the tokenizer is used to preprocess the input text data. The train_dataset is the dataset employed for training the model, adjusting its weights based on this data. The val_dataset is used to validate the model's performance during the fine-tuning process. The output_dir specifies the directory ./results_after_finetune where the results of the fine-tuning process will be stored. The epochs parameter indicates that the fine-tuning should run for 20 epochs, meaning the model will iterate over the entire training dataset 20 times to refine its performance. Once the fine-tuning process is completed, the evaluation results are stored in the eval_results_after variable, which can then be used to assess any improvements in the model's performance.
```python
# Fine-tune the pre-trained model with 20 epoch
eval_results_after = custom_fine_tune(pretrained_model, tokenizer, train_dataset, val_dataset, output_dir='./results_after_finetune', epochs=20)
```
#### Output while tuning
![While tuning](https://github.com/Sayomphon/Stock-Analysis-using-FinBERTandBART/blob/c0185d1e6fcd6f8f2d423f0fe7c343593ef488eb/While%20tuning.PNG)
#### 23.) Print evaluation results after fine-tuning
The code snippet prints several key evaluation metrics for a model after the fine-tuning process. These metrics are stored in the eval_results_after dictionary and include loss, accuracy, precision, recall, and F1-score. Each metric provides insight into different aspects of the model's performance post-fine-tuning. Printing these metrics allows for the assessment and comparison of the model's improvements and effectiveness after the fine-tuning process.
```python
# Print evaluation results after fine-tuning
print("Evaluation results after fine-tuning:")
print(f"Loss: {eval_results_after['eval_loss']}")
print(f"Accuracy: {eval_results_after['eval_accuracy']}")
print(f"Precision: {eval_results_after['eval_precision']}")
print(f"Recall: {eval_results_after['eval_recall']}")
print(f"F1-score: {eval_results_after['eval_f1']}")
```
#### Output print evaluation results after fine-tuning
![Evaluation result after fine-tuning](https://github.com/Sayomphon/Stock-Analysis-using-FinBERTandBART/blob/ec4b6f815967e6421a9195f6e4d7ad1a7e6ef9d8/Evaluation%20result%20after%20fine-tuning.PNG)
#### 24.) Load the fine-tuned model
This code snippet loads a fine-tuned model and its corresponding tokenizer from the directory ./results_after_finetune. The fine_tuned_model variable uses the AutoModelForSequenceClassification class from the transformers library to load the pre-trained model weights and configuration, which have been fine-tuned for sequence classification tasks. Similarly, the fine_tuned_tokenizer variable uses the AutoTokenizer class to load the tokenizer that was fine-tuned alongside the model. The from_pretrained method in both cases specifies the directory containing the fine-tuned model and tokenizer, allowing them to be loaded for further use.
```python
# Load the fine-tuned model
fine_tuned_model = AutoModelForSequenceClassification.from_pretrained('./results_after_finetune')
fine_tuned_tokenizer = AutoTokenizer.from_pretrained('./results_after_finetune')
```
#### 25.) Save model after fine-tuning
This code snippet saves both the fine-tuned model and its associated tokenizer to the directory ./saved_model. The fine_tuned_model.save_pretrained('./saved_model') method saves the model weights and configuration to the specified directory, making it possible to reload and use the model at a later time. Similarly, the fine_tuned_tokenizer.save_pretrained('./saved_model') method saves the tokenizer's configuration and vocabulary files to the same directory. This ensures that the tokenizer, tuned alongside the model, can also be reloaded and used later. By doing so, both the model and tokenizer are preserved, facilitating consistent future use or deployment.
```python
# Save model after fine-tuning
fine_tuned_model.save_pretrained('./saved_model')
fine_tuned_tokenizer.save_pretrained('./saved_model')
```
#### Result
![Save model after fine-tuning](https://github.com/Sayomphon/Stock-Analysis-using-FinBERTandBART/blob/a43a9802a68697464485fb9ab555d027e9d142e4/Save%20model%20after%20fine-tuning.PNG)
#### 26.) Function for stock analysis after fine-tuning
This code defines a function analyze_stock_after_finetune that performs stock analysis using a fine-tuned model for sentiment analysis of news articles. The function takes three parameters: alpha_vantage_api_key, news_api_key, and symbol. It first fetches the latest stock data using the Alpha Vantage API and retrieves the most recent closing price. Next, it fetches the latest news articles related to the stock symbol using a news API. The function then processes the first five news articles, cleans the text, tokenizes them, and uses the fine-tuned model to predict the sentiment of each article. The sentiment is labeled as "Positive," "Negative," or "Neutral" and added to a list along with the article title and cleaned description. The function creates a detailed prompt that incorporates market trends, stock performance, and news data to generate comprehensive investment advice. This prompt is passed to a function named generate_advice, which produces the investment advice based on the provided context. Finally, the function returns the latest stock price, the generated investment advice, and the list of cleaned news summaries along with their sentiment labels. This combined analysis offers a comprehensive view of the stock's current status and future potential.
```python
# Function for stock analysis after fine-tuning
def analyze_stock_after_finetune(alpha_vantage_api_key, news_api_key, symbol):
    stock_data = get_stock_data(alpha_vantage_api_key, symbol)
    latest_price = stock_data['close'].iloc[0]

    news_data = get_latest_news(symbol, news_api_key)
    news_summary_cleaned = []

    if news_data and 'articles' in news_data:
        for article in news_data['articles'][:5]:
            description = clean_text(article['description'])
            inputs = fine_tuned_tokenizer(description, return_tensors='pt', padding=True, truncation=True, max_length=512)
            outputs = fine_tuned_model(**inputs)
            sentiment = torch.argmax(outputs.logits, dim=1).item()
            sentiment_label = "Positive" if sentiment == 1 else "Negative" if sentiment == 0 else "Neutral"

            news_summary_cleaned.append((article['title'], sentiment_label, description))

    advice_prompt = (
        f"Current market trends show a bullish movement in the tech sector. {symbol} has been gaining momentum. "
        f"News articles suggest significant investor interest. Given the current price of {symbol}, generate detailed investment advice "
        f"considering market trends, stock performance, and financial news. "
        f"Provide a thorough analysis including potential risks, market conditions, and long-term investment potential. "
        f"Discuss the stock's historical performance, recent news impact, and any upcoming events that may influence its price. "
        f"Also, offer a strategic plan for both short-term and long-term investors."
    )

    advice = generate_advice(advice_prompt)
    return latest_price, advice, news_summary_cleaned
```
#### 27.) Create UI after fine-tuning
This code sets up a user interface to perform stock analysis using input widgets for API keys and stock symbols. It initializes text input fields for the Alpha Vantage API key, the News API key, and the stock symbol, along with an output area to display results. A function on_button_click is defined to handle button clicks, which clears previous output, retrieves user inputs, validates them, and performs stock analysis using the analyze_stock_after_finetune function. The results include the latest stock price, generated investment advice, and summarized news articles with sentiment analysis, displayed in the output area. Additionally, a candlestick chart visualizing the stock price data is generated and shown. If inputs are incomplete or an error occurs, appropriate messages are displayed. Finally, a button is created and configured to invoke the click handler function when clicked, completing the UI setup for stock analysis.
```python
# Create UI after fine-tuning
alpha_vantage_api_key_widget = widgets.Text(value='', placeholder='Enter your Alpha Vantage API Key', description='Alpha Vantage API Key:')
news_api_key_widget = widgets.Text(value='', placeholder='Enter your News API Key', description='News API Key:')
symbol_widget = widgets.Text(value='', placeholder='Enter stock symbol (e.g., AAPL)', description='Stock Symbol:')
output = widgets.Output()

def on_button_click(b):
    with output:
        output.clear_output()
        alpha_vantage_api_key = alpha_vantage_api_key_widget.value
        news_api_key = news_api_key_widget.value
        symbol = symbol_widget.value
        if alpha_vantage_api_key and news_api_key and symbol:
            try:
                latest_price, advice, news_summary_cleaned = analyze_stock_after_finetune(alpha_vantage_api_key, news_api_key, symbol)

                display(HTML(f'<h3>Current Price of {symbol}: {latest_price}</h3>'))
                display(HTML(f'<h4>Investment Advice:</h4><p>{advice}</p>'))
                display(HTML('<h4>Latest Financial News:</h4>'))

                for title, sentiment, summary in news_summary_cleaned:
                    display(HTML(f"<b>{title}</b> (Sentiment: {sentiment})<br>{summary}<br><br>"))

                # Visualize stock data separately
                stock_data = get_stock_data(alpha_vantage_api_key, symbol)
                fig = go.Figure(data=[go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['open'],
                    high=stock_data['high'],
                    low=stock_data['low'],
                    close=stock_data['close']
                )])
                fig.update_layout(title=f'Stock Price Data for {symbol}', yaxis_title='Price (USD)', xaxis_title='Time')
                fig.show()
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            print("Please enter the Alpha Vantage API key, News API key, and stock symbol.")

button = widgets.Button(description="Analyze Stock")
button.on_click(on_button_click)
```
#### 28.) Display UI after fine-tuning
This code snippet displays the user interface components for stock analysis in a Jupyter notebook or IPython environment. The display function is used to render the text input widgets for entering the Alpha Vantage API key, News API key, and stock symbol. It also displays the button labeled "Analyze Stock," which triggers the analysis process when clicked. Additionally, it renders the output widget, which will be used to present the results of the analysis, such as the latest stock price, investment advice, and summarized news articles with sentiment analysis. This UI setup provides an interactive interface for users to input data and view the analysis results.
```python
# Display UI before fine-tuning
display(alpha_vantage_api_key_widget, news_api_key_widget, symbol_widget, button, output)
```
#### Example UI use case
![Display UI after fine-tuning](https://github.com/Sayomphon/Stock-Analysis-using-FinBERTandBART/blob/4af80ad048a04ee6dd3df8db29202019aa372cfc/Display%20UI%20after%20fine-tuning.PNG)
#### Output after applying Analyze stock
#### Current Price of TSLA: 249.3800
#### Investment Advice:
Current market trends show a bullish movement in the tech sector. News articles suggest significant investor interest. Given the current price of TSLA, generate detailed investment advice considering market trends, stock performance, and financial news. Discuss the stock's historical performance, recent news impact, and any upcoming events that may influence its price. Also, offer a strategic plan for both short-term and long-term investors. For confidential support call the Samaritans on 08457 90 90 90 or visit a local Samaritans branch, see www.samaritans.org for details. In the U.S. call the National Suicide Prevention Lifeline on 1-800-273-8255 or visit http://www.suicidepreventionlifeline.org/. In the UK, call the helpline on 0800-847-9090 or click here for confidential support. For support in the United Kingdom, call 08457 909090 or visit the Samaritans on 08457 93 90 90.
#### Latest Financial News:
#### Tesla robotaxi won't be ready for scale until 2030: Analyst (Sentiment: Positive)
Tesla shares are trading higher on Friday despite a Bloomberg report indicating a twomonth delay in unveiling the company's highly anticipated new product. Tesla TSLA shares are Trading Higher on Friday Despite a Bloomberg Report indicating a Twomonth Delay in unveiling its highly anticipated product. The company is expected to unveil the new product on March 18. The stock is trading up 1.7% to $264.50 per share in Friday's pre-market trading. The market closed down 0.4% on Thursday.
#### Cathie Wood says she wouldn't have sold Nvidia stake 'had we known that the market was going to reward it' (Sentiment: Negative)
Cathie Woods investment fund Ark Invest sold at least 45 million worth of Nvidia stock this year per The Wall Street Journal. Cathie Woods invested in Nvidia through her company Ark Invest. Ark Invest has a $1 billion investment in the company, according to a report. The company has not responded to requests for comment on the report. For confidential support call the Samaritans on 08457 90 90 90, visit a local Samaritans branch or see www.samaritans.org for details. In the U.S. call the National Suicide Prevention Line on 1-800.
#### EV batteries may be giving consumers headaches. Here's why. (Sentiment: Neutral)
JD Power study found 266 problems per 100 electric vehicles versus 180 per 100 internal vehicles. Study found more problems with battery electric vehicles than internal combustion engines. Study was conducted by JD Power, a global consulting firm based in New York and Washington, D.C. The study was published in the Journal of Electric Vehicles. For more information, visit www.jdpower.com. For confidential support, call the Samaritans on 08457 90 90 90, visit a local Samaritans branch or see www.samaritans.org.
#### Dow Jones Futures: Micron Falls On Earnings; Tesla, Amazon In Buy Zones (Sentiment: Positive)
The major indexes rose slightly Wednesday despite modestly weak breadth. The Dow Jones Industrial Average, S&P 500, Nasdaq and Nasdaq all rose 0.2 percent. The Nasdaq was up 0.3 percent, while the Dow Jones was down 0.1 percent, and the Russell 2000 was down 1.1 per cent. The three major U.S. stock indexes were all up about 0.5 per cent on Wednesday. The Russell 2000, however, fell 0.7 per cent, and so did the Russell 1000.
#### Traders have poured $5 billion into leveraged Nvidia ETFs. They're up 425% even after the stock's big wipeout. (Sentiment: Positive)
The success of singlestock ETFs that track Nvidia has yet to spill over to other leveraged tech stocks. The success of leveraged technology stocks hasn't yet spilled over to leveraged singlestocks, either, as seen in this week's iShares MSCI Technology Index. The index is a leveraged exchange-traded fund that tracks technology stocks such as Nvidia, Twitter, Facebook, and others. The iShares Technology Index has a market capitalization of more than $1.2 billion.
#### Stock price data graph
![Stock price data after fine-tuning](https://github.com/Sayomphon/Stock-Analysis-using-FinBERTandBART/blob/69aa52ba66c0b279a9784b87304d77487a21ac6a/Stock%20price%20data%20after%20fine-tuning.PNG)
#### 29.) Load saved model
This code snippet loads a fine-tuned model and its associated tokenizer from the directory specified by the saved_model_path variable ('./saved_model'). The fine_tuned_model variable uses the AutoModelForSequenceClassification class from the transformers library to load the model's weights and configuration, which have been fine-tuned for sequence classification tasks. Similarly, the fine_tuned_tokenizer variable uses the AutoTokenizer class to load the tokenizer configuration and vocabulary that were fine-tuned alongside the model. By invoking the from_pretrained method with the specified directory path, both the model and tokenizer are reloaded, making them ready for further use in tasks such as text classification or sentiment analysis.
```python
# Load saved model
saved_model_path = './saved_model'
fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(saved_model_path)
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
```
