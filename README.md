# Stock-Analysis-using-FinBERTandBART

## Code Summary
### This project is designed to fetch stock data and financial news, predict trends, and provide investment advice using Machine Learning models through Transformers. It uses various libraries to process the data, fine-tune models, and create an interactive UI for users to input their API keys and stock symbols.

### 1. Installation and setup
#### 1.) Install necessary libraries
  !pip install requests pandas transformers ipywidgets torch plotly sentence-transformers datasets accelerate bitsandbytes
#### 2.) Enable IPyWidgets for Google Colab
  from google.colab import output
  output.enable_custom_widget_manager() 

### 2. Main Functionality of the Code
#### 1.) Installing and Enabling Libraries:
  The code installs and enables essential libraries such as requests, pandas, transformers, torch, plotly, ipywidgets, and others for data fetching, processing, and UI rendering.
#### 2.) Fetching Stock Data:
  The get_stock_data function fetches minute-level stock data from the Alpha Vantage API and converts the data into a manageable DataFrame format.
#### 3.) Fetching Financial News:
  The get_latest_news function leverages the News API to fetch the latest financial news linked to the given stock symbol provided by the user.
#### 4.) Text Processing and Cleaning:
  clean_text function cleans the text by removing URLs, emails, and special characters.
#### 5.) Loading and Using Machine Learning Models:
  The code loads various Transformer models, such as finbert for financial sentiment analysis, bart for summarizing news, and sentence_transformer for generating text embeddings.
#### 6.) Creating Custom Datasets:
  CustomDataset is a class used for creating and managing the dataset that will be used to fine-tune the BERT model by tokenizing text and preparing it for model processing.
#### 7.) Model Fine-tuning:
  The custom_fine_tune function fine-tunes the pre-trained BERT model on custom datasets, using predefined training arguments.
#### 8.) Creating an Interactive UI:
  The code uses ipywidgets to create a form where users can input their Alpha Vantage API Key, News API Key, and the stock symbol.
#### A button is available for users to click to perform stock analysis and display the results.

### 3. Use Case
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

### 4. Additional
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
### 1.) Installing Necessary Libraries
To install all the required libraries, run the following command
```python
# This command installs all the essential Python libraries needed for the project.
!pip install requests pandas transformers ipywidgets torch plotly sentence-transformers datasets accelerate bitsandbytes
```
### 2.) Enabling ipywidgets for Google Colab
If you're using Google Colab, enable ipywidgets by running the following code
```python
# This code snippet enables the custom widget manager for ipywidgets within Google Colab, allowing the creation and display of interactive widgets in the notebook.
from google.colab import output
output.enable_custom_widget_manager()
```
## 2. Main Functionality of the Code
### 1.) Installing and Enabling Libraries
The following libraries and modules are imported for the project
This project imports multiple libraries required for building an application related to applied NLP, stock analysis, and creating interactive interfaces. Detailed steps and explanations are provided in the related sections to help understand the usage and functionality of each library.
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
#### 1. HTTP Requests
```python
import requests
```
Makes HTTP requests for fetching data from web servers, such as stock data from an API.
#### 2. Data Analysis
```python
import pandas as pd
```
Manages and analyzes tabular data. pd is a commonly used alias for pandas.
#### 3. Transformer Models from Hugging Face
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, TrainingArguments, Trainer, default_data_collator, EarlyStoppingCallback, BartTokenizer, BartForConditionalGeneration
```
Handles advanced language models used in Natural Language Processing (NLP).
  - AutoTokenizer: Converts text into tokens for models.
  - AutoModelForSequenceClassification: Specialized in text classification tasks.
  - pipeline: Manages NLP task pipelines.
  - TrainingArguments: Sets parameters for training the model.
  - Trainer: Trains and evaluates the model.
  - default_data_collator: Collates data for training.
  - EarlyStoppingCallback: Stops training when no improvement is detected.
  - BartTokenizer, BartForConditionalGeneration: Used with BART models for text summarization.
#### 4. Deep Learning
```python
import torch
```
Loaded for numerical computations and deep learning tasks, leveraging GPU for efficient processing.
#### 5. Interactive Widgets
```python
import ipywidgets as widgets
```
Creates interactive widgets for Jupyter Notebooks.
#### 6. Display in Jupyter Notebook
```python
from IPython.display import display, HTML
```
Displays objects and HTML content in Jupyter Notebooks.
#### 7. Regular Expressions
```python
import re
```
Handles regular expressions for searching and managing text based on patterns.
#### 8. Data Visualization
```python
import plotly.graph_objects as go
```
Creates interactive graphs, such as candlestick charts for stock data visualization.
#### 9. Sentence Transformers
```python
from sentence_transformers import SentenceTransformer
```
Transforms sentences into vectors for mathematical analysis and machine learning.
#### 10. Ready-to-Use Datasets
```python
from datasets import load_dataset, load_metric
```
Loads datasets and evaluates NLP models with metrics.
  - load_dataset: Loads various datasets for training and testing.
  - load_metric: Loads metrics for model evaluation.
#### 11. Data Handling in PyTorch
```python
from torch.utils.data import DataLoader
```
Manages datasets for model training in PyTorch.
#### 12. Model Evaluation Metrics
```python
from sklearn.metrics import precision_recall_fscore_support
```
Evaluates and measures the performance of machine learning models, calculating precision, recall, and F1-score.
### 2.) Fetching Stock Data
Use to fetch intraday stock data for a given symbol using the Alpha Vantage API. Let's break it down step by step
```python
def get_stock_data(api_key, symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
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
```
#### 1. Function Definition
```python
def get_stock_data(api_key, symbol):
```
This line defines a function get_stock_data that takes two parameters:
  - api_key: The API key required to access the Alpha Vantage API.
  - symbol: The stock symbol for which you want to fetch data (e.g., 'AAPL' for Apple Inc.).
#### 2. Construct the API URL
```python
  url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={api_key}'
```
This line constructs the URL needed to fetch intraday data from the Alpha Vantage API. The URL includes:
  - function=TIME_SERIES_INTRADAY: Specifies the type of data to fetch (intraday stock data).
  - symbol={symbol}: The stock symbol.
  - interval=1min: The interval between data points (1 minute in this case).
  - apikey={api_key}: The API key for access.
#### 3. Make the API Request
```python
  response = requests.get(url)
```
This line sends an HTTP GET request to the Alpha Vantage API using the constructed URL. The response from the API is stored in the response variable.
#### 4. Parse the JSON Response
```python
  data = response.json()
```
This line parses the JSON response from the API into a Python dictionary named data.
#### 5. Convert the Data to a DataFrame
```python
  df = pd.DataFrame(data['Time Series (1min)']).T
```
This line converts the relevant part of the JSON data (the time series data) into a Pandas DataFrame.
  - data['Time Series (1min)']: Accesses the time series data from the JSON response.
  - .T: Transposes the DataFrame to have the dates/times as rows and the stock data (open, high, low, close, volume) as columns.
