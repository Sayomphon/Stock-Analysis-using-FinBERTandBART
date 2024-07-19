# Stock-Analysis-using-FinBERTandBART

## Code Summary
This project is designed to fetch stock data and financial news, predict trends, and provide investment advice using Machine Learning models through Transformers. It uses various libraries to process the data, fine-tune models, and create an interactive UI for users to input their API keys and stock symbols.

### Main Functionality of the Code

1. Installing and Enabling Libraries:
  The code installs and enables essential libraries such as requests, pandas, transformers, torch, plotly, ipywidgets, and others for data fetching, processing, and UI rendering.
2. Fetching Stock Data:
  The get_stock_data function fetches minute-level stock data from the Alpha Vantage API and converts the data into a manageable DataFrame format.
3. Fetching Financial News:
  The get_latest_news function leverages the News API to fetch the latest financial news linked to the given stock symbol provided by the user.
4. Text Processing and Cleaning:
  clean_text function cleans the text by removing URLs, emails, and special characters.
5. Loading and Using Machine Learning Models:
  The code loads various Transformer models, such as finbert for financial sentiment analysis, bart for summarizing news, and sentence_transformer for generating text embeddings.
6. Creating Custom Datasets:
  CustomDataset is a class used for creating and managing the dataset that will be used to fine-tune the BERT model by tokenizing text and preparing it for model processing.
7. Model Fine-tuning:
  The custom_fine_tune function fine-tunes the pre-trained BERT model on custom datasets, using predefined training arguments.
8.Creating an Interactive UI:
  The code uses ipywidgets to create a form where users can input their Alpha Vantage API Key, News API Key, and the stock symbol.
A button is available for users to click to perform stock analysis and display the results.

### Use Case
#### Investors:
  - Usage: Investors can use this code to fetch stock data and news related to a specific stock, such as AAPL (Apple Inc.), and receive a market trend prediction and investment advice.
  - Outcome: The investor will receive information about the latest stock price, summarized key news articles, and the sentiment analysis of those news articles (positive, negative, neutral), along with investment advice.
#### Financial Institutions:
  - Usage: Financial institutions can use this code to fetch data and perform investment analysis for their clients, to provide accurate and trend-based investment advice.
  - Outcome: Institutions can provide daily or weekly investment recommendations to their clients and adjust investment strategies based on the real-time market situation to ensure optimal returns for their clients.
#### Software Developers:
  - Usage: Developers can use this code as a foundation to build more complex and comprehensive systems for stock analysis and investment advice.
  - Customization: The code can be customized or extended to include additional functionalities, such as aggregating data from multiple sources or using custom Machine Learning models.
##### This code serves as a fairly comprehensive system for fetching and analyzing financial data, which can be easily extended for use in larger applications.

#### License
  This project is licensed under the MIT License - see the LICENSE.md file for details.

#### Acknowledgments
  - Special thanks to Hugging Face for providing the Transformers library.
  - Alpha Vantage and News API for offering financial data and news services.
#### Contact
  For any questions or feedback, please open an issue in this repository.
  You can copy and paste the above content into a `README.md` file for your GitHub repository. Customize it further as needed to fit your specific needs and details about your project.
