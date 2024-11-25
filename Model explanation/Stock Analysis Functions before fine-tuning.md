## Function aims to analyze stock information and provide investment advice
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
### Function Definition
Fetches the latest stock data, including the closing price, for the given stock symbol using the Alpha Vantage API.
```python
def analyze_stock_before_finetune(alpha_vantage_api_key, news_api_key, symbol):
    stock_data = get_stock_data(alpha_vantage_api_key, symbol)
    latest_price = stock_data['close'][0]
```
  - Function Definition:
    - def analyze_stock_before_finetune(alpha_vantage_api_key, news_api_key, symbol): Defines a function named analyze_stock_before_finetune to analyze stock data and related news before fine-tuning a model.
  - Parameters:
    - alpha_vantage_api_key: The API key for accessing Alpha Vantage, a provider of real-time and historical market data.
    - news_api_key: The API key for accessing a news provider API.
    - symbol: The stock symbol (ticker) to be analyzed (e.g., AAPL for Apple Inc.).
  - Stock Data Retrieval and Latest Price:
    - stock_data = get_stock_data(alpha_vantage_api_key, symbol): Calls a function to retrieve stock data using the Alpha Vantage API.
    - latest_price = stock_data['close'][0]: Extracts the latest closing price from the retrieved stock data.
### Fetching Latest News

```python
    # Fetch latest news
    news_data = get_latest_news(symbol, news_api_key)
    news_summary_cleaned = []
```
  - news_data = get_latest_news(symbol, news_api_key): Calls a function to retrieve the latest news articles related to the stock symbol using a news provider API.
  - news_summary_cleaned = []: Initializes an empty list to store cleaned and summarized news data.
### Processing News Articles
Fetches and processes the latest news articles related to the stock symbol, analyzing their sentiment, summarizing the content, and generating text embeddings.
```python
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
```
  - News Articles Processing:
    - if news_data and 'articles' in news_data: Checks if the news data contains articles.
    - for article in news_data['articles'][:5]: Loops through the top 5 news articles.
  - Processing Each Article:
    - description = clean_text(article['description']): Cleans the text description of the news article.
    - inputs = finbert_tokenizer(description, return_tensors='pt', padding=True, truncation=True, max_length=512): Tokenizes the cleaned text using the FinBERT tokenizer.
    - outputs = finbert_model(**inputs): Passes the tokenized inputs through the FinBERT model to get sentiment outputs.
    - sentiment = torch.argmax(outputs.logits, dim=1).item(): Determines the sentiment (positive, negative, or neutral) by finding the index with the highest logit score.
    - sentiment_label = "Positive" if sentiment == 1 else "Negative" if sentiment == 0 else "Neutral": Converts the sentiment index to a readable label.
  - Summary and Embeddings:
    - summary = summarize_text(description): Calls a function to summarize the cleaned text using a summarization model.
    - embeddings = get_embeddings(description): Calls a function to get the embeddings of the cleaned text.
  - Append Processed Article Data:
    - news_summary_cleaned.append((article['title'], sentiment_label, summary, embeddings)): Appends the processed news data (title, sentiment label, summary, embeddings) to the list.
### Generating Investment Advice
Creates a detailed prompt and generates investment advice taking into account market trends, stock performance, and recent news.
```python
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
```
  - advice_prompt: Constructs a detailed prompt for generating investment advice, incorporating market trends, stock performance, financial news, risks, and strategic plans for investors.
  - advice = generate_advice(advice_prompt): Calls a function to generate investment advice based on the constructed prompt.
### Returning Results
Returns the latest price, investment advice, processed news summaries, and complete stock data.
```python
    return latest_price, advice, news_summary_cleaned, stock_data
```
  - latest_price: The latest closing price of the stock.
  - advice: The generated investment advice.
  - news_summary_cleaned: List of tuples containing news titles, sentiment labels, summaries, and embeddings.
  - stock_data: Comprehensive stock data fetched earlier.
