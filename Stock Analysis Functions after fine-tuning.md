## This code defines a function analyze_stock_after_finetune that performs stock analysis using a fine-tuned model for sentiment analysis of news articles.
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
Defines a function named analyze_stock_after_finetune to perform stock analysis using a fine-tuned model.
```python
def analyze_stock_after_finetune(alpha_vantage_api_key, news_api_key, symbol):
```
  - alpha_vantage_api_key: An API key used to access stock data from the Alpha Vantage service.
  - news_api_key: An API key used to retrieve the latest news articles.
  - symbol: The stock symbol for which the analysis is being performed.
### Fetch Stock Data
Fetches stock data and retrieves the latest closing price.
```python
    stock_data = get_stock_data(alpha_vantage_api_key, symbol)
    latest_price = stock_data['close'].iloc[0]
```
  - stock_data: Holds the stock data fetched using the get_stock_data function.
  - latest_price: The most recent closing price of the stock.
### Fetch News Data
Fetches the latest news articles and initializes an empty list to store cleaned news summaries.
```python
    news_data = get_latest_news(symbol, news_api_key)
    news_summary_cleaned = []
```
  - news_data: Holds the news articles fetched using the get_latest_news function.
  - news_summary_cleaned: An empty list to store summaries of news articles along with their sentiment labels.
### Process News Articles
Processes the first five news articles, cleans the text, tokenizes it, predicts sentiment using the fine-tuned model, and stores the results.
```python
    if news_data and 'articles' in news_data:
        for article in news_data['articles'][:5]:
            description = clean_text(article['description'])
            inputs = fine_tuned_tokenizer(description, return_tensors='pt', padding=True, truncation=True, max_length=512)
            outputs = fine_tuned_model(**inputs)
            sentiment = torch.argmax(outputs.logits, dim=1).item()
            sentiment_label = "Positive" if sentiment == 1 else "Negative" if sentiment == 0 else "Neutral"

            news_summary_cleaned.append((article['title'], sentiment_label, description))
```
  - description: The cleaned text of the article's description.
  - inputs: The tokenized input ready for sentiment analysis.
  - outputs: The model's output logits for the sentiment analysis.
  - sentiment: The predicted sentiment index (0 for Negative, 1 for Positive, or another value).
  - sentiment_label: The human-readable label of the sentiment.
  - news_summary_cleaned: Appended with tuples of article title, sentiment label, and cleaned description.
### Generate Investment Advice
    advice_prompt = (
        f"Current market trends show a bullish movement in the tech sector. {symbol} has been gaining momentum. "
        f"News articles suggest significant investor interest. Given the current price of {symbol}, generate detailed investment advice "
        f"considering market trends, stock performance, and financial news. "
        f"Provide a thorough analysis including potential risks, market conditions, and long-term investment potential. "
        f"Discuss the stock's historical performance, recent news impact, and any upcoming events that may influence its price. "
        f"Also, offer a strategic plan for both short-term and long-term investors."
    )
  - advice_prompt: A string that details the context and requirements for generating investment advice.
### Call Advice Generation Function
Generates detailed investment advice based on the constructed prompt.
    advice = generate_advice(advice_prompt)
  - advice: The generated investment advice.
### Return Results
Returns the latest stock price, the investment advice, and the cleaned news summaries.
```python
    return latest_price, advice, news_summary_cleaned
```
  - Return Values
    - latest_price: The most recent closing price of the stock.
    - advice: Detailed investment advice.
    - news_summary_cleaned: A list of tuples containing article titles, sentiment labels, and cleaned descriptions.
