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
  - The analyze_stock_before_finetune function takes three parameters:
    - alpha_vantage_api_key: API key for Alpha Vantage to fetch stock data.
    - news_api_key: API key to fetch latest news (using a news API).
    - symbol: The stock symbol to be analyzed.
  - stock_data is retrieved using the get_stock_data function which fetches stock data for the given symbol.
  - latest_price is assigned the closing price of the latest trading day from the fetched stock data.
### Fetching Latest News

```python
    # Fetch latest news
    news_data = get_latest_news(symbol, news_api_key)
    news_summary_cleaned = []
```
  - news_data is fetched using the get_latest_news function which retrieves news articles related to the given stock symbol.
  - news_summary_cleaned is an empty list to store processed news summaries, sentiment labels, and embeddings.
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
  - If news_data contains articles, it iterates through the first 5 articles.
  - description is cleaned using the clean_text function.
  - inputs are tokenized using finbert_tokenizer and converted to a PyTorch tensor with padding and truncation.
  - outputs are generated using the finbert_model to get the logits.
  - sentiment is determined by applying argmax on the logits to get the sentiment label (Positive, Negative, or Neutral).
  - summary of the news is generated using the summarize_text function.
  - embeddings for the cleaned text are generated using the get_embeddings function.
  - Append a tuple of (article['title'], sentiment_label, summary, embeddings) to news_summary_cleaned.
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
  - advice_prompt is a detailed prompt constructed with information about the stock symbol, its current price, and other contextual information asking for comprehensive investment advice.
  - advice is generated by passing the advice_prompt to the generate_advice function, which uses a BART model to create detailed investment recommendations.
### Returning Results
Returns the latest price, investment advice, processed news summaries, and complete stock data.
```python
    return latest_price, advice, news_summary_cleaned, stock_data
```
  - latest_price: The latest closing price of the stock.
  - advice: The generated investment advice.
  - news_summary_cleaned: List of tuples containing news titles, sentiment labels, summaries, and embeddings.
  - stock_data: Comprehensive stock data fetched earlier.
