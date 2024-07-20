## Function to render a user interface where users can enter their Alpha Vantage API key, News API key, and stock symbol.
```python
# Display UI
display(alpha_vantage_api_key_widget, news_api_key_widget, symbol_widget, button, output)
```
  - This line of code is responsible for displaying the user interface (UI) elements that have been defined earlier in the script.
  - alpha_vantage_api_key_widget: This is a text input widget for the user to enter their Alpha Vantage API key.
  - news_api_key_widget: Another text input widget where the user can input their News API key.
  - symbol_widget: A text input widget that allows the user to enter the stock symbol they want to analyze.
  - button: A button widget labeled "Analyze Stock" that the user can click to initiate the stock analysis.
  - output: This widget is used to display the results of the stock analysis, including the current stock price, investment advice, financial news, and a candlestick chart of the stock price data.
