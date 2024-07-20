## The code analyzes the specified stock and displays the latest price, investment advice, and relevant financial news.
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
### Creating the UI Elements
Defines text input widgets for collecting the Alpha Vantage API key, News API key, and the stock symbol. Initializes an output widget to display the results.
```python
alpha_vantage_api_key_widget = widgets.Text(value='', placeholder='Enter your Alpha Vantage API Key', description='Alpha Vantage API Key:')
news_api_key_widget = widgets.Text(value='', placeholder='Enter your News API Key', description='News API Key:')
symbol_widget = widgets.Text(value='', placeholder='Enter stock symbol (e.g., AAPL)', description='Stock Symbol:')
output = widgets.Output()
```
  - alpha_vantage_api_key_widget: A text widget for the user to input their Alpha Vantage API key.
  - news_api_key_widget: A text widget for the user to input their News API key.
  - symbol_widget: A text widget for the user to input the stock symbol they want to analyze.
  - output: An output widget to display results and messages.
### Handling Button Click Event
This part starts a function that handles button click events. It captures the input values from the widgets and clears previous outputs.
```python
def on_button_click(b):
    with output:
        output.clear_output()
        alpha_vantage_api_key = alpha_vantage_api_key_widget.value
        news_api_key = news_api_key_widget.value
        symbol = symbol_widget.value
```
  - on_button_click is a function that handles the button click event.
  - with output: Allows redirection of output to the output widget.
  - output.clear_output(): Clears any previous output in the output widget.
  - Retrieves values from the input text widgets and stores them in variables: alpha_vantage_api_key, news_api_key, and symbol.
### Fetching and Displaying Stock Data and News
If all required inputs are provided, it calls analyze_stock_before_finetune to fetch stock data and news. It then displays the current stock price, investment advice, and summaries of the latest financial news.
```python
        if alpha_vantage_api_key and news_api_key and symbol:
            latest_price, advice, news_summary_cleaned, stock_data = analyze_stock_before_finetune(alpha_vantage_api_key, news_api_key, symbol)
            display(HTML(f'<h3>Current Price of {symbol}: {latest_price}</h3>'))
            display(HTML(f'<h4>Investment Advice:</h4><p>{advice}</p>'))
            display(HTML('<h4>Latest Financial News:</h4>'))
            for title, sentiment, summary, embeddings in news_summary_cleaned:
                #display(HTML(f"<b>{title}</b> (Sentiment: {sentiment})<br>{summary}<br>Embeddings: {embeddings}<br><br>"))
                display(HTML(f"<b>{title}</b> (Sentiment: {sentiment})<br>{summary}<br><br>"))
```
  - If all required inputs (alpha_vantage_api_key, news_api_key, and symbol) are provided:
    - Calls the analyze_stock_before_finetune function with the provided inputs to get stock analysis results.
    - Displays the current stock price using display(HTML(...)).
    - Displays the generated investment advice.
    - Iterates through the summarized news articles and displays their titles, sentiment labels, and summaries in HTML format.
### Visualizing Stock Data with a Candlestick Chart
This part creates and displays a candlestick chart for the stock data using Plotly. It also manages error handling by prompting the user to input all required keys if any are missing
```python
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
```
  - Constructs a candlestick chart using Plotly (go.Figure).
    - go.Candlestick plots the open, high, low, and close prices on the x-axis with time as the y-axis.
    - Updates the layout to include titles and axis labels.
    - Shows the generated chart.
  - If any of the required inputs are missing:
    - Prints a message prompting the user to input all necessary details.
### Button Creation and Event Binding
```python
button = widgets.Button(description="Analyze Stock")
button.on_click(on_button_click)
```
  - Button Initialization:
    - button = widgets.Button(description="Analyze Stock"): Creates a button widget with the label "Analyze Stock".
  - Event Binding:
    - button.on_click(on_button_click): Binds the on_button_click function to the button's click event. When the button is clicked, the function is executed.
