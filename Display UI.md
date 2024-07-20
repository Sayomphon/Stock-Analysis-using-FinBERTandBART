## Function to render a user interface where users can enter their Alpha Vantage API key, News API key, and stock symbol.
```python
# Display UI
display(alpha_vantage_api_key_widget, news_api_key_widget, symbol_widget, button, output)
```
This line of code is responsible for displaying the user interface (UI) elements that have been defined earlier in the script.
  - alpha_vantage_api_key_widget: This is a text input widget for the user to enter their Alpha Vantage API key.
  - news_api_key_widget: Another text input widget where the user can input their News API key.
  - symbol_widget: A text input widget that allows the user to enter the stock symbol they want to analyze.
  - button: A button widget labeled "Analyze Stock" that the user can click to initiate the stock analysis.
  - output: This widget is used to display the results of the stock analysis, including the current stock price, investment advice, financial news, and a candlestick chart of the stock price data.
### Example UI use case
![Display UI before fine-tuning](https://github.com/Sayomphon/Stock-Analysis-using-FinBERTandBART/blob/bcc278dce82d0ed339873e211168cb172a4dea31/Display%20UI.PNG)
### Output after applying Analyze stock
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
### Stock price data graph
![Stock price data before fine-tuning](https://github.com/Sayomphon/Stock-Analysis-using-FinBERTandBART/blob/d59009e070519b46a61852092ed2434fd44ad6ef/Stock%20price%20data%20before%20fine-tuning.PNG)



