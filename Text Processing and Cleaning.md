## This function clean_text is used to clean up a given text
```python
# Clean text function
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)  # Remove special characters
    text = text.strip()
    return text
```
### Defining the function
```python
def clean_text(text):
```
  - Function Definition:
    - def clean_text(text): Defines a function named clean_text to clean and preprocess input text.
  - Parameters:
    - text: A string containing the text to be cleaned.
### Removing URLs
```python
    text = re.sub(r'http\S+', '', text)
```
  - Removing URLs
    re.sub(r'http\S+', '', text): Uses a regular expression to find and remove URLs from the text. The pattern 'http\S+' matches any substring that starts with 'http' followed by any number of non-whitespace characters.
### Removing email addresses
```python
    text = re.sub(r'\S+@\S+', '', text)
```
  - Removing Emails
    - re.sub(r'\S+@\S+', '', text): Uses a regular expression to find and remove email addresses from the text. The pattern '\S+@\S+' matches any email-like substring.
### Removing special characters
```python
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
```
  - Removing Special Characters
    - re.sub(r'[^A-Za-z0-9\s]+', '', text): Uses a regular expression to find and remove any special characters from the text, keeping only letters, numbers, and whitespace. The pattern [^A-Za-z0-9\s]+ matches any character that is not a letter, number, or whitespace.
### Stripping whitespace
```python      
      text = text.strip()
```
  - Stripping Whitespace
    - text.strip(): Removes leading and trailing whitespace from the text.
### Returning the cleaned text
```python
    return text
```
Finally, the function returns the cleaned text.
