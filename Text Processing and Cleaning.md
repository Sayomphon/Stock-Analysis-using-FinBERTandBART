## This function clean_text is used to clean up a given text
### Defining the function:
def clean_text(text):

This line defines a function named clean_text that takes one parameter text.
Removing URLs:
text = re.sub(r'http\S+', '', text)

This line uses the re.sub function from the re (regular expression) module to find and remove all URLs from the text. The pattern r'http\S+' matches any substring that starts with "http" followed by any number of non-whitespace characters.
Removing email addresses:
text = re.sub(r'\S+@\S+', '', text)

This line finds and removes all email addresses from the text. The pattern r'\S+@\S+' matches any sequence of non-whitespace characters (like an email address) that contains an "@" symbol.
Removing special characters:
text = re.sub(r'[^A-Za-z0-9\s]+', '', text)

This line removes any character that is not an uppercase letter, a lowercase letter, a digit, or a whitespace. The pattern r'[^A-Za-z0-9\s]+' matches any character that is not in the specified set and removes it from the text.
Stripping whitespace:
text = text.strip()

This line removes any leading or trailing whitespace from the cleaned text.
Returning the cleaned text:
return text

Finally, the function returns the cleaned text.
