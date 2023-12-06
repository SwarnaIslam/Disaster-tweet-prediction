import re


def remove_links(text):
    url_pattern = r'https?://\S+|www\.\S+|t\.co/\w+'

    # Use the re.sub() function to replace URLs with an empty string
    text_without_links = re.sub(url_pattern, '', text)

    return text_without_links


# Example usage:
tweet = "Weather forecast for Thailand  A Whirlwind is coming ...2 september https://t.co/rUKjYjG9oQ."
cleaned_tweet = remove_links(tweet)
print(cleaned_tweet)
