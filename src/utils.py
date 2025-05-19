import re
import string
from nltk.corpus import stopwords

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text
