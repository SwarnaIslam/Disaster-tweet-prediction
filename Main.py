import pandas as pd
import nltk, re, string
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

df = pd.read_csv('train.csv')

stop = set(stopwords.words('english'))
lemma = WordNetLemmatizer()


def cleanTweet(txt):
    txt = txt.lower()
    words = nltk.word_tokenize(txt)
    words = ' '.join([lemma.lemmatize(word) for word in words if word not in (stop)])
    text = "".join(words)
    txt = re.sub('[^a-z]', ' ', text)
    return txt


df['cleaned_tweets'] = df['text'].apply(cleanTweet)

y=df.target
x=df.cleaned_tweets
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.20,stratify=y, random_state=0)


plt.show()
