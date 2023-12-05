import pandas as pd
import nltk, re, string
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import model_selection
from nltk.stem import WordNetLemmatizer
from nltk.tree import Tree

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")

df = pd.read_csv('train.csv')

stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)
lemma = WordNetLemmatizer()

def penn_to_wordnet(tag):
    if tag.startswith('J'):
        return 'a'  # Adjective
    elif tag.startswith('V'):
        return 'v'  # Verb
    elif tag.startswith('N'):
        return 'n'  # Noun
    elif tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return 'n'  # Noun (default)

def cleanTweet(txt):
    txt = txt.lower()
    words = nltk.word_tokenize(txt)
    pos_tags = nltk.pos_tag(words)
    words = ' '.join([lemma.lemmatize(token, penn_to_wordnet(pos)) for token, pos in pos_tags if token not in (stop)])
    text = "".join(words)
    txt = re.sub('[^a-z]', ' ', text)
    return txt


def remove_links(text):
    url_pattern = r'https?://\S+|www\.\S+|t\.co/\w+'
    text_without_links = re.sub(url_pattern, '', text)
    text_without_links = re.sub("&amp", '', text_without_links)
    return text_without_links


def remove_ne(txt):
    words = nltk.word_tokenize(txt.lower())
    pos_tags = nltk.pos_tag(words)

    def contains_named_entity(subtree):
        return isinstance(subtree, Tree) and subtree.label() == 'NE'

    non_entities = [word for word, pos in pos_tags if not contains_named_entity(pos)]
    cleaned_text = ' '.join(non_entities)
    return cleaned_text


df['cleaned_tweets'] = df['text'].apply(remove_links)
df['cleaned_tweets'] = df['cleaned_tweets'].apply(cleanTweet)
df['cleaned_tweets'] = df['cleaned_tweets'].apply(remove_ne)

print(df.head(10))
from wordcloud import WordCloud

plt.figure(figsize=(20, 20))  # Text that is Disaster tweets
wc = WordCloud(max_words=1000, width=1600, height=800).generate(" ".join(df[df['target'] == 1].cleaned_tweets))
plt.imshow(wc, interpolation='bilinear')

y = df.target
x = df.cleaned_tweets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, stratify=y, random_state=0)

# term frequency and inverse document frequency
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, ngram_range=(1, 3))
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

pass_tf = PassiveAggressiveClassifier()
pass_tf.fit(tfidf_train, y_train)

kfold = model_selection.KFold(n_splits=10)
scoring = 'accuracy'

acc_pass = cross_val_score(estimator=pass_tf, X=tfidf_train, y=y_train, cv=kfold, scoring=scoring)
print(acc_pass.mean())

pred_pass3 = pass_tf.predict(tfidf_test)
CM = confusion_matrix(y_test, pred_pass3)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
specificity = TN / (TN + FP)

acc = accuracy_score(y_test, pred_pass3)
prec = precision_score(y_test, pred_pass3)
rec = recall_score(y_test, pred_pass3)
f1 = f1_score(y_test, pred_pass3)

mod1_results = pd.DataFrame([['PA', acc, prec, rec, specificity, f1]],
                            columns=['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1 Score'])
print(mod1_results)

sentences = [
    "Just happened a terrible car crash",
    "Heard about #earthquake is different cities, stay safe everyone.",
    "No I don't like cold!",
    "@RosieGray Now in all sincerety do you think the UN would move to Israel if there was a fraction of a chance of being annihilated?",
    "My heart is shaking with sorrow"
]

tfidf_trigram = tfidf_vectorizer.transform(sentences)

predictions = pass_tf.predict(tfidf_trigram)

for text, label in zip(sentences, predictions):
    if label == 1:
        target = "Disaster Tweet"
        print("text:", text, "\nClass:", target)
        print()
    else:
        target = "Normal Tweet"
        print("text:", text, "\nClass:", target)
        print()
plt.show()
