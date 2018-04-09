from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import re



tv = TfidfVectorizer(stop_words=None)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


''' remove punctuations '''
def clean_punct(text):
    text = str(text)
    text = text.lower()

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text


''' clean sentence '''
def clean(sent):
    lemmo = []
    sent = clean_punct(sent)
    for word in word_tokenize(sent):
        if word not in stop_words:
            lemmo.append(lemmatizer.lemmatize(word))
    return " ".join(lemmo)


def similarity(sent1, sent2):
    clean_sent1 = clean(sent1)
    clean_sent2 = clean(sent2)
    val = tv.fit_transform([clean_sent1,clean_sent2]).toarray()
    return cosine_similarity(val[0], val[1])


''' Initialize output file '''
op = pd.DataFrame(columns=["test_id", "probability"])



count = 1
while True:
    sent1 = input()
    sent2 = input()
    try:
        score = similarity(sent1, sent2)
    except ValueError:
        score = [[0]]
    op = op.set_value(value=[count, score[0][0]], index=len(op), col=["test_id", "probability"])
    
    #write results to file
    op.to_csv("results.csv", index=None)
    count += 1




