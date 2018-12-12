import os
import math
import csv
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

def main():

    # Dataset Reading
    doc = {}
    N = 1
    c1 = c2 = c3 = c4 = 0
    limit = 1250
    with open("../dataset/democratvsrepublicantweets/output.csv",encoding="Latin-1") as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if c1 == limit and c2 == limit:
                break

            if row[0] == "Republican" and c1 < limit:
                doc[N] = preprocess(row[2])
                doc[N].append("real republican")
                N += 1
                c1 += 1
            elif row[0] == "Democrat" and c2 < limit:
                doc[N] = preprocess(row[2])
                doc[N].append("real democrat")
                N += 1
                c2 += 1
            else:
                continue

    with open("../dataset/russian-troll-tweets/IRAhandle_tweets_1.csv",encoding="Latin-1") as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if c3 == limit and c4 == limit:
                break

            if row[len(row)-1] == "RightTroll" and c3 < limit:
                doc[N] = preprocess(row[2])
                doc[N].append("fake republican")
                N += 1
                c3 += 1
            elif row[len(row)-1] == "LeftTroll" and c4 < limit:
                doc[N] = preprocess(row[2])
                doc[N].append("fake democrat")
                N += 1
                c4 += 1
            else:
                continue

    # Term Frequency
    corpus = {}
    for d in doc:
        for w in doc[d]:
            if " " in w:
                continue
            elif w in corpus:
                corpus[w][d] += 1
            else:
                corpus[w] = {}
                for i in range(N):
                    corpus[w][i+1] = 0
                corpus[w][d] = 1


    # Vector Space
    if not os.path.exists("vector_space.csv"):
        with open('../vector_space.csv', 'w', newline='') as csv_td:
            writer = csv.writer(csv_td)
            writer.writerow(list(corpus.keys()))
            for d in doc:
                row = []
                for w in corpus:
                    row.append(corpus[w][d])
                row.append(doc[d][len(doc[d])-1])
                writer.writerow(row)

# preprocess of document
def preprocess( doc ):
    preprocessed = tokenize(doc)

    skip = ["RT"]
    preprocessed = [ w for w in preprocessed if w not in skip and "http" not in w ]

    preprocessed = normalize(preprocessed)
    preprocessed = lemmatize(preprocessed)
    preprocessed = stem(preprocessed)

    return preprocessed

# tokenization of document
def tokenize( doc ):

    # tokenizing
    tokenized = word_tokenize(doc)
    new_tokenized = []
    i = 1
    while i < len(tokenized):
        if tokenized[i-1] == '#':
            new_tokenized.append('#'+tokenized[i])
        elif tokenized[i] != '#':
            new_tokenized.append(tokenized[i])
        i+=1


    return new_tokenized

# normalization and filtration of document
def normalize( tokenized ):

    # remove punctuations
    normalized = [ word for word in tokenized if word.isalpha() or word[0] is '#' ]

    # removing stopwords
    stop_words = set(stopwords.words('english'))
    filtered = [ w for w in normalized if w not in stop_words ]

    return filtered

# lemmatization of document
def lemmatize( tokenized ):
    lemmatizer = WordNetLemmatizer()
    lemmatized = [ lemmatizer.lemmatize(w) for w in tokenized ]
    return lemmatized

# stemming of document
def stem( tokenized ):
    stemmer = PorterStemmer()
    stemmed = [ str(stemmer.stem(w)) for w in tokenized if is_ascii(str(stemmer.stem(w))) ]
    return stemmed

def is_ascii(s):
    return all(ord(c) < 128 for c in s)


if __name__ == "__main__":
    main()