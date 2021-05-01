import pandas as pd
import sklearn.datasets as skd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


import re
import numpy as np
from nltk.corpus import stopwords


def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

TAG_RE = re.compile(r'<[^>]+>')



def remove_tags(text):
    return TAG_RE.sub('', text)


def main():
    
    data_nspec_train = "nspec_train.tok"
    data_spec_train = "spec_train.tok"
    data_nspec_test = "nspec_test.tok"
    data_spec_test = "spec_test.tok"
    df_nspec_train = pd.read_csv(data_nspec_train, header=None)
    df_spec_train = pd.read_csv(data_spec_train, header=None)
    df_nspec_test = pd.read_csv(data_nspec_test, header=None)
    df_spec_test = pd.read_csv(data_spec_test, header=None)


    #Assign binary values to classify sentences, 0 for non-speculative and 1 for speculative. 
    #Clean dataframe with column headings to identify data of sentence or spec classifier. 
    df_nspec_train.columns = ["Sentences"]
    df_nspec_test.columns = ["Sentences"]
    df_spec_train.columns = ["Sentences"]
    df_spec_test.columns = ["Sentences"]
    vals = np.zeros(len(df_nspec_train), dtype=int)
    df_nspec_train["Speculative"] = vals
    vals = np.zeros(len(df_nspec_test), dtype=int)
    df_nspec_test["Speculative"] = vals
    vals = [1]*len(df_spec_train)
    df_spec_train["Speculative"] = vals
    vals = [1]*len(df_spec_test)
    df_spec_test["Speculative"] = vals



    #Combine 4 data sources into 2 dataframes, one for test data and one for train data.
    #Each dataframe contains sentences with value 0 for non-spec and 1 for spec. 
    df_train = pd.concat([df_nspec_train, df_spec_train], ignore_index=True)
    df_test = pd.concat([df_nspec_test, df_spec_test], ignore_index=True)


    #PREPROCESSING TEXT for values in sentence columns in test and train dataframes. 
    df_train.Sentences = [preprocess_text(i) for i in df_train.Sentences]
    df_test.Sentences = [preprocess_text(i) for i in df_test.Sentences]


    #Assign respective  values for X(sentences) and Y(value of speculative or not) for train and test data for classifier.
    X_train = df_train.Sentences
    y_train = df_train.Speculative
    X_test = df_test.Sentences
    y_test = df_test.Speculative


    #Use vectorizer to represent sentences as lists of vectors representing words and use for sentences in both train/test data.
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)



    #TESTING DIFFERENT ML METHODS (Print classification score and confusion matrix to see True/False Neg/Pos results)

    #Support vector machine
    print("Support vector machine algorithm:")
    clf = SVC()
    clf.fit(X_train, y_train)
    print(confusion_matrix(y_test,clf.predict(X_test)))
    print(metrics.classification_report(y_test, clf.predict(X_test)))


    #Logistic regression
    print("Logistic regression algorithm:")
    logreg = LogisticRegression(max_iter=200)
    logreg.fit(X_train, y_train)
    print(confusion_matrix(y_test,logreg.predict(X_test)))
    print(metrics.classification_report(y_test, logreg.predict(X_test)))


    #Naive Bayes
    print("Naive Bayes algorithm:")
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    print(confusion_matrix(y_test,nb.predict(X_test)))
    print(metrics.classification_report(y_test, nb.predict(X_test)))


    #Decision Tree
    print("Decision Tree algorithm:")
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    print(confusion_matrix(y_test,dt.predict(X_test)))
    print(metrics.classification_report(y_test, dt.predict(X_test)))


    #Random forest
    print("Random forest algorithm:")
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    print(confusion_matrix(y_test,rf.predict(X_test)))
    print(metrics.classification_report(y_test, rf.predict(X_test)))


if __name__ == "__main__":
    main()


