import numpy as np
from sklearnex import patch_sklearn

patch_sklearn()
import pandas as pd
import os
import pickle
import sys
from re import sub
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


dir = os.path.abspath("sentiment labelled sentences")
df = pd.DataFrame(columns=["data", "target"])

for file in os.listdir(dir):
    if file != "readme.txt":
        df = pd.concat(
            [
                df,
                pd.read_csv(
                    os.path.join(dir, file),
                    sep="\t",
                    header=None,
                    encoding='ISO-8859-1',
                    names=df.columns
                )
            ], axis=0, ignore_index=True
        )
df.dropna(inplace=True)
df['target'] = df['target'].astype('int')
train, test = train_test_split(df, test_size=0.2, shuffle=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_data(data_set):
    cleaned_data = []
    for data in data_set:
        data = sub(r'http\S+', '', data)
        data = sub(r'@[A-Za-z0-9_]+', '', data)
        data = sub(r'#', '', data)
        tokens = word_tokenize(data)
        tokens = [token for token in tokens if token.lower() not in stop_words]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Joining the tokens back into a string
        cleaned_data.append(' '.join(tokens))

    return cleaned_data


# train = fetch_20newsgroups(subset='train', shuffle=True)
# test = fetch_20newsgroups(subset='test', shuffle=True)

models = dict()

clean_function = FunctionTransformer(clean_data)
text_clf = Pipeline([('clean', clean_function),
                     ('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])
text_clf = text_clf.fit(train['data'], train['target'])

predicted = text_clf.predict(test['data'])
test_report = classification_report(test['target'], predicted)
print("MultinomialNB Report: \n", test_report)
models["nb"] = (accuracy_score(test['target'], predicted), text_clf)

text_clf = Pipeline([('clean', clean_function),
                     ('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42)),
                     ])
_ = text_clf.fit(train['data'], train['target'])
predicted = text_clf.predict(test['data'])
test_report = classification_report(test['target'], predicted)
print("Multi-layer SGD_Classifier Report: \n", test_report)
models["SGD"] = (accuracy_score(test['target'], predicted), text_clf)

clfs = [{
    'clf': MultinomialNB(),
    'name': 'grid_nb',
    'param_grid': {'alpha': [0.001, 0.01, 0.1]}
}, {
    'clf': RandomForestClassifier(),
    'name': 'grid_rf',
    'param_grid': {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30]}
}, {
    'clf': MLPClassifier(),
    'name': 'grid_mlp',
    'param_grid': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001, 0.01]}
}]

for clf in clfs:
    grid = Pipeline([('clean', clean_function),
                     ('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf-grid', GridSearchCV(clf['clf'], param_grid=clf['param_grid'], cv=5)),
                     ])
    grid.fit(train['data'], train['target'])
    predicted = grid.predict(test['data'])
    test_report = classification_report(test['target'], predicted)
    print("Multi-layer perceptron Classifier validation report: \n", test_report)
    models[clf["name"]] = (accuracy_score(test['target'], predicted), grid)

best_acc = -sys.maxsize - 1

for model in models:
    with open(f"{model}.joblib", "wb") as fp:
        pickle.dump(models[model][1], fp)
        if models[model][0] > best_acc:
            best_acc = models[model][0]
            with open("best.joblib", "wb") as best:
                pickle.dump(models[model][1], fp)

