import sys
import json
import time
import pandas as pd
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.corpus import stopwords
nltk.download('stopwords')

from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from joblib import parallel_backend
import joblib

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np




def load_data(database_filepath):
    '''Load data from sqlite table named message_response to a tuple
    of features (X), labels (Y) and name of labels (catogories).

    Input: Path of database
    Return: A tuple of X,Y, and categories'''

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("SELECT * FROM message_response", con=engine)
    # print(df.info())

    # reduce memory, converting int64 to int8
    for column in df.columns[2:]:
        df[column] = df[column].astype('int8')

    # convert message to str, 
    df['message'] = df['message'].astype(str)

    # spit features, labels
    X = df['message'].values
    Y = df.drop(['message', 'genre'], axis=1).copy().values

    # return names of the columns except the first two
    categories = df.columns[2:]

    return (X, Y, categories)


def tokenize(text):
    '''Process text to lower case, remove stopwords, and lemmatize.

    Input: A line of text
    Return: a list of words (tokens)
    '''

    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)   
    tokens = word_tokenize(text)
    
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model_simple():
    '''a simple model without GridSearch'''

    pipeline_knn = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier(n_jobs=-1)))
    ])

    return pipeline_knn


def build_model():
    '''Build model using GridSearchCV and Pipeline.

    Return: a model from Pipeline object'''


    # setup pipeline
    pipeline_knn = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    # uncomment some of parameter for optional training
    parameters = {
        # 'vect__stop_words': (None, 'english'),
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.25, 0.5, 0.75),
        'vect__max_features': (500, 1000, 2000),
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__smooth_idf': (True, False),
        # 'tfidf__sublinear_tf': (True, False),
        'clf__estimator__n_neighbors': [3, 5, 10],
        # 'clf__estimator__weights': ['uniform', 'distance'],
        # 'clf__estimator__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        # 'clf__estimator__leaf_size': [10,30,50, 100],
        # 'clf__estimator__p': [1,2,3]
    }

    cv = GridSearchCV(pipeline_knn, param_grid=parameters, cv=5, verbose=True)
    
    return cv



def evaluate_model(model, X_test, Y_test, category_names, 
        verbose=False, score_report='evaluation_score.txt'):
    '''Evaluate model by testset data using `classification_report`.

    Input: trained model, testset (X_test, Y_test), and category name.
    If verbose is True, details of reports is incuded, otherwise, summary values 
    of each category

    Return: a dictionary contains precision, recall and f1-score with key as category name
    '''

    scores = dict()
    Y_predict = model.predict(X_test)

    for i in range(len(category_names)):
        report = classification_report(Y_predict[:, i], Y_test[:, i], 
                                           output_dict=True)
        if verbose:
            scores.update({category_names[i]: report})
        else:
            scores.update({category_names[i]: report['weighted avg']})
    with open(score_report, 'w+') as f:
        f.write(json.dumps(scores))
    return scores

def visualize_report(score_report, img_path='test_score.png'):
    '''produce a graph visualizing the report scores'''


    df_scores = pd.DataFrame.from_dict(data=score_report, orient='index')
    fig, ax = plt.subplots(figsize=(10,6))
    width = 0.2
    score_types = ['precision', 'recall', 'f1-score']
    x = np.arange(0, len(df_scores))
    for i, label in enumerate(score_types):
        ax.bar(x+i*width, df_scores[label], width=width, label=label)
    ax.set_xlim(0.5, len(df_scores))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xticklabels(df_scores.index, rotation=90)
    ax.set_title('Scores on testing data')
    fig.legend(ncol=3, loc='lower center')
    fig.tight_layout();
    fig.savefig(img_path)
    return None


def save_model(model, model_filepath):
    '''save trained model to a pickle file'''

    joblib.dump(model, model_filepath)
    return None


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                        test_size=0.3, random_state=1)
        
        print('Building model...')
        # model with GridSearCV
        model = build_model()
        # model with only Pipeline
        # model = build_model_simple()
        
        print('Training model...')
        start = time.time()

        # with parallel_backend('threading', n_jobs=-1):
        model.fit(X_train, Y_train)     

        last_for = time.time() - start
        print(f'Total training time: {last_for:.1f} seconds')
        
        print('Evaluating model...')
        start = time.time()
        scores = evaluate_model(model, X_test, Y_test, category_names)
        print(f'Total evaluation time: {last_for:.1f} seconds')

        print('Producing a graph for scores')
        visualize_report(scores)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    print('If the training takes too long, consider running `simple model` \n'
        'to test out the code.')

    main()