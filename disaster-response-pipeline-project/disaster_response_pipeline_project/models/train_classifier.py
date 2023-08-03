# Import libraries
import pickle
import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine

import re
from sklearn.pipeline import Pipeline

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


nltk.download(['punkt', 'wordnet', 'stopwords'])

def load_data(database_filepath):
    # Load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_messages', engine)
    # Create X, Y variables
    X = df['message'] # Message Column
    Y = df.iloc[:, 4:] # Classification Label
    return X, Y


def tokenize(text):
    '''
    Tokenizes words:
    Splits them into words then groups together similar phrases.
    '''
    # Normalize words to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize the text
    token_word = word_tokenize(text)
    
    # Use this to group together similar words
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in token_word:
        clean_token = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_token)
    
    return clean_tokens


def build_model():
    '''
    Builds model, then tunes it using grid search
    '''
    # Create Pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier())) #Random Forest Used
    ])
    
    # Create grid search parameters
    parameters = {
        'clf__estimator__n_estimators': [10, 20]
    }

    # Grid search
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test):
    """
    Evaluates the model. Prints precision, recall and f1 score.
    """
    y_pred = model.predict(X_test)
    i=0
    for col in Y_test:
        print('Feature {}: {}'.format(i + 1, col))
        print(classification_report(Y_test[col], y_pred[:, i]))
        i = i + 1

    # Accuracy Test
    accuracy = (y_pred == Y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))


def save_model(model, model_filepath):
    """
    This function exports the model as a pickle file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()