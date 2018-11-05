import sys
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

def load_data(database_filepath):
    '''Load data from database and convert to two data: X(message) and Y(label)
    
    Input:
    database_filepath: filepath of the database
    
    Output:
    X: messages
    Y: labels
    category_names: names of label
    '''
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_response', engine)
    X = df.loc[:,'message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns.tolist()
    
    return X, Y, category_names

def tokenize(text):
    '''Tokenize a text message to a list of lemmatized words
    
    Input:
    text: The text meassage to be tokenized
    
    Output:
    clean_words: List of tokenized and lemmatized words
    '''   
    text = word_tokenize(re.sub(r'[^a-zA-Z0-9]',' ',text).lower().strip())
    lemmatizer = WordNetLemmatizer()
    
    clean_words = []
    for word in text:
        clean_word = lemmatizer.lemmatize(word, pos='v')
        clean_words.append(clean_word)

    return clean_words

class TextLengthExtractor(BaseEstimator, TransformerMixin):
    '''
    Add the length of the text message as a feature to dataset
    
    The assumption is people who is in urgent disaster condition will prefer to use less words to express
    '''
    
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X).applymap(len)

def build_model():
    '''Build the machine learning model based on training data
    
    Input:
    None
    
    Output:
    None
    '''
    
    model = Pipeline([
        ('tfidfvect', TfidfVectorizer(tokenizer=tokenize, stop_words='english', ngram_range=(1,2), max_df=0.75, max_features=2000)),
        ('moc', MultiOutputClassifier(RandomForestClassifier(n_estimators=15, max_depth=None, min_samples_split=5)))
    ])
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''Print out result of classification report for each label
    
    Input:
    model: trained model
    X_test: messages for testing
    Y_test: labels for validating
    category_names: list of names of label
    
    Output:
    None
    '''
    y_pred = pd.DataFrame(model.predict(X_test), columns = category_names)
    
    tot_acc = 0
    tot_f1 = 0
    for i in category_names:
        print(i, 'accuracy: {:.2f}'.format(accuracy_score(Y_test.loc[:, i], y_pred.loc[:, i])),'\n',
          classification_report(Y_test.loc[:, i], y_pred.loc[:, i]),'\n')
        tot_acc += accuracy_score(Y_test.loc[:, i], y_pred.loc[:, i])
        tot_f1 += precision_recall_fscore_support(Y_test.loc[:, i], y_pred.loc[:, i], average = 'weighted')[2]
    print('The average accuracy score is', round(tot_acc/len(category_names),4), ', average f1-score is', round(tot_f1/len(category_names),4))

def save_model(model, model_filepath):
    '''Save 
    
    Input:
    model: trained model
    X_test: messages for testing
    Y_test: labels for validating
    category_names: list of names of label
    
    Output:
    None
    '''
    
    joblib.dump(model, f'{model_filepath}')

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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