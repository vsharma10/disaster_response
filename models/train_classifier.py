import sys
import re
import nltk
import pickle
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import make_scorer, accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

def load_data(database_filepath):
    """
    INPUT
    database_filepath - Path to disaster response database file
    OUTPUT
    X - Feature values
    y - Labels
    category_names - Categories names for labels
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM ResponseData",engine)
    
    X = df['message'].values
    y = df.drop(['id','message','original','genre'], axis=1)   
    category_names = y.columns
    
    return X, y, category_names
                       
                           

def tokenize(text):
    """
    INPUT
    text - Input response text
    OUTPUT
    tokens - list of tokens from processed text
    """
    tokens = nltk.word_tokenize(txt)
    tokens = [w for w in tokens if bool(re.search(r"[^a-zA-Z0-9]", w)) != True]
    tokens = [WordNetLemmatizer().lemmatize(w, pos='v') for w in tokens if stopwords.words("english")]
    tokens = [PorterStemmer().stem(w) for w in tokens]
    
    return tokens


def build_model():
    """
    Builds classification model:
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-1)
    return cv



def precision(y_test, y_pred):
	"""
	INPUT
	y_test - Labels from test set for a given category
	y_pred - predicted labels for a given category
	OUTPUT
	Precision value for a given category
	"""
    count = 0
    for i in range(len(y_test)):
        if y_test[i] == 1 & y_pred[i] == 1:
            count += 1
    
    if list(y_pred).count(1) == 0:
        return 0.0
    else:
        return round(count/list(y_pred).count(1), 4)

def recall(y_test, y_pred):
	"""
	INPUT
	y_test - Labels from test set for a given category
	y_pred - predicted labels for a given category
	OUTPUT
	Recall value for a given category
	"""
    count = 0
    for i in range(len(y_test)):
        if y_test[i] == 1 & y_pred[i] == 1:
            count += 1
    
    if list(y_test).count(1) == 0:
        return 0.0
    else:
        return round(count/list(y_test).count(1), 4)


def f1(y_test, y_pred):
	"""
	INPUT
	y_test - Labels from test set for a given category
	y_pred - predicted labels for a given category
	OUTPUT
	F1 Score for a given category
	"""
    p = precision(y_test, y_pred)
    r = recall(y_test, y_pred)
    if p + r == 0:
        return 0.0
    else:
        return round((2 * (p * r) / (p + r)), 4)



def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model against test dataset
    INPUT
    model - Optimized model
    X_test - Test dataset features
    Y_test - Test dataset labels
    category_names - Category names for labels
    
    OUTPUT
    Accuracy scores for each category printed out
    """
    y_pred = model.predict(X_test)
    print("Accuracy scores for each category:\n")
    for  idx, cat in enumerate(Y_test.columns.values):
        accuracy  = round(accuracy_score(Y_test.values[:,idx], y_pred[:, idx]), 2)
        pres      = precision(Y_test.values[:,idx], y_pred[:, idx])
    	reca      = recall(Y_test.values[:,idx], y_pred[:, idx])
    	f1_scr    = f1(Y_test.values[:,idx], y_pred[:, idx])
    	print("{}:".format(cat))
        print("\t-- Accuracy = {}; Precision = {}; Recall = {}; F1 Score = {}\n".format(accuracy, pres, reca, f1_scr))



def save_model(model, model_filepath):
    """
    Saves the model as pickle file (*.pkl)
    INPUT
    model - Optimized trained model
    model_filepath - Path to save the model
    OUTPUT
    Saves the model as pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))



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