# disaster_response
Udacity Data Science Nanodegree: Disaster Response project

## Disaster response pipeline project
This project aims to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.

### Installations:

The required libraries for running the code are part of the Anaconda distribution for Python 3.6.2. Following libraries were used:

* Pandas
* Plotly
* JSON
* NLTK
* Flask
* sklearn
* sqlalchemy
* re
* pickle

### Project Motivation

This project combines aspects of data engineering and web app builing skills for analysis and representation of disaster response data. This data consists of messages provided by FigureEight that were collected during natural disasters from social media or from disaster organizations. are analyzed. The messages and their respective labelled categories serve as a basis for classification of new messages into one or more categories depending upon the word level features. The dataset is processed using an ETL pipeline followed by builing a machine learning pipeline to train and optimize a classifier. The results from previous steps is compiled and presented as a web app which can classify and assign categories to a new message.

### File Descriptions

There are three components for this project:

1. ETL Pipeline: The Python script, process_data.py, performs data cleaning that includes:
  * Loading the messages and categories datasets
  * Merging the two datasets
  * Cleaning the data
  * Store it in a SQLite database

2. ML Pipeline: The Python script, train_classifier.py, applys a machine learning pipeline that:
  * Loads data from the SQLite database
  * Splits the dataset into training and test sets
  * Builds a text processing and machine learning pipeline
  * Trains and tunes a model using GridSearchCV
  * Outputs results on the test set
  *  Exports the final model as a pickle file

3. Flask Web App: The files in the /app directory is for building the flask web app where an emergency response input gets classified into one or more of several categories. The web app also displays visualizations of the data (genres and categories).


### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Licensing and Acknowledgements:

The data for this project and related information was provided by FigureEight in collaboration with Udacity. I would like to thank the mentors at Udacity for coaching necessary background skills in Data Analysis.
