# 1. Purpose
This project is a web app using machine learning which is for predicting classification for disaster type

The machine learning model is based on data from [Figure Eight](www.figure-eight.com)

The final app will be presented with on a web page. Users can type any sentence to test the model and see how it category sentence to existing labels

# 2. File Description

- data:
    - process_data.py: pipeline for data wrangling and stores clean data in SQLite database
    - DisasterResponse.db: database stored clean data from process_data.py
    - disaster_messages.csv: messages dataset
    - disaster_categories.csv: category for message text
- models:
    - train_classifier.py: machine learning module to create appropriate model
- app:
    - templates:
        - go.html: interact with message and show results from machine learning model
        - master.html: basic html for web app
    - run.py: include flask for backend and plotly with data to create figure
- ETL_Pipeline_Preparation.ipynb: Prepare file to build the data extraction-tranform-load pipeline
- ML_Pipeline_Preparation.ipynb: Prepare file to grid search best parameters in different machine learning models

# 3. How to Interact with the Project

Any improvement suggestion is appreciated especially for the better machine learning model.

The pushed code should follow PEP-8 style.


# 4. Licensing

BSD 3-clause

# 5. Authors
[Max Qiu](https://github.com/ft9738962)
