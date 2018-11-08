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
- pic:
    - category.jpg: Figure used in README.md
- ETL_Pipeline_Preparation.ipynb: Prepare file to build the data extraction-tranform-load pipeline
- ML_Pipeline_Preparation.ipynb: Prepare file to grid search best parameters in different machine learning models


# 3. Defect of the Raw Dataset
In the main page of the web application, a graph of category count of the raw dataset has been provided.

![category figure](https://www.jianguoyun.com/p/DYNljbMQ0JeABhif0oEB)

From the figure, it turns out that the proportion of each category is imbalanced. For some categories like "Storm", "food", "water", messages text resource is plenty. While for some other categories, such as "missing people", "fire", the number of related message text is rare. Especially there is no message belongs to "child_alone" category.

So the fact will result inaccuracy of the model to judge message which should be classified to categories with minimum training resource.

# 4. How to Interact with the Project

Any improvement suggestion is appreciated especially for the better machine learning model.

The pushed code should follow PEP-8 style.

# 5. Licensing

BSD 3-clause

# 6. Authors
[Max Qiu](https://github.com/ft9738962)
