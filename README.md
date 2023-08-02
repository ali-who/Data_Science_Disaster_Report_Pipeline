# Data_Science_Disaster_Report_Pipeline
## Project Overview
In this project, I've learned and built on my data engineering skills to expand my opportunities and potential as a data scientist. In this project, I will apply these skills to analyze disaster data from Appen (formally Figure 8) to build a model for an API that classifies disaster messages.

In the Project Workspace, you will find a data set containing real messages that were sent during disaster events. I will be creating a machine learning pipeline to categorize these events so that I can send the messages to an appropriate disaster relief agency.

My project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. This project will show off my software skills, including my ability to create basic data pipelines and write clean, organized code!

Below are a few screenshots of the web app.
![image](https://github.com/ali-who/Data_Science_Disaster_Report_Pipeline/assets/126905703/fb490304-a3ef-42cb-972e-6d5a89ffbccb)
![image](https://github.com/ali-who/Data_Science_Disaster_Report_Pipeline/assets/126905703/c49d4c0a-0876-4c7c-99d0-7ea7896fd82b)
![image](https://github.com/ali-who/Data_Science_Disaster_Report_Pipeline/assets/126905703/834d673e-164b-4a20-8ccb-357bbd1a0706)
![image](https://github.com/ali-who/Data_Science_Disaster_Report_Pipeline/assets/126905703/5e9c6a3b-3f8a-4ba8-b125-9beeb2837077)

## Project Components
There are three components for this project.

### 1. ETL Pipeline
In a Python script, process_data.py, a data cleaning pipeline that:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database
### 2. ML Pipeline
In a Python script, train_classifier.py, a machine learning pipeline that:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file
### 3. Flask Web App
This was provided to me by Udacity.
