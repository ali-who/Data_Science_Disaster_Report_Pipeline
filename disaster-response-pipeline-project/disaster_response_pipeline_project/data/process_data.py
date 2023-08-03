# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sqlite3
import sqlalchemy
from sqlalchemy import create_engine

import sys


def load_data(messages_filepath, categories_filepath):
    '''
    This function loads and merges datasets from two different paths:
        messages_filepath
        categories_filepath
    Returns combined df
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Outer Join using ID column
    df = messages.merge(categories, how ='outer', on =['id'])

    return df


def clean_data(df):
    '''
    This function cleans the dataframe
    '''
    categories = df['categories'].str.split(';', expand=True)
    # create a dataframe of the 36 individual category columns
    # select the first row of the categories dataframe
    row = categories.head(1)
    # Extract a list of new column names for categories.
    # Apply a lambda function that takes everything up to the second to last character of each string with slicing
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0,:]
    # Rename the columns of "categories"
    categories.columns = category_colnames

    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # Convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # Replace 2s with 1s in related column
    categories['related'] = categories['related'].replace(to_replace=2, value=1)
    
    # Drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # Concatenate categories with the original dataframe
    df = pd.concat([df, categories], axis=1)
    # Drop potential duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filepath):
    '''Save date ti a SQLlite DB '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()