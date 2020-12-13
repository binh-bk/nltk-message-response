import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''Load data in .csv format to DataFrame.

    Input:
    1. Filepath of csv messages
    2. File of csv categories

    Return: a combined DataFrame joined on id of each CSV file
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_on='id', right_on='id')
    return df  


def clean_data(df):
    '''Clean DataFrame and expand categories in text to numeric.

    Input: A DataFrame from a load_data function
    Return: cleand DataFrame for machine learning 
    '''

    # Drop duplicated and unrelevant to machine learning
    df.drop_duplicates(inplace=True)
    df.drop(['original', 'id'], axis=1, inplace=True)

    # drop empty rows
    df.dropna(how='all', axis=0, inplace=True)
    
    

    # Split categories by semicolons; one item per column
    df_cat = df['categories'].str.split(';', expand=True)

    # rename columns with label of each category
    columns_names = df_cat.iloc[0].values
    columns_names = [each.split('-')[0] for each in columns_names]
    df_cat.columns = columns_names

    # iterate over each column, take the numeric value from each cell
    for column in df_cat.columns:
        df_cat[column] = df_cat[column].apply(lambda row: int(row.split('-')[1]))

    # Drop categories column and merge new values from df_cat
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, df_cat], axis=1)

    # remove duplicated row one more time
    df.drop_duplicates(inplace=True)
    # print(f'Duplicated : {df.duplicated().sum()}')
    # print(f'Shape: {df.shape}')
    return df


def save_data(df, database_filename):
    '''save clean DataFrame to an SQL database'''

    db_filepath = f'sqlite:///{database_filename}'
    engine = create_engine(db_filepath)
    df.to_sql('message_response', engine, index=False, if_exists='replace')
    return None  


def main():
    '''Main function, combine load_data, clean_data and save_data
    Inputs: filepaths of messages.csv, categories.csv, path to sqlite db

    Check the dubug message for example to use this script in terminal
    '''

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