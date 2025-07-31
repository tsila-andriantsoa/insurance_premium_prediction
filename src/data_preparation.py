import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

RAW_DATA_PATH = "data/raw/"
PREPARED_DATA_PATH = "data/prepared/"
os.makedirs(PREPARED_DATA_PATH, exist_ok=True)

def load_raw_data():
    train = pd.read_parquet(RAW_DATA_PATH + 'train.parquet')
    test = pd.read_parquet(RAW_DATA_PATH + 'test.parquet')
    print("raw data loaded")
    return train, test

def preprocess_data(df):

    # rename columns    
    df.columns = [str.lower(col).replace(' ','_') for col in df.columns]

    # remove null value
    df.dropna(subset='premium_amount',inplace=True)

    # remove unused columns
    df.drop(columns=['id'], inplace = True)
    df.drop(columns = ['previous_claims', 'occupation'], inplace = True)

    print("train data preprocessed")
    return df

def prepared_train_validation_data(df):

    numerical_features = df.select_dtypes(include = 'float').columns.tolist()
    numerical_features = [n for n in numerical_features if n != 'premium_amount' if n!= 'previous_claims']
    categorical_features = df.select_dtypes(include = 'object').columns.tolist()
    categorical_features = [n for n in categorical_features if n != 'policy_start_date' if n!= 'occupation']

    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  
        ('onehot', OneHotEncoder(drop = 'first', handle_unknown='ignore'))
    ])

    # Combine preprocessors in a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    pipeline = Pipeline([
        ('preprocessor',  preprocessor)
    ])

    X_train = df.drop(columns = ['premium_amount'])
    Y_train = df['premium_amount']

    pipeline.fit(X_train, Y_train)

    X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
    numerical_transformed_columns = pipeline.named_steps['preprocessor'].transformers_[0][2]
    categorical_transformed_columns = pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out().tolist()
    all_columns = [numerical_transformed_columns + categorical_transformed_columns]
    all_columns = all_columns[0]

    all_columns = [str.lower(col).replace('\'','').replace(' ','_') for col in all_columns]

    df_X_full_train_transformed = pd.DataFrame(
        X_train_transformed,
        columns = all_columns
    )

    # select best columns
    top_features = ['credit_score', 'customer_feedback_good', 'annual_income', 'health_score']

    X_train_part, X_validation_part, y_train_part, y_validation_part = train_test_split(df_X_full_train_transformed[top_features], Y_train, test_size=0.2, random_state=42)

    # save data
    df_training = X_train_part.copy()
    df_training['premium_amount'] = y_train_part.values  # Or just y_training_part if the index aligns

    df_validation = X_validation_part.copy()
    df_validation['premium_amount'] = y_validation_part.values  # Or just y_training_part if the index aligns

    df_training.to_parquet(PREPARED_DATA_PATH + 'df_training.parquet', index=False)
    df_validation.to_parquet(PREPARED_DATA_PATH + 'df_validation.parquet', index=False)
    print('train data and validation data saved into path: ', PREPARED_DATA_PATH)


if __name__ == '__main__':
    df_train, df_test = load_raw_data()
    df_train_preprocessed = preprocess_data(df_train)
    prepared_train_validation_data(df_train_preprocessed) 
