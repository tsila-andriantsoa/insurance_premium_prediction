import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# Load datasets
# train = pd.read_csv('../data/raw/train.csv', )
# test = pd.read_csv('../data/raw/test.csv', )

train = pd.read_parquet('../data/raw/train.parquet')
test = pd.read_parquet('../data/raw/test.parquet')

print(train.shape, test.shape)

# rename columns
train.columns = [str.lower(col).replace(' ','_') for col in train.columns]

# remove null value
train.dropna(subset='premium_amount',inplace=True)
# remove unused columns
train.drop(columns=['id'], inplace = True)
train.drop(columns = ['previous_claims', 'occupation'], inplace = True)

numerical_features = train.select_dtypes(include = 'float').columns.tolist()
numerical_features = [n for n in numerical_features if n != 'premium_amount']
categorical_features = train.select_dtypes(include = 'object').columns.tolist()
categorical_features = [n for n in categorical_features if n != 'policy_start_date']

numerical_features = [n for n in numerical_features if n != 'previous_claims']
categorical_features = [n for n in categorical_features if n != 'occupation']

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

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

from sklearn.feature_selection import mutual_info_classif

X_train = train.drop(columns = ['premium_amount'])
Y_train = train['premium_amount']

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

df_training.to_parquet('../data/prepared/df_training.parquet', index=False)
df_validation.to_parquet('../data/prepared/df_validation.parquet', index=False)

print(df_training.shape, df_validation.shape)