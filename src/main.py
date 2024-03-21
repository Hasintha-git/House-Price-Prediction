# -*- coding: utf-8 -*-
"""House Price Prediction - 2024

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/house-price-prediction-2024-799c9245-b663-4a06-b772-ce479dd01f04.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20240316/auto/storage/goog4_request%26X-Goog-Date%3D20240316T062651Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3Da1437370b84075e49ae8f888a0631294dd4caacc9e9e43b60c4703189f900bbc63823836aff8528005588b552293a80e43f77845aa8d7f42b23560a36e2a99d44e13c54edbee2782a62c31f7a4f9f3baee4f89e8d4d3a1563d0849fed96ab89dcf8332698f99b3b7aaa440c066207ccca08bbedcadc40c0c7886bed5344a1e1ddceb3e673a4065ebade92c334dd321c571ac9922f5d3fe3bd82e58f59a956d7b54cc186b9a73fce939df85f235be6f0838bca8bc3a55617fbd6fd56d09f953972413c5cf6a7685ff232cb2d285ea5fe98a490fd640c8ce90bbc3a0929c245fbbe36fc613792db530ea8e3235a701b5f3a8be70c14f962591f3060c964da5ebe1
"""

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES
# TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'house-prices-advanced-regression-techniques:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-competitions-data%2Fkaggle-v2%2F5407%2F868283%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240316%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240316T062651Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D4b4b9ace9fa6727d46f82b66f09befc5d04f4e0fc8dfdcce89f276fd07597ffd52acac3b97a64c2cddf4514085ee320ac6f17a16b39ab1f2e0241a8b43f1500c7e9d95f2e5ae32453e88b2a1e6d67102882237f1097532bbe1b9da9690645278c79a5ba353798483fc41cee99ef48177dccde70015aa0812f732042dd41fb779111111a8e803463658559c692d53b8c406dd00dc7121b9cb8346ef09c64c04e7af4f965e4abfcdf01cd2ecf6ed52660f7395c40c5535a6fd18cc987b643c1f1b59730b1923c7628b154c92715a0300a0749928e2fab6842886f3347e777a45045550aef494819a79282900c61127a9d8823dbd136a79fbd2150912239c4fbad5'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

"""# **Step 1: Load and Explore the Data**"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the datasets
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# Shape of the dataset
print("Shape of the dataset :", train_df.shape)

# Check for the missing values
print("\nMissing Values :")
print(train_df.isnull().sum())

train_df.head()

test_df.head()

columns_to_keep = ['Id','MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street','SaleCondition','YearBuilt','BedroomAbvGr','HouseStyle', 'SalePrice']
columns_to_keep2 = ['Id','MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street','SaleCondition','YearBuilt','BedroomAbvGr','HouseStyle']

# Select only the desired columns
test_df = test_df[columns_to_keep2]
train_df = train_df[columns_to_keep]

"""# Step 2: Visualizations****

**1. Distribution of Sale Prices**
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Distribution of Sale Prices
plt.figure(figsize=(10, 6))
sns.histplot(train_df['SalePrice'], kde=True, bins=30, color='blue')
plt.title('Distribution of Sale Prices')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
# Save the plot as an image file (e.g., PNG)
plt.savefig('sale_price_distribution.png', bbox_inches='tight')

# Show the plot
plt.show()

"""**2. Correlation Heatmap**"""

# Correlation matrix of numerical features
corr_matrix = train_df.select_dtypes(include=['int64', 'float64']).corr()

# Heatmap of correlations
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

# Save the plot as an image file (e.g., PNG)
plt.savefig('Heatmap.png', bbox_inches='tight')

"""# **Step 3: Preprocess the Data**"""

# Separate target variable and predictors
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = np.log(train_df['SalePrice'])  # Transform target variable with logarithm
X_test = test_df.drop(['Id'], axis=1)

# Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# null value fill
median_lot_frontage = train_df['LotFrontage'].median()
train_df['LotFrontage'].fillna(median_lot_frontage, inplace=True)


# Check for the after missing values fill
print("\nMissing Values :")
print(train_df.isnull().sum())

"""# Univariate Analysis"""

train_df

# Historical for all numerical columns
fig, ax = plt.subplots(figsize=(20, 15))
train_df.hist(bins=50, ax=ax)

# Save the plot as an image file (e.g., PNG)
fig.savefig('histogram.png', bbox_inches='tight')

# Optionally, close the plot to free up memory
plt.close(fig)

correlation_matrix = train_df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Plot the heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(train_df.corr(), annot=True)
plt.title('Correlation Heatmap')

# Show the plot
plt.show()

sns.boxplot(x=train_df['SalePrice'])
plt.title('Boxplot for Sale Price')
plt.show()

"""# **Step 4: Define the Model and Bundle Preprocessing and Modeling Code in a Pipeline**"""

model = RandomForestRegressor(n_estimators=100, random_state=0)

X_train

y_train

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])

# Train the model
my_pipeline.fit(X_train, y_train)

"""# **Step 5: Predict and Prepare Submission**"""

X_test

# first_row_train = X_train.iloc[[0]]
predictions_df_train = pd.DataFrame({
    'MSSubClass': [60],
    'MSZoning': ['RH'],
    'LotFrontage': [65.0],
    'LotArea': [8450],
    'Street': ['Pave'],
    'SaleCondition': ['Normal'],
    'YearBuilt': ['2003'],
    'BedroomAbvGr': [3],
    'HouseStyle': ['2Story'],
})

predictions_log_scale = my_pipeline.predict(predictions_df_train)


decoded_prices = np.exp(predictions_log_scale)
print(decoded_prices)

import os
import pickle

# Assuming your model is named vr
# Save the model to a file
with open('house_price_prediction_model.pkl', 'wb') as f:
    pickle.dump(my_pipeline, f)

# Get the size of the saved file
file_size = os.path.getsize('house_price_prediction_model.pkl')

print("Model saved successfully as 'house_price_prediction_model.pkl'")
file_size_MB = 20641183 / (1024 * 1024)
print("File size:", file_size_MB, "MB")

predictions_log_scale2 = my_pipeline.predict(X_test)


decoded_prices = np.exp(predictions_log_scale2)
print(decoded_prices)


print('----------')
print(test_df['Id'].shape)
print(decoded_prices.shape)
# Create a DataFrame for the predictions
predictions_df = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': decoded_prices
})

predictions_df.head()

predictions_df.to_csv("submission.csv", index=False)

"""# **Model Test With Server**"""

# Assuming X_test is your new data for prediction
X_test  = pd.DataFrame({
    'MSSubClass': [60],
    'MSZoning': ['RH'],
    'LotFrontage': [65.0],
    'LotArea': [8450],
    'Street': ['Pave'],
    'SaleCondition': ['Normal'],
    'YearBuilt': ['2003'],
    'BedroomAbvGr': [3],
    'HouseStyle': ['2Story'],
})


print("ready..")

# Load the model from the .pkl file
try:
    with open('house_price_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")

    # Make predictions
    predictions = model.predict(X_test)
    print("Predictions:", predictions)

    decoded_prices = np.exp(predictions)
    print(decoded_prices)


except FileNotFoundError:
    print("Error: Model file not found.")
except EOFError:
    print("Error: Model file is empty or corrupted.")
except Exception as e:
    print("Error:", e)