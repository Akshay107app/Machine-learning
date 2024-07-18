# Machine-learning
Machine Learning (ML) is a subfield of artificial intelligence (AI) that focuses on building systems capable of learning from data, identifying patterns, and making decisions with minimal human intervention. These systems improve their performance on a given task over time as they are exposed to more data.

Key Concepts
Supervised Learning: Involves training a model on labeled data, where the input data and the corresponding output labels are known. Common algorithms include linear regression, decision trees, and support vector machines.

Unsupervised Learning: The model learns patterns from unlabeled data. It is often used for clustering and association tasks. Examples include k-means clustering and principal component analysis (PCA).

Reinforcement Learning: The model learns by interacting with an environment, receiving rewards or penalties based on its actions. This approach is commonly used in robotics, game playing, and autonomous vehicles.

Deep Learning: A subset of ML that uses neural networks with many layers (deep neural networks) to model complex patterns in large datasets. It is particularly effective in tasks like image and speech recognition.

Applications
Machine learning is transforming various industries through its applications in:

Healthcare: Predictive analytics for disease diagnosis and personalized treatment plans.
Finance: Fraud detection, algorithmic trading, and credit scoring.
Marketing: Customer segmentation, recommendation systems, and sentiment analysis.
Transportation: Autonomous driving, traffic prediction, and route optimization.

### Project1
# Fraud Detection in E-commerce Transactions

## Problem Statement

E-commerce websites often transact huge amounts of money. Whenever a huge amount of money is moved, there is a high risk of users performing fraudulent activities, such as using stolen credit cards or laundering money.

XYZ is an e-commerce site that sells wholesale electronics. You have been contracted to build a model that predicts whether a given transaction is fraudulent or not. You only have information about each user’s first transaction on XYZ's website. If you fail to identify a fraudulent transaction, XYZ loses money equivalent to the price of the fraudulently purchased product. If you incorrectly flag a real transaction as fraudulent, it inconveniences XYZ customers whose valid transactions are flagged—a cost your client values at $8.

## Data Description

Information about the first transaction of each user is provided. The dataset contains the following columns:

- `user_id`: Id of the user. Unique by user.
- `signup_time`: The time when the user created their account (GMT time).
- `purchase_time`: The time when the user bought the item (GMT time).
- `purchase_value`: The cost of the item purchased (USD).
- `device_id`: The device id. Assumed to be unique by device. Two transactions with the same device ID indicate the same physical device was used to buy.
- `source`: User marketing channel: ads, SEO, Direct (i.e., came to the site by directly typing the site address on the browser).
- `browser`: The browser used by the user.
- `sex`: Male/Female.
- `age`: User age.
- `ip_address`: User numeric IP address.
- `class`: The target variable, indicating whether the activity was fraudulent (1) or not (0).
- `country`: Country of the IP address.

## Objective

Build a machine learning model that predicts the probability that the first transaction of a new user on XYZ's website is fraudulent.

## Cost Considerations

- False Negative (Failing to identify a fraudulent transaction): XYZ loses money equivalent to the price of the fraudulently purchased product.
- False Positive (Incorrectly flagging a valid transaction as fraudulent): Inconvenience to XYZ customers, valued at $8 per incorrect flag.

## Project Structure
```
ecommerce-fraud-detection/
│
├── data/
│   ├── raw/
│   │   └── transactions.csv
│   └── processed/
│       └── transactions_processed.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── model_evaluation.py
│
├── tests/
│   ├── test_data_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_model_training.py
│   └── test_model_evaluation.py
│
├── .gitignore
├── README.md
└── requirements.txt
```


## Data Processing:
```
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Handle missing values
    df.fillna(method='ffill', inplace=True)
    
    # Convert categorical variables to numerical
    categorical_cols = ['source', 'browser', 'sex', 'country']
    encoder = OneHotEncoder()
    encoded_features = encoder.fit_transform(df[categorical_cols])
    df_encoded = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names(categorical_cols))
    
    df = df.drop(categorical_cols, axis=1)
    df = pd.concat([df, df_encoded], axis=1)
    
    # Convert signup_time and purchase_time to datetime
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    # Feature engineering: time difference between signup and purchase
    df['signup_purchase_diff'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    
    return df

if __name__ == "__main__":
    data = load_data('data/raw/transactions.csv')
    processed_data = preprocess_data(data)
    processed_data.to_csv('data/processed/transactions_processed.csv', index=False)
```
