# Install necessary libraries
!pip install pandas numpy matplotlib seaborn scikit-learn mlxtend

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# Configure visual settings
sns.set(style="whitegrid")


# Sample data for transactions
transactions_data = {
    'transaction_id': [1, 2, 3, 4, 5],
    'customer_id': [101, 102, 103, 104, 105],
    'product_id': [201, 202, 203, 204, 205],
    'quantity': [2, 1, 3, 2, 1],
    'price': [20, 30, 40, 25, 50],
    'transaction_date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
}

# Sample data for customers
customers_data = {
    'customer_id': [101, 102, 103, 104, 105],
    'age': [25, 34, 45, 23, 35],
    'gender': ['M', 'F', 'F', 'M', 'M'],
    'location': ['NY', 'CA', 'TX', 'FL', 'NV']
}

# Sample data for products
products_data = {
    'product_id': [201, 202, 203, 204, 205],
    'product_name': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'],
    'product_category': ['Category 1', 'Category 2', 'Category 3', 'Category 1', 'Category 2']
}

# Convert to DataFrames
transactions = pd.DataFrame(transactions_data)
customers = pd.DataFrame(customers_data)
products = pd.DataFrame(products_data)

# Preview data
print("Transactions Data:")
print(transactions.head())
print("\nCustomers Data:")
print(customers.head())
print("\nProducts Data:")
print(products.head())


# Merge datasets
data = transactions.merge(customers, on='customer_id').merge(products, on='product_id')

# Data cleaning and transformation
data['transaction_date'] = pd.to_datetime(data['transaction_date'])
data['total_value'] = data['quantity'] * data['price']

# Handle missing values and duplicates
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

print(data.info())


# Summary statistics
print(data.describe())

# Frequency distributions
print(data['product_category'].value_counts())


# Customer demographics
customer_demographics = data.groupby('customer_id').agg({
    'age': 'mean',
    'gender': lambda x: x.mode()[0],
    'location': lambda x: x.mode()[0]
}).reset_index()

print(customer_demographics.head())

# RFM Analysis
rfm = data.groupby('customer_id').agg({
    'transaction_date': lambda x: (datetime.now() - x.max()).days,
    'transaction_id': 'count',
    'total_value': 'sum'
}).reset_index()

rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

# Apply KMeans clustering for segmentation
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['segment'] = kmeans.fit_predict(rfm[['recency', 'frequency', 'monetary']])

print(rfm.head())

# Visualization of customer segments
sns.scatterplot(x='recency', y='monetary', hue='segment', data=rfm)
plt.title('Customer Segmentation')
plt.show()


# Transaction trends over time
data.set_index('transaction_date')['total_value'].resample('M').sum().plot()
plt.title('Monthly Transaction Value')
plt.xlabel('Month')
plt.ylabel('Total Value')
plt.show()

# Seasonality
data['month'] = data['transaction_date'].dt.month
monthly_sales = data.groupby('month')['total_value'].sum()
monthly_sales.plot(kind='bar')
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.show()


# Product popularity
top_products = data['product_name'].value_counts().head(10)
top_products.plot(kind='bar')
plt.title('Top 10 Products')
plt.xlabel('Product Name')
plt.ylabel('Number of Transactions')
plt.show()

# Price analysis
sns.scatterplot(x='price', y='quantity', data=data)
plt.title('Price vs Quantity')
plt.show()


# Prepare data for market basket analysis
basket = data.pivot_table(index='transaction_id', columns='product_name', values='quantity', aggfunc='sum').fillna(0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

# Apply Apriori algorithm
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)

print(rules.head())

# Visualization of association rules
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Association Rules')
plt.show()
