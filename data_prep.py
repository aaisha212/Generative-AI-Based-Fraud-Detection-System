
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


df = pd.read_csv('onlinefraud.csv')


# Encoding Transaction Type
df = pd.get_dummies(df, columns=['type'], prefix='type')

# Label Encoding
le = LabelEncoder()
df['nameOrig_encoded'] = le.fit_transform(df['nameOrig'])
df['nameDest_encoded'] = le.fit_transform(df['nameDest'])

# Drop unnecessary columns
df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud', 'type'], axis=1, inplace=True)

# Column Standardization
cols_to_standardize = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
scaler = StandardScaler()
df[cols_to_standardize] = scaler.fit_transform(df[cols_to_standardize])

# Display the cleaned data
print(df.head())

# Save the cleaned data
df.to_csv('cleaned_data.csv', index=False)
print("Cleaned data saved to 'cleaned_data.csv'.")