import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# load the data
file_path = 'go_training_data.xlsx'
data = pd.read_excel(file_path, sheet_name='merge')

# fill missing 'hazard' values with "Missing"
data['hazard'].fillna("Missing", inplace=True)

# mapping of country_name to the hazard for that country
hazard_mapping = data.groupby('country_name')['hazard'].agg(lambda x: pd.Series.mode(x)[0])

# features to include in the model
features = [
    'appeal_type', 'num_beneficiaries', 'country_name', 'dtype_name', 'hazard'
]
X = data[features]
y = data['amount_requested']

# preprocessing for numerical features
numeric_features = ['appeal_type', 'num_beneficiaries']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# preprocessing for categorical features
categorical_features = ['country_name', 'dtype_name', 'hazard']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create a modeling pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', RandomForestRegressor(n_estimators=100, random_state=42))])

# fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# predict on the testing data
y_pred = pipeline.predict(X_test)

# calculate R² score and mse
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R² Score:", r2)
print("Mean Squared Error:", mse)

def get_user_input_with_hazard():
    print("Please provide the following details:")

    appeal_type = int(input("Appeal Type (numeric): "))
    num_beneficiaries = int(input("Number of Beneficiaries (numeric): "))
    country_name = input("Country Name (e.g., 'Kenya'): ")
    dtype_name = input("Disaster Type Name (e.g., 'Flood'): ")

    # find country hazard level
    hazard = hazard_mapping.get(country_name, "Unknown")

    print(f"{country_name} is a {hazard} hazard country.")

    # Create a DataFrame from the user inputs
    new_data = pd.DataFrame({
        'appeal_type': [appeal_type],
        'num_beneficiaries': [num_beneficiaries],
        'country_name': [country_name],
        'dtype_name': [dtype_name],
        'hazard': [hazard]
    })

    return new_data

# get user input
new_data = get_user_input_with_hazard()

# use the pipeline to preprocess the new data and make predictions
new_data_pred = pipeline.predict(new_data)

print("\nPredicted amount requested:", new_data_pred[0])