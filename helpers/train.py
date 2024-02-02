import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load your CSV data
data = pd.read_csv('data/train_data.csv')

# Split the dataset into input features (X_train) and target (y_train)
X_train = data.drop(columns=['price'])
y_train = data['price']

# Create and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust hyperparameters as needed

# Train the Random Forest model
rf_model.fit(X_train, y_train)

# Save the trained model to a .pkl file
joblib.dump(rf_model, 'random_forest_model.pkl')
