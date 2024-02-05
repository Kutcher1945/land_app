import pandas as pd

# Load your CSV data
csv_data = pd.read_csv('data/land_area.csv')

# Function to remove the last three zeros from price
def remove_last_three_zeros(price_str):
    try:
        price = int(price_str)
        price //= 1000  # Remove the last three zeros
        return str(price)
    except ValueError:
        return price_str  # If it's not a valid integer, leave it as is

# Apply the function to the 'price' column
csv_data['price'] = csv_data['price'].apply(remove_last_three_zeros)

# Save the updated CSV file
csv_data.to_csv('data/land_area_updated.csv', index=False)

# Now, you can use 'data/land_area_updated.csv' in your Streamlit app

#config.toml base
#[theme] # You have to add this line

#primaryColor = '#FF8C02' # Bright Orange

#secondaryBackgroundColor = '#D1D8D0' # Lighter Blue



# [theme] # You have to add this line

# primaryColor = '#FF8C02' # Bright Orange

# backgroundColor = '#00325B' # Dark Blue

# secondaryBackgroundColor = '#55B2FF' # Lighter Blue

# textColor = '#FFFFFF'