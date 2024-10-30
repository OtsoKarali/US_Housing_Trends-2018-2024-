#!/usr/bin/env python
# coding: utf-8

# # Import Statements

# In[36]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from tqdm import tqdm
import time


# # Loading in Dataset

# In[37]:


#Dataset link : https://www.kaggle.com/datasets/clovisdalmolinvieira/us-housing-trends-values-time-and-price-cuts
file_path = "/Users/otsok/Downloads/USRealEstateTrends.csv"
data = pd.read_csv(file_path)

# Coordinate Website : https://simplemaps.com/data/us-cities
city_coords = pd.read_csv("/Users/otsok/Downloads/simplemaps_uscities_basicv1.79/uscities.csv")


# In[38]:


data.head()


# In[39]:


# Check the percentage of missing values in each column
missing_percentage = data.isnull().mean() * 100

# Drop columns with more than 50% missing values, except for 'RegionName'
threshold = 50
columns_to_drop = missing_percentage[(missing_percentage > threshold) & (missing_percentage.index != 'RegionName')].index
data_cleaned = data.drop(columns=columns_to_drop)

# Ensure 'RegionName' is still in the dataset
if 'RegionName' not in data_cleaned.columns:
    data_cleaned['RegionName'] = data['RegionName']

# Fill remaining missing values with the mean in numeric columns only
numeric_cols = data_cleaned.select_dtypes(include=[np.number]).columns
data_cleaned[numeric_cols] = data_cleaned[numeric_cols].fillna(data_cleaned[numeric_cols].mean())

# Display the first few rows of the cleaned data to confirm 'RegionName' is retained
print(data_cleaned.head())


# In[40]:


# Check column names in both datasets
print("Columns in data_cleaned:", data.columns)
print("Columns in city_coords:", city_coords.columns)

# Once confirmed, proceed with further steps based on the actual column names


# In[41]:


# Print column names to verify structure
print("Columns in data_cleaned:", data_cleaned.columns)

# Check if 'RegionName' or an equivalent column exists and split it if found
if 'RegionName' in data_cleaned.columns:
    # Split 'RegionName' to create 'city' and 'state_id'
    data_cleaned[['city', 'state_id']] = data_cleaned['RegionName'].str.split(', ', expand=True)
elif 'City_State_Column_Name' in data_cleaned.columns:  # Replace 'City_State_Column_Name' if it's different
    # Adjust to the actual name if different
    data_cleaned[['city', 'state_id']] = data_cleaned['City_State_Column_Name'].str.split(', ', expand=True)
else:
    print("Error: Expected column for city and state (e.g., 'RegionName') not found in data_cleaned.")

# Continue with the rest of the code if columns are found and split
if 'city' in data_cleaned.columns and 'state_id' in data_cleaned.columns:
    # Standardize formats
    data_cleaned['city'] = data_cleaned['city'].str.upper().str.strip()
    data_cleaned['state_id'] = data_cleaned['state_id'].str.upper().str.strip()
    city_coords['city'] = city_coords['city'].str.upper().str.strip()
    city_coords['state_id'] = city_coords['state_id'].str.upper().str.strip()

    # Merge and plot as before
    merged_data = pd.merge(data_cleaned, city_coords[['city', 'state_id', 'lat', 'lng']],
                           on=['city', 'state_id'], how='inner')
    merged_data = merged_data.rename(columns={'lat': 'Latitude', 'lng': 'Longitude'})

    print("Merged data shape:", merged_data.shape)
    print(merged_data.head())
else:
    print("Error: The necessary columns ('city' and 'state_id') were not created. Check column names.")


# In[42]:


import pandas as pd
import plotly.graph_objects as go

# Split 'RegionName' in 'data_cleaned' to create separate 'city' and 'state_id' columns
data_cleaned[['city', 'state_id']] = data_cleaned['RegionName'].str.split(', ', expand=True)

# Standardize city and state_id formats to uppercase
data_cleaned['city'] = data_cleaned['city'].str.upper().str.strip()
data_cleaned['state_id'] = data_cleaned['state_id'].str.upper().str.strip()
city_coords['city'] = city_coords['city'].str.upper().str.strip()
city_coords['state_id'] = city_coords['state_id'].str.upper().str.strip()

# Check overlap in unique city-state combinations
unique_data_cities = set(data_cleaned[['city', 'state_id']].apply(tuple, axis=1))
unique_coords_cities = set(city_coords[['city', 'state_id']].apply(tuple, axis=1))
overlap = unique_data_cities.intersection(unique_coords_cities)

print(f"Unique city-state pairs in data_cleaned: {len(unique_data_cities)}")
print(f"Unique city-state pairs in city_coords: {len(unique_coords_cities)}")
print(f"Matching city-state pairs: {len(overlap)}")

# Merge on 'city' and 'state_id'
merged_data = pd.merge(data_cleaned, city_coords[['city', 'state_id', 'lat', 'lng']],
                       on=['city', 'state_id'], how='inner')

# Rename lat and lng to Latitude and Longitude for consistency
merged_data = merged_data.rename(columns={'lat': 'Latitude', 'lng': 'Longitude'})

# Check if merged_data has rows after the merge
print("Merged data shape:", merged_data.shape)
print(merged_data.head())

# Proceed if merge was successful
if not merged_data.empty:
    # Scale the home values for better visualization
    merged_data['scaled_home_value'] = merged_data['2024-05-HomeValue'] / 1000  # Scale down for bar height

    # Extract data for plotting
    lats = merged_data['Latitude']
    lons = merged_data['Longitude']
    heights = merged_data['scaled_home_value']

    # Create the 3D scatter plot
    fig = go.Figure(go.Scatter3d(
        x=lons,
        y=lats,
        z=heights,
        text=merged_data['city'],
        mode='markers',
        marker=dict(
            size=5,
            color=heights,
            colorscale='Viridis',
            colorbar=dict(title="Avg Home Value (in 1000s $)"),
            opacity=0.8
        )
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Home Value (in 1000s $)',
            xaxis=dict(backgroundcolor="rgb(200, 200, 230)"),
            yaxis=dict(backgroundcolor="rgb(230, 200,230)"),
            zaxis=dict(backgroundcolor="rgb(230, 230,200)"),
        ),
        title="Average Home Value by City in the United States (May 2024)",
        height=700
    )

    fig.show()
else:
    print("Error: The merged dataset is empty. Check the merge step for issues.")



# In[66]:


import plotly.express as px

fig = px.scatter_mapbox(
    merged_data,
    lat="Latitude",
    lon="Longitude",
    size="scaled_home_value",  # Circle size based on scaled home values
    color="scaled_home_value",  # Color based on scaled home values
    color_continuous_scale="Turbo",  # Choose a vibrant color scale
    size_max=18,  # Set a maximum size for the circles
    zoom=3,
    center=dict(lat=37.0902, lon=-95.7129),  # Center over the U.S.
    mapbox_style="carto-positron",
    title="Average Home Value by City in the United States (May 2024)"
)

# Adjust the layout
fig.update_layout(
    height=700,
    coloraxis_colorbar=dict(title="Avg Home Value (in 1000s $)")
)

fig.show()


# In[69]:


import plotly.express as px

# Filter out DC and Delaware
filtered_data = merged_data[(merged_data["state_id"] != "DC") & (merged_data["state_id"] != "DE")]

# Sort data alphabetically by state_id
filtered_data = filtered_data.sort_values(by="state_id")

fig = px.box(
    filtered_data,  # Use filtered data
    x="state_id",        # State names on the x-axis
    y="scaled_home_value",  # Home values on the y-axis
    points=False,  # Remove all individual points
    title="Distribution of Average Home Values by State (May 2024)",
    labels={
        "state_id": "State",
        "scaled_home_value": "Home Value (in 1000s $)"
    },
    color_discrete_sequence=["#1f77b4"]  # Set a single color for all box plots
)

fig.update_layout(
    height=700,
    xaxis_title="State",
    yaxis_title="Average Home Value (in 1000s $)",
    showlegend=False  # Hide legend as all boxes are the same color
)

fig.show()


# In[74]:


# Convert Date to year and calculate yearly averages
filtered_data_long['Year'] = pd.to_datetime(filtered_data_long['Date']).dt.year
yearly_data = filtered_data_long.groupby(['Year', 'state_id'])['HomeValue'].mean().reset_index()

fig = go.Figure()

# Plot each state with yearly averaged values
for state in key_states:
    state_data = yearly_data[yearly_data["state_id"] == state]
    fig.add_trace(go.Scatter(
        x=state_data["Year"],
        y=state_data["HomeValue"],
        mode='lines+markers',
        name=state,
        line_shape='linear'
    ))

# Update layout for a clean, continuous line chart
fig.update_layout(
    title="Yearly Average Home Prices by State",
    xaxis_title="Year",
    yaxis_title="Average Home Value (in 1000s $)",
    height=600,
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True),
    legend=dict(title="State")
)

fig.show()



# In[77]:


# Select only columns related to home values over time
home_value_columns = [col for col in data_cleaned.columns if "HomeValue" in col]

# Extract data for initial and final months for each state
initial_values = data_cleaned[home_value_columns[0]]
final_values = data_cleaned[home_value_columns[-1]]

# Calculate percentage growth for each state and create a new DataFrame to avoid fragmentation
percentage_growth = ((final_values - initial_values) / initial_values) * 100
growth_df = pd.DataFrame({
    'StateName': data_cleaned['StateName'],
    'Percentage_Growth': percentage_growth
})

# Remove any potential duplicates
growth_df = growth_df.drop_duplicates(subset=['StateName'])

# Sort by highest growth
growth_df = growth_df.sort_values(by='Percentage_Growth', ascending=False).reset_index(drop=True)

# Display the full resulting DataFrame with percentage growth for each state
growth_df


# In[86]:


fig = px.choropleth(
    growth_df,
    locations='StateName',
    locationmode="USA-states",
    color='Percentage_Growth',
    color_continuous_scale="Plasma",
    scope="usa",
    title="Percentage Growth in Housing Prices by State",
    labels={'Percentage_Growth': 'Growth (%)'}
)

fig.update_layout(
    title_x=0.5,
    geo=dict(showlakes=True, lakecolor='rgb(255, 255, 255)')
)
fig.show()


# In[89]:


import plotly.express as px

# Use the last column in the home value data (May 2024) to get the most recent average home values
latest_home_values = data_cleaned[['StateName', '2024-05-HomeValue']].groupby('StateName').mean().reset_index()
latest_home_values.columns = ['StateName', 'Average_Home_Value']

# Plotting the choropleth map
fig = px.choropleth(
    latest_home_values,
    locations='StateName',
    locationmode="USA-states",
    color='Average_Home_Value',
    color_continuous_scale="YlOrRd",
    scope="usa",
    title="Average Home Value by State (May 2024)",
    labels={'Average_Home_Value': 'Average Home Value ($)'}
)

fig.update_layout(
    title_x=0.5,
    geo=dict(showlakes=True, lakecolor='rgb(255, 255, 255)')
)
fig.show()


# In[93]:


import re
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Filter for columns with HomeValue and DaysPending across all months
homevalue_columns = [col for col in data_cleaned.columns if "HomeValue" in col]
dayspending_columns = [col for col in data_cleaned.columns if "DaysPending" in col]

# Remove "HomeValue" and "DaysPending" suffixes to isolate the dates
homevalue_dates = [re.sub(r"-HomeValue", "", col) for col in homevalue_columns]
dayspending_dates = [re.sub(r"-DaysPending", "", col) for col in dayspending_columns]

# Convert isolated dates to datetime format
monthly_avg_homevalue = data_cleaned[homevalue_columns].mean(axis=0)
monthly_avg_homevalue.index = pd.to_datetime(homevalue_dates, errors='coerce', format='%Y-%m')

monthly_avg_dayspending = data_cleaned[dayspending_columns].mean(axis=0)
monthly_avg_dayspending.index = pd.to_datetime(dayspending_dates, errors='coerce', format='%Y-%m')

# Check for any parsing issues
print("HomeValue Dates:", monthly_avg_homevalue.index)
print("DaysPending Dates:", monthly_avg_dayspending.index)

# Merge into a single DataFrame for regression analysis
monthly_data = pd.DataFrame({
    'HomeValue': monthly_avg_homevalue,
    'DaysPending': monthly_avg_dayspending
}).dropna()

# Reshape for regression
X = monthly_data['DaysPending'].values.reshape(-1, 1)
y = monthly_data['HomeValue'].values

# Fit the regression model
model = LinearRegression()
model.fit(X, y)

# Predict and calculate R²
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

# Plotting the regression
plt.figure(figsize=(10, 6))
plt.scatter(monthly_data['DaysPending'], monthly_data['HomeValue'], color='blue', label='Data points')
plt.plot(monthly_data['DaysPending'], y_pred, color='red', label=f'Regression Line (R²={r2:.2f})')
plt.xlabel("Days Pending")
plt.ylabel("Average Home Value (in 1000s $)")
plt.title("Regression Analysis of Home Value and Days Pending")
plt.legend()
plt.show()

# Display the coefficient and intercept
print(f"Regression Coefficient (Slope): {model.coef_[0]}")
print(f"Regression Intercept: {model.intercept_}")
print(f"R² Score: {r2}")



# In[95]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Prepare the time series data
# Extract columns related to home values and strip out "HomeValue" to get just the date
home_value_columns = [col for col in data_cleaned.columns if "HomeValue" in col]
home_value_data = data_cleaned[home_value_columns].mean(axis=0)

# Rename columns to extract the date portion and convert to datetime
home_value_data.index = pd.to_datetime([col.split('-')[0] + '-' + col.split('-')[1] for col in home_value_columns], format='%Y-%m')

# Step 2: Visualize the time series data
plt.figure(figsize=(10, 6))
plt.plot(home_value_data, label='Average Home Value')
plt.title('Average Home Value Over Time')
plt.xlabel('Time')
plt.ylabel('Home Value (in 1000s $)')
plt.legend()
plt.show()

# Step 3: Fit an ARIMA model to the time series data
model = ARIMA(home_value_data, order=(1, 1, 1))  # Adjust ARIMA order if needed
fitted_model = model.fit()

# Step 4: Forecast future values
forecast = fitted_model.forecast(steps=12)  # Forecasting the next 12 months

# Step 5: Plot the forecast
plt.figure(figsize=(10, 6))
plt.plot(home_value_data, label='Historical Data')
plt.plot(forecast, label='Forecast', color='red')
plt.title('Forecasted Average Home Value')
plt.xlabel('Time')
plt.ylabel('Home Value (in 1000s $)')
plt.legend()
plt.show()

# Optional: Print model summary
print(fitted_model.summary())



# In[98]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

# Encode categorical variables
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Predict on test data and calculate R2 score
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
print("R2 Score: ", r2)



# In[99]:


from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Perform 5-fold cross-validation and calculate R^2 scores for each fold
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
kfolds = list(range(1, 6))

# Plot the cross-validation scores
plt.figure(figsize=(10, 5))
sns.lineplot(x=kfolds, y=scores, marker='o')
plt.xlabel("K-fold")
plt.ylabel("R^2 Score")
plt.title("Cross-Validation Scores - Average R^2: {:.2f}".format(np.mean(scores)))
plt.show()


# In[100]:


from sklearn.model_selection import cross_val_predict

# Predict using cross-validation for residual analysis
predicted = cross_val_predict(model, X_train, y_train, cv=5, n_jobs=-1)
residual = y_train - predicted

# Residual plot
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_train, y=residual)
plt.axhline(0, color='red', linestyle='--', lw=2)
plt.xlabel("Observed Values")
plt.ylabel("Residuals")
plt.title("Residual Plot - Model Analysis")
plt.show()


# In[ ]:




