#Selection of libraries for our Project
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le=LabelEncoder()
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()

from logging import StreamHandler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Intialize thr models
lr=LinearRegression()
dtr=DecisionTreeRegressor()
rfr=RandomForestRegressor()

from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error

seoul_df=pd.read_csv('SeoulBikeData.csv', encoding='latin-1')

seoul_df.head()

#TITLE OF THE APP


st.title("Predicting Bicycle demand in Seoul: A machine Learning Model")

#data overview
st.header("data Overview for first 10 row")
st.write(seoul_df.head(10))   

st.header("Dataset Metadata")

# Number of rows and columns
num_rows, num_columns = seoul_df.shape
st.write(f"**Number of rows:** {num_rows}")
st.write(f"**Number of columns:** {num_columns}")

# Overview of column names and types
st.write("**Column Names and Data Types:**")
st.write(seoul_df.dtypes)

st.header("Statistical Summary")

# Display descriptive statistics
st.write("**Descriptive Statistics of the Dataset:**")
st.write(seoul_df.describe())



#Encoding categorical columns. Label encoding for features 'Holiday' and 'Functioning Day', and OHE (One hot enconding) for feature 'Seasons'
print(seoul_df['Seasons'].value_counts(), seoul_df['Holiday'].value_counts(), seoul_df['Functioning Day'].value_counts())

seoul_df['Holiday']=le.fit_transform(seoul_df['Holiday'])
seoul_df['Functioning Day']=le.fit_transform(seoul_df['Functioning Day'])



ohe=OneHotEncoder()
encoder=ohe.fit_transform(seoul_df[['Seasons']])
encoder_arr=encoder.toarray()
df_seasons=pd.DataFrame(encoder_arr, columns=ohe.get_feature_names_out(['Seasons']), dtype=int)

seoul_fin_df=pd.concat([seoul_df, df_seasons], axis=1)
seoul_fin_df.drop('Seasons', axis=1, inplace=True)

num_cols=seoul_fin_df[['Rented Bike Count','Hour','Temperature(°C)','Humidity(%)','Wind speed (m/s)','Visibility (10m)','Dew point temperature(°C)','Solar Radiation (MJ/m2)','Rainfall(mm)','Snowfall (cm)']]
cat_cols=seoul_fin_df.drop(num_cols, axis=1)
num_cols=ss.fit_transform(num_cols)
num_df=pd.DataFrame(num_cols, columns=['Rented Bike Count','Hour','Temperature(°C)','Humidity(%)','Wind speed (m/s)','Visibility (10m)','Dew point temperature(°C)','Solar Radiation (MJ/m2)','Rainfall(mm)','Snowfall (cm)'])
seoul=pd.concat([num_df, cat_cols], axis=1)


#Split the data into 2 parts: input and output
X=seoul.drop(['Rented Bike Count', 'Date'], axis=1) #input - features

y=seoul['Rented Bike Count'] #Output - Target variable

#split the data into 2 parts: train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)



# run the predictive Models

lr = LinearRegression()
lr.fit(X_train, y_train)  # Model training

dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)  # Model training

rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)  # Model training


mod=st.selectbox("Select a model", ("Linear Regression", "Random Forest", "Decision tree"))


models={
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "decision tree":DecisionTreeRegressor()
}

#train the Model
selected_model=models[mod] #initializing the selected model

# train the selected model
selected_model.fit(X_train,y_train)

#make predictions
y_pred=selected_model.predict(X_test)

#model evaluation
r2=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)

#display results
st.write(f"r2 Score: {r2}")
st.write(f"Mean Square Error:{mse}")
st.write(f"Mean Absolute Error:{mae}")



st.header("Enter Input Values for Prediction")

user_input = {}
for column in X.columns:
    user_input[column] = st.number_input(
        column,
        min_value=float(np.min(X[column])),
        max_value=float(np.max(X[column])),
        value=float(np.mean(X[column])),
    )

# Convert user input to DataFrame
user_input_df = pd.DataFrame([user_input])


# Ensure the 'Seasons' column is handled properly
if 'Seasons' in seoul_df.columns:
    # Add a dropdown for 'Seasons'
    user_input['Seasons'] = st.selectbox("Select Season", options=seoul_df['Seasons'].unique())
    
    # Apply one-hot encoding for 'Seasons'
    if 'Seasons' in user_input_df.columns:
        seasons_ohe = ohe.transform(user_input_df[['Seasons']]).toarray()
        seasons_df = pd.DataFrame(seasons_ohe, columns=ohe.get_feature_names_out(['Seasons']))
        user_input_df = pd.concat([user_input_df, seasons_df], axis=1)
        user_input_df.drop('Seasons', axis=1, inplace=True)
    else:
        st.error("The column 'Seasons' is missing from the user input.")
else:
    st.error("'Seasons' column not found in the original dataset.")





# Standardize the user input
user_input_scaled = ss.transform(user_input_df)

# Predict using the selected model
predicted_rentals = selected_model.predict(user_input_scaled)

# Display prediction
st.header("Predicted Output")
st.write(f"The predicted number of rented bikes for the given inputs is: **{int(predicted_rentals[0])}**")