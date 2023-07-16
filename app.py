import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import folium
from streamlit_folium import folium_static

# Load the trained models
with open('./Models/M1_DT.pkl', 'rb') as f:
    M1_model_DT = pickle.load(f)
with open("./Models/M1_RF.pkl", 'rb') as f:
    M1_model_RF = pickle.load(f)
with open("./Models/M1_GD.pkl", 'rb') as f:
    M1_model_GD = pickle.load(f)
with open("./Models/M1_Ridge.pkl", 'rb') as f:
    M1_Model_Ridge = pickle.load(f)
with open("./Models/M1_Elastic.pkl", 'rb') as f:
    M1_Model_Elastic = pickle.load(f)

with open("./Models/M2_DT.pkl", 'rb') as f:
    M2_model_DT = pickle.load(f)
with open("./Models/M2_RF.pkl", 'rb') as f:
    M2_model_RF = pickle.load(f)
with open("./Models/M2_GD.pkl", 'rb') as f:
    M2_model_GD = pickle.load(f)
with open("./Models/M2_Ridge.pkl", 'rb') as f:
    M2_model_Ridge = pickle.load(f)

# Zip Code Dataframe
zip_df = pd.read_csv('./data/uszips.csv')

# Load in modified ML dataframe
df = pd.read_csv('./data/df.csv')

def create_map():
    # Merge the dataframes based on the zip code column
    merged_df = pd.merge(df, zip_df, left_on='zip_code', right_on='zip', how='inner')

    # Compute the average values for each county
    county_stats = merged_df.groupby('county_name').agg(
        avg_population=('population', 'mean'),
        avg_density=('density', 'mean'),
        avg_price=('price', 'mean'),
        total_price=('price', 'sum'),
        total_size=('house_size', 'sum')
    ).reset_index()

    # Calculate price per square foot
    county_stats['avg_price_per_sqft'] = county_stats['total_price'] / county_stats['total_size']

    # Drop the total price and size as they are no longer needed
    county_stats = county_stats.drop(columns=['total_price', 'total_size'])

    # Create a map centered around the mean latitude and longitude
    m = folium.Map(location=[merged_df['lat'].mean(), merged_df['lng'].mean()], zoom_start=10)

    # Add markers for each county
    for county in county_stats['county_name'].unique():
        county_data = county_stats[county_stats['county_name'] == county].iloc[0]
        lat, lng = merged_df[merged_df['county_name'] == county][['lat', 'lng']].mean()

        folium.Marker(
            [lat, lng],
            popup=(
                f"County: {county}<br>"
                f"Avg Population: {county_data['avg_population']}<br>"
                f"Avg Density: {county_data['avg_density']}<br>"
                f"Avg Price: {county_data['avg_price']}<br>"
                f"Avg Price per sqft: {county_data['avg_price_per_sqft']}"
            )
        ).add_to(m)

    return m

# Streamlit Application
def main():
    st.sidebar.title("Navigation Pane")
    tabs = ["Real-Estate Fortune Teller", "Around the US Map"]
    selected_tab = st.sidebar.radio("Tab Options", tabs)

    if selected_tab == "Real-Estate Fortune Teller":
        col1, col2 = st.columns(2)

        with col1:
            st.title('Real Estate Price Predictor')

            # Input features
            state = st.selectbox("State", options=df['state'].unique())
            # Filter the dataframe based on the state selection
            df_filtered_by_state = df[df['state'] == state]

            city = st.selectbox("City", options=df_filtered_by_state['city'].unique())
            # Further filter the dataframe based on the city selection
            df_filtered_by_city = df_filtered_by_state[df_filtered_by_state['city'] == city]

            zip_code = st.selectbox("Zip Code", options=df_filtered_by_city['zip_code'].unique().astype(int))

            bed = st.selectbox("Enter number of bedrooms", options=list(range(1,10)))
            bath = st.selectbox("Enter number of bathrooms", options=list(range(1,10)))
            acre_lot = st.selectbox("Enter acre lot", options=list(range(1,100)))
            house_size = st.number_input("Enter house size", min_value=500, max_value=12000, value=1000)
            
            # Make dataframe from inputs
            data = pd.DataFrame(data=[[bed, bath, acre_lot, house_size, city, state, zip_code]], columns=['bed', 'bath', 'acre_lot', 'house_size', 'city', 'state', 'zip_code'])
            
            # Encoding categorical variables
            data = pd.get_dummies(data, columns=['city', 'state', 'zip_code'])

            # Select the model
            model_choice = st.selectbox("Select Model", options=['Decision Tree ~ 87%', 'Random Forest ~ 95%' ,'Gradient Boost ~93%' ,'Ridge CV ~77%' , 'ElasticNet CV ~ 2%'])


            X = df[['bed', 'bath', 'acre_lot', 'zip_code', 'house_size', 'city', 'state']]
            y = df['price']
            X = pd.get_dummies(X, columns=['city', 'state', 'zip_code'])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


            if st.button('Predict Price'):
                # Ensure the user input dataframe has the same features as the training set
                data = data.reindex(columns=X_train.columns, fill_value=0)
                
                if model_choice == 'Decision Tree ~ 87%':
                    prediction = M2_model_DT.predict(data)
                    st.write(f"The Decision Tree model, which has an overall accuracy of 87%, on the training data, predicts the price to be ${round(prediction[0], 2)}")
                elif model_choice == 'Random Forest ~ 95%':
                    prediction = M1_model_RF.predict(data)
                    st.write(f"The Random Forest model, which has an overall accuracy of 95%, on the training data, predicts the price to be ${round(prediction[0], 2)}")
                elif model_choice == 'Gradient Boost ~93%':
                    prediction = M2_model_GD.predict(data)
                    st.write(f"The Gradient Boosting model, which has an overall accuracy of 93%, on the training data, predicts the price to be ${round(prediction[0], 2)}")
                elif model_choice == 'Ridge CV ~77%':
                    prediction = M1_Model_Ridge.predict(data)
                    st.write(f"The Ridge CV model, which has an overall accuracy of 77%, on the training data, predicts the price to be ${round(prediction[0], 2)}")
                elif model_choice == 'ElasticNet CV ~ 2%':
                    prediction = M1_Model_Elastic.predict(data)
                    st.write(f"The Elastic Net model, which has an overall accuracy of 2%, on the training data, predicts the price to be ${round(prediction[0], 2)}")

        with col2:
            col2.title("Click the pinpoint!")
            # Get the selected zip code details
            zip_details = zip_df[zip_df['zip'] == zip_code]
            # Create a map centered around the selected zip code
            m = folium.Map(location=[zip_details['lat'].values[0], zip_details['lng'].values[0]], zoom_start=13)

            # Add a marker for the selected zip code
            folium.Marker(
                [zip_details['lat'].values[0], zip_details['lng'].values[0]],
                popup=f"<i>County: {zip_details['county_names_all'].values[0]}</i><br>\
                        <i>Population: {zip_details['population'].values[0]}</i><br>\
                        <i>Density: {zip_details['density'].values[0]}</i><br>\
                        <i>State: {zip_details['state_name'].values[0]}</i>",
            ).add_to(m)

            # Display the map in the Streamlit app
            folium_static(m)



    elif selected_tab == "Around the US Map":
        st.title("Map")
        # Create the scatter plot map
        scatter_map = create_map()

        # Display the map in the Streamlit app
        folium_static(scatter_map)
if __name__ == '__main__':
    main()
