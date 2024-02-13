# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np 
import plotly.express as px
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.decomposition import PCA

# Load data
@st.cache_resource()
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/aritra36-ck/Capstone_File/main/customer_info.csv')
    return df.copy()  # Return a copy of the DataFrame to avoid mutation issues

df = load_data()

#Feature Selection
features = df.copy()
features = features.drop(['customer_id', 'total_charges','gender','name','state','email'], axis=1)

## Scale Tenure and Monthly Charges
scaler = StandardScaler()
features[['tenure', 'monthly_charges']] = scaler.fit_transform(features[['tenure', 'monthly_charges']])

#Selecting all variables except tenure and Monthly Charges
encoder=OrdinalEncoder()
features[features.columns[~features.columns.isin(['tenure','monthly_charges'])]] = encoder.fit_transform(features[features.columns[~features.columns.isin(['tenure','monthly_charges'])]])

# Initial data insights
st.write("### Initial Data Insights:")
st.write("Explore some initial insights from the dataset here, such as summary statistics or a few sample rows:")
st.write(df.describe())  # Display summary statistics

st.sidebar.title("Features")
selected_features = ['senior_citizen', 'partner', 'dependents', 'tenure', 'phone_service', 'multiple_lines', 'internet_service', 'online_security', 'online_backup', 'device_protection',  'tech_support', 'streaming_tv', 'streaming_movies', 'contract', 'paperless_billing', 'payment_method', 'monthly_charges', 'churn']
# selected_features = ['senior_citizen', 'partner', 'dependents', 'tenure']

# Display the list as bullet points
st.sidebar.write("<ul>", unsafe_allow_html=True)  # Start unordered list
for item in selected_features:
    st.sidebar.write(f"<li>{item}</li>", unsafe_allow_html=True)  # List item
st.sidebar.write("</ul>", unsafe_allow_html=True)  # End unordered list

# Button for training the dataset
if st.sidebar.button('Train the dataset'):
    # KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(features)
    st.session_state['trained'] = True  # Set the 'trained' state to True


# Display the visualization using Plotly
if st.sidebar.button('View visualization'):
    if 'trained' in st.session_state and st.session_state['trained']: # Check if the dataset is trained
        # PCA
        pca = PCA(n_components=3)  # Use 3 components for 3D plot
        df[['PC1', 'PC2', 'PC3']] = pca.fit_transform(features)

          # 3D Scatter plot with Plotly
        fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Cluster', title="Customer Segments",
                            labels={'PC1': 'Component_1', 'PC2': 'Component_2', 'PC3': 'Component_3'},
                            opacity=0.8, size_max=10, color_continuous_scale='viridis')
        st.plotly_chart(fig)

        # Calculate average feature values for each cluster
        cluster_means = features.groupby('Cluster')[item].mean()

        # Display the average feature values for each cluster
        st.write("Segment Profiling: Understanding Customer Behavior through **Cluster Analysis**:")
        st.write(cluster_means)

        cluster_counts = df['Cluster'].value_counts()
        cluster_counts_df = pd.DataFrame({'Cluster': cluster_counts.index, 'Count': cluster_counts.values})
        cluster_counts_df = cluster_counts_df.sort_values(by='Cluster')   

          # Create bar chart using Plotly Express with custom colors and data labels
        fig = px.bar(cluster_counts_df, x='Cluster', y='Count', title='Number of Data Points in Each Cluster',
        labels={'Cluster': 'Cluster Label', 'Count': 'Number of Data Points'},
        color='Cluster', color_continuous_scale='viridis', text='Count')

        fig.update_traces(textposition='outside', textfont=dict(color='black', size=12))
        st.plotly_chart(fig)

        # loadings
        # loadings = pca.components_

        # # Create a DataFrame to display the loadings
        # loadings_df = pd.DataFrame(loadings, columns=features[selected_features].columns, index=['PC1', 'PC2', 'PC3'])

        # # Display the loadings
        # st.write("Unveiling Data Patterns: Exploring Key Drivers through **Loadings Analysis**:")
        # st.write(loadings_df)

        # # Calculate the overall effect of each feature
        # overall_effect = loadings_df.abs().sum()

        # # Find the feature with the highest overall effect
        # most_effective_feature = overall_effect.idxmax()
        # highest_effect_value = overall_effect.max()

        # # Display the result
        # st.write("Overall Effect of Features:")
        # st.write(overall_effect)
        # st.write(f"The feature with the highest overall effect is **{most_effective_feature}** with a total effect of **{highest_effect_value}**.")
        # Inference
        st.markdown("**Inference:**")
        st.markdown("The 3D visualization displays customer segments based on the selected features for clustering.")
        st.markdown("Each cluster represents a group of customers with similar characteristics.")
        st.markdown("This information can be used to tailor marketing strategies, improve sales, and understand customer behavior.")
        
    else:
        st.sidebar.error('Please train the dataset first.')

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Team 4 ")
st.sidebar.markdown("- Gajalakshmi")
st.sidebar.markdown("- Danish Anjum")
st.sidebar.markdown("- Praveen Asokan") 
st.sidebar.markdown("- Aritra Chakraborty")   
