
#Clarinda Ewurama Awotwi
#10211100239
#Computer Science
#Level 400

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

def clustering_section():
    st.header("AI-Powered Clustering Playground")
    st.markdown("Dive into clustering! Upload a dataset, pick your features, and let’s explore how your data naturally groups together.")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="clustering")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("First Glance at the Dataset")
        st.dataframe(data.head())

        columns = data.columns.tolist()
        selected_features = st.multiselect("Choose Features for Clustering", columns)
        if len(selected_features) < 2:
            st.warning("Select at least **two** features to run clustering.")
            return

        if st.checkbox("Drop rows with missing values"):
            data = data.dropna(subset=selected_features)
            st.success("Cleaned up! Missing rows removed.")

        X = data[selected_features]

        if not all(np.issubdtype(X[feat].dtype, np.number) for feat in selected_features):
            st.error("Clustering only works with numeric data. Check your selected columns.")
            return

        num_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(X)
        data['Cluster'] = clusters

        st.subheader("Cluster Centroids")
        try:
            numeric_cols = data.select_dtypes(include='number').columns
            st.dataframe(data[numeric_cols].groupby('Cluster').mean().reset_index())
        except Exception as e:
            st.warning(f"Oops! Couldn't compute centroids: {e}")

        st.subheader("Cluster Visualization")

        if len(selected_features) == 2:
            fig = px.scatter(
                data, x=selected_features[0], y=selected_features[1],
                color=data['Cluster'].astype(str),
                symbol=data['Cluster'].astype(str),
                title="2D Cluster View",
                labels={"color": "Cluster"}
            )
            fig.update_traces(marker=dict(size=10, line=dict(width=1, color='black')))
            st.plotly_chart(fig, use_container_width=True)

        elif len(selected_features) == 3:
            fig = px.scatter_3d(
                data, x=selected_features[0], y=selected_features[1], z=selected_features[2],
                color=data['Cluster'].astype(str), title="3D Cluster View"
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Using PCA to reduce dimensions for visualization...")
            pca = PCA(n_components=3)
            components = pca.fit_transform(X)
            data[['PC1', 'PC2', 'PC3']] = components
            fig = px.scatter_3d(
                data, x='PC1', y='PC2', z='PC3',
                color=data['Cluster'].astype(str),
                title="PCA 3D Clustering View"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Download Your Results")
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Clustered Data",
            data=csv,
            file_name="clustered_dataset.csv",
            mime="text/csv"
        )
