
#Clarinda Ewurama Awotwi
#10211100239
#Computer Science
#Level 400

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

def neural_network_section():
    st.header("Neural Network Classification")
    st.write("Upload a dataset, configure your neural network, train the model, and view classification results.")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="neural_network")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        columns = data.columns.tolist()
        target_column = st.selectbox("Select the target column (label)", columns)
        feature_columns = st.multiselect("Select input feature columns", [col for col in columns if col != target_column])

        if not feature_columns:
            st.warning("Please select at least one feature column to proceed.")
            return

        if st.checkbox("Drop rows with missing values"):
            data = data.dropna(subset=feature_columns + [target_column])
            st.success("Missing rows removed successfully.")

        if data[target_column].nunique() > 50 and not st.checkbox("Convert continuous target to categorical values"):
            st.error("The target column appears to be continuous. This module supports only classification tasks.")
            return

        if data[target_column].nunique() > 50 and st.checkbox("Bin target into 10 categories"):
            data[target_column] = pd.qcut(data[target_column], q=10, labels=False)
            st.info("Target column has been discretized into 10 classes.")

        X = data[feature_columns].values
        y = data[target_column].values

        if not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y)
        else:
            le = None

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        st.subheader("Model Configuration")
        epochs = st.slider("Number of training epochs", 1, 100, 10)
        batch_size = st.slider("Batch size", 8, 128, 32)
        learning_rate = st.number_input("Learning rate", min_value=0.0001, max_value=1.0, value=0.001, step=0.0001, format="%.4f")

        num_classes = len(np.unique(y_train))
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        st.subheader("Training Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()

