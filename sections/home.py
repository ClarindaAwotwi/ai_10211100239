
#Clarinda Ewurama Awotwi
#10211100239
#Computer Science
#Level 400

import streamlit as st

def home_section():
    st.title("AI Model Playground")
    st.markdown("""
    This is the **AI Model Playground** – your all-in-one hub for exploring the power of artificial intelligence.  
     Here, you can try out different machine learning techniques including regression analysis, clustering, neural networks, and even generative AI tools.

     This app was built using **Streamlit**, with backend support from **TensorFlow**, **Scikit-learn**, and **Google Generative AI**.  
     """
    )

    st.markdown("---")

    st.header("AI Model Playground Functions")
    
    st.markdown(
    """
        - 📈 **Regression**: Analyze and predict continuous trends from your data.
        - 🧩 **Clustering**: Group data into clusters using unsupervised learning.
        - 🧠 **Neural Networks**: Explore deep learning with TensorFlow.
        - 🤖 **Generative AI**: Generate content and insights using LLMs.

        Navigate to each section using the sidebar on your left!
        """
    )
    
    st.markdown("---")

    st.info("✨ Tip: Feel free to upload your own datasets or interact with the default ones provided!")