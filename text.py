import streamlit as st

class Info:
    def intro(self):
        st.header("🧬 ")
        st.markdown("""
        Brain tumors are abnormal growths of tissue within the brain that can disrupt 
        normal neural functions. Early and accurate detection is crucial for effective treatment, 
        as delayed diagnosis may lead to severe neurological complications.

        Our project applies **Convolutional Neural Networks (CNNs)** — a type of deep learning model — 
        to analyze MRI (Magnetic Resonance Imaging) scans. CNNs are capable of recognizing subtle 
        differences in texture, intensity, and structure across MRI slices, which allows them to 
        distinguish between healthy and tumorous brain regions with high accuracy.

        ---
        ### 🔍 How it Works
        1. **Image Input**: An MRI brain scan is uploaded.
        2. **Preprocessing**: The image is resized and normalized to match the model input format.
        3. **Prediction**: A CNN model trained on thousands of MRI images classifies the scan 
           as either **Tumor** or **No Tumor**.
        4. **Grad-CAM Visualization**: Gradient-weighted Class Activation Mapping highlights 
           the regions in the brain where the model focuses most while predicting.
        
        ---
        ### 💡 Why It Matters
        - Early tumor detection through MRI scans helps **reduce mortality** and **improves recovery outcomes**.
        - AI-assisted diagnosis offers **faster and more consistent analysis** for radiologists.
        - Grad-CAM provides **explainability**, ensuring medical transparency and trust in AI-driven decisions.

        ---
        ### 🧠 Technologies Used
        - **TensorFlow / Keras** – Deep Learning framework  
        - **OpenCV** – Image processing  
        - **Streamlit** – Web interface  
        - **Grad-CAM** – Model interpretability visualization  
        - **Python** – Core programming language  
        """)

    def symptoms(self):
        st.markdown("## ⚠️ Common Symptoms of Brain Tumor")
        st.markdown("""
        - Persistent or severe headaches  
        - Blurred or double vision  
        - Nausea and vomiting  
        - Seizures  
        - Personality or memory changes  
        - Difficulty speaking, walking, or balancing  
        """)

    def causes(self):
        st.markdown("## 🧩 Possible Causes and Risk Factors")
        st.markdown("""
        - Genetic mutations or hereditary factors  
        - Exposure to ionizing radiation  
        - Family history of brain tumors  
        - Environmental toxins or carcinogens  
        - Age-related cell abnormalities  

        While not all brain tumors are cancerous, timely screening through MRI 
        is one of the most reliable diagnostic techniques.
        """)

class Headers:
    def mainHead(self):
        st.title("🧠 BrainScanX: Deep Learning–Based Brain Tumor Detection with Grad-CAM Explainability")
        st.subheader("Deep Learning-Based MRI Analysis with Grad-CAM Explainability")

    def about(self):
        st.sidebar.header("About the Project")
        st.sidebar.markdown("""
        Upload an MRI image to detect the presence of a brain tumor 
        using a trained deep learning model.
        """)

    def scanConsole(self):
        st.sidebar.header("Prediction Console")

    def scanInst(self):
        st.header("🔬 Upload MRI Image for Prediction")

    def contributions(self):
        st.sidebar.header("Developed by:")
        st.sidebar.markdown("""
        - Siya Amrutkar  
        - Sharadhi Rao  
        - Ayushi Bindroo
        """)
