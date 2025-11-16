import streamlit as st
import time
from predictor import predict_tumor, generate_gradcam
from text import Info, Headers
from PIL import Image

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

Head = Headers()
InfoPage = Info()

Head.mainHead()
Head.about()

# ---------------- SIDEBAR INFO ---------------- #
if st.sidebar.checkbox("Show Project Info"):
    InfoPage.intro()
    InfoPage.symptoms()
    InfoPage.causes()

Head.scanConsole()

# ---------------- MAIN UI ---------------- #
if st.sidebar.checkbox("Open Prediction Console"):
    Head.scanInst()

    uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

    if st.button("Predict"):
        if uploaded_file is not None:
            with st.spinner("Analyzing MRI..."):
                # Save uploaded file temporarily
                with open("temp.jpg", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Progress bar for analysis
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i + 1)

                # Run prediction
                label, confidence = predict_tumor("temp.jpg")

                # ---------------- RESULTS SECTION ---------------- #
                col1, col2 = st.columns([1.5, 1.5])  # Balanced layout

                with col1:
                    st.markdown("### 🧠 Uploaded MRI Image")
                    img = Image.open("temp.jpg")
                    st.image(img, caption="MRI Image", use_container_width=True)

                with col2:
                    st.markdown("### 🧬 Prediction Result")

                    if label == 0:
                        st.success("🩺 **No Tumor Detected**")
                        st.markdown(f"**Confidence:** {confidence*100:.2f}%")
                    else:
                        st.error("⚠️ **Tumor Detected**")
                        st.markdown(f"**Confidence:** {confidence*100:.2f}%")

                st.markdown("---")

                # ---------------- CONDITIONAL GRAD-CAM ---------------- #
                if label == 1:  # Only show heatmap if tumor detected
                    col3, col4 = st.columns([1.5, 1.5])
                    with col3:
                        st.markdown("### 🔍 Grad-CAM Heatmap (Model Focus Area)")
                        gradcam_path = generate_gradcam("temp.jpg")
                        st.image(gradcam_path, caption="Grad-CAM Visualization", use_container_width=True)

                    with col4:
                        st.markdown("""
                        **Interpretation:**  
                        The **red and yellow** regions indicate where the model focused its attention 
                        while predicting the presence of a tumor.  
                        This helps doctors and researchers **understand model reasoning** and visually 
                        verify predictions using explainable AI.
                        """)
                else:
                    st.info("✅ No heatmap generated — the model detected **no tumor region** in the MRI image.")

        else:
            st.warning("Please upload an MRI image first!")

Head.contributions()
