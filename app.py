import streamlit as st
import pandas as pd
from churn_pipeline import ChurnPipeline

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# LOAD PIPELINE
# --------------------------------------------------
@st.cache_resource
def load_pipeline():
    return ChurnPipeline()

pipeline = load_pipeline()

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>📉 Customer Churn Prediction System</h1>
    <p style='text-align: center; font-size:18px;'>
    Predict whether a customer is likely to leave the bank using Deep Learning
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------
st.sidebar.header("🧾 Customer Information")

credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)
age = st.sidebar.slider("Age", 18, 100, 40)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 3)
balance = st.sidebar.number_input("Account Balance", 0.0, 300000.0, 60000.0)
salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_card = st.sidebar.selectbox("Has Credit Card", [0, 1])
is_active = st.sidebar.selectbox("Is Active Member", [0, 1])
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])

# Geography encoding
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

# Gender encoding (already preprocessed logic)
gender_val = 1 if gender == "Male" else 0

# --------------------------------------------------
# MAIN CONTENT
# --------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Prediction Overview")
    st.write(
        """
        This system uses a **Deep Learning ANN model** trained on customer data
        to predict **churn probability**.
        
        - Model Accuracy: **~86%**
        - Dataset: Bank Customer Churn
        - Algorithm: Artificial Neural Network
        """
    )

with col2:
    st.subheader("📌 Model Details")
    st.info(
        """
        **Model Type:** ANN  
        **Loss:** Binary Crossentropy  
        **Optimizer:** Adam  
        **Deployment:** Streamlit Cloud
        """
    )

st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button("🚀 Predict Churn", use_container_width=True):

    user_input = {
        "CreditScore": credit_score,
        "Gender": gender_val,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": salary,
        "Geography_Germany": geo_germany,
        "Geography_Spain": geo_spain
    }

    result = pipeline.predict(user_input)

    st.markdown("<br>", unsafe_allow_html=True)

    if "CHURN" in result:
        st.error("❌ High Risk: Customer is likely to CHURN")
    else:
        st.success("✅ Low Risk: Customer is NOT likely to churn")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:14px;'>
    Built with ❤️ using Deep Learning & Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
