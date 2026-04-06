import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Learning Preference Survey",
    page_icon="🎓",
    layout="wide"
)
sns.set_theme(style="whitegrid")

# ==========================================
# 2. DATA PERSISTENCE LAYER
# ==========================================
DATA_FILE = "survey_results.csv"

def load_existing_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        # Create 20 rows of realistic "starter" data so AI works immediately
        np.random.seed(42)
        seed_data = {
            'Name': [f"Student_{i}" for i in range(20)],
            'Email': [f"test{i}@example.com" for i in range(20)],
            'Age': np.random.randint(18, 25, 20),
            'Preferred_Mode': np.random.choice(['Online', 'Offline', 'Hybrid'], 20),
            'Avg_Daily_Study_Hours': np.random.uniform(2, 10, 20).round(1),
            'Engagement_Level': np.random.randint(4, 11, 20),
            'Internet_Issue': np.random.choice(['Yes', 'No'], 20),
            'Understanding_Rating': np.random.randint(3, 11, 20)
        }
        df_seed = pd.DataFrame(seed_data)
        df_seed.to_csv(DATA_FILE, index=False)
        return df_seed

def save_new_response(data_dict):
    df = load_existing_data()
    new_row = pd.DataFrame([data_dict])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    return df

# ==========================================
# 3. USER INTERFACE (SIDEBAR)
# ==========================================
st.sidebar.title("📝 Student Survey")
with st.sidebar.form("survey_form", clear_on_submit=False): # Changed to False to keep data visible
    user_name = st.text_input("Full Name")
    user_email = st.text_input("Email Address")
    age = st.number_input("What is your age?", min_value=10, max_value=100, value=20)
    mode = st.selectbox("Preferred Learning Mode", options=["Online", "Offline", "Hybrid"])
    hours = st.slider("Average daily study hours", 0.0, 15.0, 4.0, step=0.5)
    engagement = st.select_slider("Engagement (1-10)", options=list(range(1, 11)), value=5)
    internet = st.radio("Do you face frequent internet issues?", ["Yes", "No"])
    understanding = st.slider("Rate your understanding (1-10)", 1, 10, 5)
    submit_button = st.form_submit_button("Submit & Predict")

# ==========================================
# 4. MAIN DASHBOARD LOGIC
# ==========================================
st.title("🎓 Online vs. Offline Learning Analysis")
df = load_existing_data()

if submit_button:
    if user_name and user_email:
        current_response = {
            'Name': user_name, 'Email': user_email, 'Age': age,
            'Preferred_Mode': mode, 'Avg_Daily_Study_Hours': hours,
            'Engagement_Level': engagement, 'Internet_Issue': internet,
            'Understanding_Rating': understanding
        }
        df = save_new_response(current_response)
        st.success(f"Response recorded for {user_name}!")
    else:
        st.error("Please provide both your name and email.")

# ==========================================
# 5. AI PREDICTOR (NOW PROMINENT)
# ==========================================
st.divider()
st.header("🔮 AI Prediction Result")

if len(df) > 5:
    # Train the model on all current data
    le_mode = LabelEncoder()
    le_internet = LabelEncoder()
    train_df = df.copy()
    train_df['Mode_N'] = le_mode.fit_transform(train_df['Preferred_Mode'])
    train_df['Internet_N'] = le_internet.fit_transform(train_df['Internet_Issue'])

    X = train_df[['Age', 'Avg_Daily_Study_Hours', 'Engagement_Level', 'Mode_N', 'Internet_N']]
    y = train_df['Understanding_Rating']

    model = LinearRegression()
    model.fit(X, y)

    # Automatically predict using the sidebar values immediately
    p_mode_n = le_mode.transform([mode])[0]
    p_internet_n = le_internet.transform([internet])[0]
    
    prediction = model.predict([[age, hours, engagement, p_mode_n, p_internet_n]])
    
    # Large Display for the Prediction
    st.metric("Predicted Understanding Score", f"{prediction[0]:.2f} / 10")
    st.progress(min(max(prediction[0]/10, 0.0), 1.0))
    st.write("This score is calculated based on your inputs in the sidebar compared to current trends.")

# ==========================================
# 6. DATA VISUALIZATION (ALWAYS VISIBLE)
# ==========================================
st.divider()
st.subheader("📊 Community Trends")
col1, col2, col3 = st.columns(3)
col1.metric("Total Responses", len(df))
col2.metric("Avg Engagement", f"{df['Engagement_Level'].mean():.1f}/10")
col3.metric("Avg Study Hours", f"{df['Avg_Daily_Study_Hours'].mean():.1f} hrs")

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='Preferred_Mode', palette='viridis', ax=ax1)
    st.pyplot(fig1)

with chart_col2:
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x='Preferred_Mode', y='Understanding_Rating', palette='Set2', ax=ax2)
    st.pyplot(fig2)

with st.expander("View Raw Data Table"):
    st.dataframe(df, use_container_width=True)
