import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# 1. Load and preprocess data
def load_data(filepath):
    df = pd.read_csv(filepath)
    # Basic preprocessing: drop rows with missing target, fillna for features
    df = df.dropna(subset=[df.columns[-1]])
    df = df.fillna(df.median(numeric_only=True))
    # Encode categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

# 2. Feature engineering example
def feature_engineering(df):
    if 'debts' in df.columns and 'income' in df.columns:
        df['debt_income_ratio'] = df['debts'] / (df['income'] + 1e-6)
    return df

# 3. Split dataset
def split_data(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Train and evaluate model
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    metrics = {
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    }
    return metrics

# Streamlit App
st.set_page_config(page_title='Credit Scoring Model', layout='centered')
st.title('Credit Scoring Model')
st.write('The default dataset (credit_score.csv) is loaded automatically.')

# Load and preprocess dataset
data_load_state = st.info('Loading and preprocessing dataset...')
try:
    df = load_data('credit_score.csv')
    df = feature_engineering(df)
    X_train, X_test, y_train, y_test = split_data(df)
    data_load_state.success('Dataset loaded and preprocessed!')
except Exception as e:
    st.error(f'Failed to load dataset: {e}')
    st.stop()

# Show a preview of the data
if st.checkbox('Show raw data'):
    st.subheader('Raw Data')
    st.write(df.head())

# Model selection
st.subheader('Select Classifier')
model_name = st.selectbox('Choose a model:', ['Logistic Regression', 'Decision Tree', 'Random Forest'])
MODELS = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Train and evaluate
if st.button('Train Model'):
    with st.spinner('Training and evaluating model...'):
        try:
            metrics = train_and_evaluate(MODELS[model_name], X_train, X_test, y_train, y_test)
            st.success('Model trained and evaluated!')
            st.subheader('Evaluation Metrics')
            st.table(pd.DataFrame(metrics, index=['Score']).T)
        except Exception as e:
            st.error(f'Failed to train/evaluate model: {e}') 