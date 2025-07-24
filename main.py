import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import tkinter as tk
from tkinter import messagebox, ttk

# Global variables
DATA = None
X_train, X_test, y_train, y_test = None, None, None, None
MODELS = {
    'Logistic Regression': LogisticRegression,
    'Decision Tree': DecisionTreeClassifier,
    'Random Forest': RandomForestClassifier
}

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
def train_and_evaluate(model_name):
    model = MODELS[model_name]()
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

# 5. GUI
class CreditScoringApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Credit Scoring Model')
        self.root.geometry('420x350')
        self.create_widgets()
        self.load_default_dataset()

    def create_widgets(self):
        title = tk.Label(self.root, text='Credit Scoring Model', font=('Arial', 18, 'bold'))
        title.pack(pady=(15, 5))

        instructions = tk.Label(
            self.root,
            text='The default dataset (credit_score.csv) is loaded automatically.',
            font=('Arial', 10), justify='center')
        instructions.pack(pady=(0, 10))

        model_frame = tk.Frame(self.root)
        model_frame.pack(pady=10)
        model_label = tk.Label(model_frame, text='Select Classifier:', font=('Arial', 11))
        model_label.pack(side='left', padx=(0, 8))
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(model_frame, textvariable=self.model_var, values=list(MODELS.keys()), state='readonly', width=20)
        self.model_dropdown.set('Logistic Regression')
        self.model_dropdown.pack(side='left')

        self.train_btn = tk.Button(self.root, text='Train Model', font=('Arial', 11), command=self.train_model, width=20, bg='#4CAF50', fg='white')
        self.train_btn.pack(pady=10)

        self.result_label = tk.Label(self.root, text='', justify='left', font=('Consolas', 11), bg='#f0f0f0', anchor='w', width=40, height=6, relief='groove')
        self.result_label.pack(pady=10)

    def load_default_dataset(self):
        global DATA, X_train, X_test, y_train, y_test
        try:
            df = load_data('credit_score.csv')
            df = feature_engineering(df)
            X_train, X_test, y_train, y_test = split_data(df)
            DATA = df
            messagebox.showinfo('Dataset Loaded', 'Default dataset (credit_score.csv) loaded and preprocessed!')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load default dataset: {e}')

    def train_model(self):
        if DATA is None or X_train is None:
            messagebox.showwarning('Warning', 'Dataset not loaded!')
            return
        model_name = self.model_var.get()
        try:
            metrics = train_and_evaluate(model_name)
            result_text = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
            self.result_label.config(text=result_text)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to train/evaluate model: {e}')

if __name__ == '__main__':
    root = tk.Tk()
    app = CreditScoringApp(root)
    root.mainloop() 