"""
Smart Loan Risk Analyzer - Streamlit App
Author: (Michael Boakye)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from io import BytesIO
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# ---------------------------
# Page config & basic theme
# ---------------------------
st.set_page_config(page_title="Smart Loan Risk Analyzer", layout="wide")
st.markdown(
    """
    <style>
    .header-bar {
      background: linear-gradient(90deg, #1F4E79, #D4AF37);
      padding: 12px 18px;
      border-radius: 8px;
      color: white;
      margin-bottom: 12px;
      box-shadow: 0 3px 12px rgba(0,0,0,0.12);
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown('<div class="header-bar"><h2 style="margin:0">💡 Smart Loan Risk Analyzer</h2></div>', unsafe_allow_html=True)

# ---------------------------
# Helper functions
# ---------------------------
def safe_train_test_split(X, y, test_size=0.2):
    """
    Attempt stratified split; if not possible (insufficient class samples),
    fall back to a normal split and warn the user.
    """
    try:
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    except ValueError:
        st.warning("⚠️ Stratified split failed due to insufficient samples per class — using non-stratified split.")
        return train_test_split(X, y, test_size=test_size, random_state=42)

def basic_fill(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "Missing")
    return df

def train_model(model_type: str, X_train: pd.DataFrame, y_train: pd.Series):
    """Train a model (no caching here to keep logic simple). class_weight='balanced' used where available."""
    if model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier(class_weight="balanced", random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    return model

def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    roc_auc = roc_auc_score(y_test, probs) if probs is not None else np.nan
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, probs) if probs is not None else (None, None, None)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc, "cm": cm, "fpr": fpr, "tpr": tpr}

def top_features_df(model, X_train_cols, top_n=10):
    """Return top features DataFrame using coef_ or feature_importances_."""
    try:
        if hasattr(model, "coef_"):
            imp = np.abs(model.coef_[0])
        elif hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        else:
            return None
        df = pd.DataFrame({"Feature": X_train_cols, "Importance": imp})
        df = df.sort_values("Importance", ascending=False).head(top_n).reset_index(drop=True)
        return df
    except Exception:
        return None

def confusion_plot(cm, labels=None, title="Confusion Matrix"):
    z = cm[::-1]
    if labels is None:
        labels = list(range(cm.shape[0]))
    fig = ff.create_annotated_heatmap(z, x=labels, y=list(reversed(labels)), colorscale='Blues')
    fig.update_layout(title=title, width=520, height=420)
    return fig

def model_to_bytes(model):
    bio = BytesIO()
    joblib.dump(model, bio)
    bio.seek(0)
    return bio

# ---------------------------
# App navigation
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Home", "EDA", "Data Processing", "Visualization", "Modeling", "Model Evaluation", "Prediction"])

# ---------------------------
# HOME
# ---------------------------
if page == "Home":
    st.header("Welcome to Smart Loan Risk Analyzer")
    st.markdown("""
    This app helps you analyze loan data, train classification models to predict loan default,
    compare model performance, and make predictions with the best model.

    **Workflow**: Upload data → EDA → Data Processing → Visualization → Modeling → Evaluation → Prediction
    """)
    st.markdown("**Dataset example:** https://www.kaggle.com/datasets/yasserh/loan-default-dataset")
    uploaded_file = st.file_uploader("Upload CSV dataset (optional)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['uploaded_data'] = df
        st.session_state['original_df'] = df.copy()
        st.success("Dataset uploaded and stored in session.")
        st.dataframe(df.head())

# ---------------------------
# EDA
# ---------------------------
elif page == "EDA":
    st.header("Exploratory Data Analysis")
    if 'uploaded_data' not in st.session_state:
        st.warning("Upload a dataset on the Home page first.")
    else:
        df = st.session_state['uploaded_data']
        st.subheader("Snapshot")
        st.dataframe(df.head(10))
        st.write("Shape:", df.shape)
        st.subheader("Missing values")
        miss = df.isnull().sum().sort_values(ascending=False)
        st.dataframe(miss[miss > 0].to_frame("missing_count"))
        st.subheader("Numeric summary")
        st.dataframe(df.describe().T)

# ---------------------------
# Data Processing
# ---------------------------
elif page == "Data Processing":
    st.header("Data Processing")
    if 'uploaded_data' not in st.session_state:
        st.warning("Upload a dataset on Home first.")
    else:
        df = st.session_state['uploaded_data']
        if st.button("Auto fill missing (median/mode)"):
            df_clean = basic_fill(df)
            st.session_state['uploaded_data'] = df_clean
            st.session_state['original_df'] = df_clean.copy()
            st.success("Missing values filled and saved to session.")
        if st.checkbox("Show dtypes"):
            st.dataframe(df.dtypes)
        if st.checkbox("Show sample"):
            st.dataframe(df.sample(min(len(df), 100)))

# ---------------------------
# Visualization
# ---------------------------
elif page == "Visualization":
    st.header("Visualization")
    if 'uploaded_data' not in st.session_state:
        st.warning("Upload dataset on Home.")
    else:
        df = st.session_state['uploaded_data']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        st.markdown("**Included visualizations:** default rate by gender, loan_type, education; correlation heatmap; numeric distributions.")
        viz = st.selectbox("Choose visualization", ["Default rate by categorical", "Correlation Heatmap", "Numeric Distribution", "Category Counts", "Pie Chart"])
        if viz == "Default rate by categorical":
            if not cat_cols:
                st.warning("No categorical columns found.")
            else:
                col = st.selectbox("Select categorical column", cat_cols)
                tmp = df[[col,]].copy()
                # try a 'default' column if present else ask user
                if 'default' in df.columns:
                    tmp['default'] = df['default']
                else:
                    st.info("No 'default' column found; select your binary target in Modeling page.")
                if 'default' in tmp.columns:
                    rate = tmp.groupby(col)['default'].mean().reset_index().sort_values('default', ascending=False)
                    fig = px.bar(rate, x=col, y='default', title=f"Default rate by {col}", labels={'default':'Default Rate'})
                    st.plotly_chart(fig, use_container_width=True)
        elif viz == "Correlation Heatmap":
            if len(numeric_cols) < 2:
                st.warning("Need at least two numeric columns.")
            else:
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)
        elif viz == "Numeric Distribution":
            if not numeric_cols:
                st.warning("No numeric columns.")
            else:
                col = st.selectbox("Numeric column", numeric_cols)
                fig = px.histogram(df.sample(min(len(df), 20000)), x=col, nbins=40, title=f"Distribution: {col}", color_discrete_sequence=["#1F4E79"])
                st.plotly_chart(fig, use_container_width=True)
        elif viz == "Category Counts":
            if not cat_cols:
                st.warning("No categorical columns.")
            else:
                col = st.selectbox("Categorical column", cat_cols)
                vc = df[col].value_counts().reset_index()
                vc.columns = [col, "count"]
                fig = px.bar(vc, x=col, y="count", title=f"Counts of {col}", color_discrete_sequence=["#D4AF37"])
                st.plotly_chart(fig, use_container_width=True)
        else:
            if not cat_cols:
                st.warning("No categorical columns.")
            else:
                col = st.selectbox("Categorical column", cat_cols)
                vc = df[col].value_counts().reset_index()
                vc.columns = [col, "count"]
                fig = px.pie(vc, names=col, values="count", title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Modeling
# ---------------------------
elif page == "Modeling":
    st.header("Modeling")
    if 'uploaded_data' not in st.session_state:
        st.warning("Upload data on Home first.")
    else:
        df = st.session_state['uploaded_data']
        default_hint = 'default' if 'default' in df.columns else None
        target_col = st.selectbox("Select target column (binary)", df.columns, index=df.columns.get_loc(default_hint) if default_hint else 0)
        feature_cols = st.multiselect("Select feature columns for training", [c for c in df.columns if c != target_col], default=[c for c in df.columns if c != target_col])

        if feature_cols:
            X = df[feature_cols].copy()
            y = df[target_col].copy()

            st.markdown("**Preprocessing & train/test split**")
            test_pct = st.slider("Test size (%)", 10, 50, 20) / 100.0

            # Fill and encode
            X = basic_fill(X)
            cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
            if cat_cols:
                X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

            # split safely
            X_train, X_test, y_train, y_test = safe_train_test_split(X, y, test_size=test_pct)

            # scale numeric
            num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            scaler = StandardScaler()
            if num_cols:
                X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
                X_test[num_cols] = scaler.transform(X_test[num_cols])

            st.markdown("**Choose model**")
            model_choice = st.selectbox("Model", ["Logistic Regression", "Decision Tree", "Random Forest"])
            if st.button("Train model"):
                run_id = f"{model_choice.replace(' ','_')}_{int(time.time())}"
                with st.spinner("Training model..."):
                    model = train_model(model_choice, X_train, y_train)
                    metrics = compute_metrics(model, X_test, y_test)

                    # initialize session storage
                    if 'model_runs' not in st.session_state:
                        st.session_state['model_runs'] = []
                        st.session_state['models'] = {}

                    st.session_state['model_runs'].append({
                        "id": run_id,
                        "Model": model_choice,
                        "Accuracy": metrics['accuracy'],
                        "Precision": metrics['precision'],
                        "Recall": metrics['recall'],
                        "F1 Score": metrics['f1'],
                        "ROC AUC": metrics['roc_auc']
                    })

                    # Store model artifact and metadata
                    st.session_state['models'][run_id] = {
                        "model": model,
                        "model_type": model_choice,
                        "X_train_columns": X_train.columns.tolist(),
                        "scaler": scaler
                    }

                    # store ROC/conf matrices for evaluation
                    if 'roc_data' not in st.session_state:
                        st.session_state['roc_data'] = []
                        st.session_state['conf_matrices'] = []
                    st.session_state['roc_data'].append({"id": run_id, "model": model_choice, "fpr": metrics['fpr'], "tpr": metrics['tpr'], "roc_auc": metrics['roc_auc']})
                    st.session_state['conf_matrices'].append({"id": run_id, "model": model_choice, "matrix": metrics['cm'], "labels": np.unique(y_test).tolist()})

                    st.session_state['selected_features'] = X_train.columns.tolist()
                    st.success(f"Trained {model_choice} — ROC AUC: {metrics['roc_auc']:.3f}")

                    # Display metrics / visuals
                    st.markdown("**Metrics**")
                    st.table(pd.DataFrame([{
                        "Accuracy": metrics['accuracy'],
                        "Precision": metrics['precision'],
                        "Recall": metrics['recall'],
                        "F1 Score": metrics['f1'],
                        "ROC AUC": metrics['roc_auc']
                    }]).T)

                    # Confusion matrix (unique key)
                    st.plotly_chart(confusion_plot(metrics['cm'], labels=np.unique(y_test).tolist(), title=f"{model_choice} Confusion Matrix"), use_container_width=False, key=f"cm_train_{run_id}")

                    # ROC
                    if metrics['fpr'] is not None:
                        fig, ax = plt.subplots(figsize=(6,4))
                        ax.plot(metrics['fpr'], metrics['tpr'], label=f"AUC={metrics['roc_auc']:.3f}", color="#1F4E79")
                        ax.plot([0,1],[0,1],'r--')
                        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
                        st.pyplot(fig)

                    # Top features
                    fi = top_features_df(model, X_train.columns.tolist(), top_n=10)
                    if fi is not None:
                        st.markdown("**Top 10 Features (this run)**")
                        st.dataframe(fi.style.background_gradient(cmap="YlOrRd", subset=["Importance"]))
                        fig_bar = go.Figure(go.Bar(x=fi["Importance"][::-1], y=fi["Feature"][::-1], orientation='h', marker_color="#1F4E79"))
                        fig_bar.update_layout(title="Top 10 Feature Importances", height=380)
                        st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------
# Model Evaluation
# ---------------------------
elif page == "Model Evaluation":
    st.header("Model Evaluation & Comparison")
    if 'model_runs' not in st.session_state or len(st.session_state['model_runs']) == 0:
        st.warning("You need to train at least one model in Modeling page.")
    else:
        results_df = pd.DataFrame(st.session_state['model_runs'])
        metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
        results_df[metric_cols] = results_df[metric_cols].round(3)
        st.subheader("Performance summary")
        st.dataframe(results_df.style.background_gradient(cmap="YlGnBu", subset=metric_cols))

        best_idx = results_df['ROC AUC'].idxmax()
        best_run = results_df.loc[best_idx]
        st.success(f"Best run: {best_run['Model']} (id={best_run['id']}) — ROC AUC {best_run['ROC AUC']:.3f}")

        # Bar comparison (unique key)
        st.markdown("### Metric Comparison")
        long_df = results_df.melt(id_vars=['id','Model'], value_vars=metric_cols, var_name='Metric', value_name='Score')
        fig_bar = px.bar(long_df, x='Model', y='Score', color='Metric', barmode='group', text='Score', title="Model Metrics Comparison")
        fig_bar.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True, key="metric_bar_chart")

        # ROC comparison
        st.markdown("### ROC Curves")
        fig, ax = plt.subplots(figsize=(8,5))
        for r in st.session_state['roc_data']:
            if r['fpr'] is not None:
                ax.plot(r['fpr'], r['tpr'], label=f"{r['model']} (id={r['id']}) AUC={r['roc_auc']:.2f}")
        ax.plot([0,1],[0,1],'r--'); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.legend()
        st.pyplot(fig)

        # Confusion matrices with unique keys
        st.markdown("### Confusion Matrices")
        for cm_entry in st.session_state['conf_matrices']:
            st.markdown(f"**{cm_entry['model']} (id={cm_entry['id']})**")
            st.plotly_chart(confusion_plot(cm_entry['matrix'], labels=cm_entry['labels'], title=f"{cm_entry['model']} Confusion Matrix"),
                            use_container_width=False, key=f"cm_eval_{cm_entry['id']}")

        # Top 10 features for each run in tabs
        st.markdown("### Top 10 Features per Run")
        run_ids = [r['id'] for r in st.session_state['model_runs']]
        tabs = st.tabs(run_ids)
        for tab, run_id in zip(tabs, run_ids):
            with tab:
                meta = st.session_state['models'].get(run_id)
                if not meta:
                    st.info("Model artifact missing.")
                else:
                    model = meta['model']
                    cols = meta['X_train_columns']
                    # create dummy X_train just to let top_features_df read column names
                    dummy = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
                    fi = top_features_df(model, cols, top_n=10)
                    # If top_features_df needed real coef/imp mapping, we already compute on model and columns
                    if fi is None:
                        st.info("No feature importances/coefficients available for this model.")
                    else:
                        st.dataframe(fi.style.background_gradient(cmap="YlGnBu", subset=["Importance"]))
                        fig_bar = go.Figure(go.Bar(x=fi["Importance"][::-1], y=fi["Feature"][::-1], orientation='h', marker_color="#1F4E79"))
                        fig_bar.update_layout(title=f"Top 10 Features — {run_id}", height=420)
                        st.plotly_chart(fig_bar, use_container_width=True)

        # export evaluation CSV
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download evaluation results (CSV)", csv, file_name="model_evaluation_results.csv", mime="text/csv")

        # download best model
        if st.button("Download best model (.pkl)"):
            best_id = best_run['id']
            meta = st.session_state['models'].get(best_id)
            if meta:
                bio = model_to_bytes(meta['model'])
                st.download_button("Click to download", data=bio, file_name=f"{best_run['Model']}_{best_id}.pkl", mime="application/octet-stream")

# ---------------------- PAGE 5: PREDICTION ----------------------
elif page == "🔮 Prediction":

    st.title("🔮 Loan Default Prediction")
    st.markdown(
        """
        Upload your dataset or manually input customer details to predict **loan default risk**.
        The model uses only the selected training features and automatically aligns your data before prediction.
        """
    )

    prediction_mode = st.radio(
        "Choose input method:",
        ["📁 Upload CSV file", "🧍 Manual Input"],
        horizontal=True,
    )

    # --- Load saved training columns and scaler ---
    if "train_cols" not in st.session_state or "scaler" not in st.session_state:
        st.warning("⚠️ Please complete the Data Processing and Modeling steps first before making predictions.")
    else:
        train_cols = st.session_state["train_cols"]
        scaler = st.session_state["scaler"]
        best_model = st.session_state["best_model"]

        # --- 📁 CSV UPLOAD MODE ---
        if prediction_mode == "📁 Upload CSV file":
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

            if uploaded_file:
                try:
                    df_in = pd.read_csv(uploaded_file)
                    st.subheader("📊 Uploaded Data Preview")
                    st.dataframe(df_in.head())

                    # --- Encoding & alignment with training columns ---
                    df_enc = pd.get_dummies(df_in)
                    df_enc = df_enc.reindex(columns=train_cols, fill_value=0)

                    # --- Safe numeric scaling ---
                    numeric_cols = [
                        c for c in train_cols
                        if c in df_enc.columns and df_enc[c].dtype in [np.float64, np.int64, np.float32, np.int32]
                    ]
                    try:
                        if numeric_cols:
                            df_enc[numeric_cols] = scaler.transform(df_enc[numeric_cols])
                    except Exception as e:
                        st.error(f"Scaling failed: {e}")

                    # --- Make predictions ---
                    preds = best_model.predict(df_enc)
                    probs = (
                        best_model.predict_proba(df_enc)[:, 1]
                        if hasattr(best_model, "predict_proba")
                        else None
                    )

                    # --- Show predictions ---
                    df_in["Predicted_Default"] = preds
                    if probs is not None:
                        df_in["Default_Probability"] = probs

                    st.success("✅ Predictions successfully generated!")
                    st.dataframe(df_in.head(10))

                    # --- Download predictions ---
                    csv = df_in.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Predictions",
                        csv,
                        "loan_default_predictions.csv",
                        "text/csv",
                        use_container_width=True,
                    )

                except Exception as e:
                    st.error(f"❌ Error processing file: {e}")

        # --- 🧍 MANUAL INPUT MODE ---
        else:
            st.subheader("🧾 Enter Details Manually")

            # Example input fields
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", 18, 100, 30)
            income = st.number_input("Annual Income ($)", 1000, 500000, 50000)
            loan_amount = st.number_input("Loan Amount ($)", 1000, 500000, 15000)
            credit_worthiness = st.selectbox("Credit Worthiness", ["l1", "l2"])
            region = st.selectbox("Region", ["central", "north-east", "north-west", "south-east", "south-west"])
            employment_type = st.selectbox("Employment Type", ["Government", "Private", "Self-employed"])
            experience = st.number_input("Years of Experience", 0, 50, 5)

            if st.button("🔍 Predict Default Risk"):
                user_df = pd.DataFrame(
                    [
                        {
                            "Gender": gender,
                            "Age": age,
                            "Income": income,
                            "LoanAmount": loan_amount,
                            "Credit_Worthiness": credit_worthiness,
                            "Region": region,
                            "Employment_Type": employment_type,
                            "Experience": experience,
                        }
                    ]
                )

                # --- Align & encode manually entered data ---
                user_enc = pd.get_dummies(user_df)
                user_enc = user_enc.reindex(columns=train_cols, fill_value=0)

                # --- Safe numeric scaling ---
                numeric_cols = [
                    c for c in train_cols
                    if c in user_enc.columns and user_enc[c].dtype in [np.float64, np.int64, np.float32, np.int32]
                ]
                try:
                    if numeric_cols:
                        user_enc[numeric_cols] = scaler.transform(user_enc[numeric_cols])
                except Exception as e:
                    st.error(f"Scaling failed: {e}")

                # --- Predict using best model ---
                try:
                    prediction = best_model.predict(user_enc)[0]
                    probability = (
                        best_model.predict_proba(user_enc)[0, 1]
                        if hasattr(best_model, "predict_proba")
                        else None
                    )

                    st.subheader("📈 Prediction Result")
                    if prediction == 1:
                        st.error("⚠️ This applicant is **likely to default**.")
                    else:
                        st.success("✅ This applicant is **unlikely to default**.")

                    if probability is not None:
                        st.info(f"**Default Probability:** {probability:.2%}")

                    st.dataframe(user_df)

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
