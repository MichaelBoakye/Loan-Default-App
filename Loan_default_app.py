"""
Loan_default_app.py
Professional Loan Default Prediction Streamlit App
Author: (your name)
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
# Page config & styles
# ---------------------------
st.set_page_config(page_title="Loan Default Prediction App", layout="wide")


# ---------------------------
# Helpers & Cached functions
# ---------------------------

@st.cache_data(show_spinner=False)
def load_csv_from_file(file):
    """Load CSV into DataFrame (cached)."""
    return pd.read_csv(file)

@st.cache_data(show_spinner=False)
def preprocess_dataframe(df, selected_features=None):
    """
    Basic preprocessing:
     - Fill missing values (median for numeric, mode for categorical)
     - Optionally restrict to selected_features (list of names)
     - Returns processed df and list of numeric columns
    """
    df = df.copy()
    # basic fills
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "Missing")

    if selected_features is not None:
        # keep only selected features that exist
        keep = [c for c in selected_features if c in df.columns]
        df = df[keep].copy()

    return df

@st.cache_resource(show_spinner=False)
def train_model_cached(model_type, X_train, y_train, random_state=42):
    """Train and return model object (cached to avoid re-training if inputs unchanged)."""
    if model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=random_state)
    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier(random_state=random_state)
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=random_state)

    model.fit(X_train, y_train)
    return model

def compute_metrics(model, X_test, y_test):
    """Compute common classification metrics and confusion matrix + roc data"""
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    roc_auc = roc_auc_score(y_test, probs) if probs is not None else np.nan
    cm = confusion_matrix(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, probs) if probs is not None else (None, None, None)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr
    }

def get_top_features(model, X_train, top_n=10):
    """Return DataFrame with top_n features: uses coef_ for LR, feature_importances_ for trees."""
    try:
        if hasattr(model, "coef_"):
            imp = np.abs(model.coef_[0])
        elif hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        else:
            return None

        fi = pd.DataFrame({"Feature": X_train.columns, "Importance": imp})
        fi = fi.sort_values("Importance", ascending=False).head(top_n).reset_index(drop=True)
        return fi
    except Exception:
        return None

def make_confusion_plot(cm, labels=None, title="Confusion Matrix"):
    z = cm[::-1]  # invert for annotated heatmap display
    if labels is None:
        labels = list(range(cm.shape[0]))
    fig = ff.create_annotated_heatmap(z, x=labels, y=list(reversed(labels)), colorscale='Blues')
    fig.update_layout(title=title, width=520, height=420)
    return fig

def model_to_bytes(model):
    """Serialize model to bytes for download."""
    bio = BytesIO()
    joblib.dump(model, bio)
    bio.seek(0)
    return bio

# ---------------------------
# Sidebar navigation
# ---------------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("", ["Home", "Data Overview (EDA)", "Visualization", "Modeling", "Model Evaluation", "Prediction"])

# ---------------------------
# HOME
# ---------------------------
if page == "Home":
    # --- CUSTOM STYLES AND HEADER ---
    st.markdown("""
        <style>
        /* === GENERAL PAGE STYLING === */
        body {
            background-color: #fdfdfd;
            font-family: "Helvetica Neue", sans-serif;
        }

        /* === HEADER BAR === */
        .header-bar {
            background: linear-gradient(90deg, #1F4E79, #D4AF37);
            padding: 18px 30px;
            border-radius: 10px;
            margin-bottom: 25px;
            color: white;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 12px rgba(0,0,0,0.15);
        }

        .header-title {
            font-size: 28px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }

        .header-icon {
            font-size: 35px;
            margin-right: 10px;
        }

        /* === INTRO BOX === */
        .intro-box {
            background-color: #fefcf6;  /* soft ivory */
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
            color: #1A1A1A;
        }
        .intro-box h1 {
            color: #1F4E79;  /* deep blue */
            font-weight: 700;
            margin-bottom: 10px;
        }
        .intro-box h3 {
            color: #D4AF37;  /* gold accent */
            font-weight: 600;
            margin-top: 25px;
        }
        .intro-box ul {
            line-height: 1.7;
        }
        .intro-box li {
            margin-bottom: 6px;
        }
        </style>

        <!-- === HEADER BAR === -->
        <div class="header-bar">
            <div style="display: flex; align-items: center;">
                <span class="header-icon">💼</span>
                <span class="header-title">Loan Default Prediction App</span>
            </div>
            <div>
                <span style="font-size:16px; font-weight:500;">Empowering Smarter Credit Decisions</span>
            </div>
        </div>

        <!-- === INTRO SECTION === -->
        <div class="intro-box">

        <h1>Welcome!</h1>

        <p>The <b>Loan Default Prediction App</b> is a modern analytics solution that helps financial institutions assess and manage credit risk efficiently.</p>

        <p>By applying <b>machine learning algorithms</b>, the app predicts the likelihood of borrower default based on financial and demographic information. It provides a simple yet powerful interface for exploring datasets, training models, evaluating performance, and generating real-time predictions.</p>

        <h3>🚀 Key Features</h3>
        <ul>
        <li>📂 <b>Data Upload & Exploration</b> – Import and visualize loan datasets with descriptive statistics and insights.</li>
        <li>⚙️ <b>Model Training & Comparison</b> – Train models such as Logistic Regression, Decision Tree, and Random Forest, and compare their performance.</li>
        <li>📊 <b>Performance Evaluation</b> – Analyze confusion matrices, ROC curves, and model accuracy with clear visualizations.</li>
        <li>🔮 <b>Risk Prediction</b> – Input borrower details and instantly obtain a data-driven default risk prediction.</li>
        <li>🔮 <b>Download data from</b> – https://www.kaggle.com/datasets/yasserh/loan-default-dataset.</li>
        </ul>

        <p style="margin-top: 20px;">Designed to help <b>credit officers</b>, <b>data analysts</b>, and <b>financial managers</b> make informed, data-backed lending decisions that enhance portfolio performance.</p>

        </div>
        """, unsafe_allow_html=True)
    uploaded = st.file_uploader("📁 Upload a CSV (or leave blank to use local sample)", type=["csv"])
    if uploaded:
        df = load_csv_from_file(uploaded)
        st.session_state['uploaded_data'] = df
        st.session_state['original_df'] = df.copy()
        st.success("Dataset loaded into session.")
        st.write("Preview:")
        st.dataframe(df.head())

# ---------------------------
# DATA OVERVIEW (EDA)
# ---------------------------
elif page == "Data Overview (EDA)":
    st.header("📊 Exploratory Data Analysis (EDA)")
    if 'uploaded_data' not in st.session_state:
        st.warning("Please upload data on Home page.")
    else:
        df = st.session_state['uploaded_data']
        st.subheader("Dataset snapshot")
        st.dataframe(df.head(10))
        st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.subheader("Missing values")
        missing = df.isnull().sum().sort_values(ascending=False)
        st.dataframe(missing[missing > 0].to_frame("MissingCount"))

        st.subheader("Describe numeric features")
        st.dataframe(df.describe().T)

        st.markdown("### Quick cleaning tools")
        if st.button("🔧 Fill missing (median numeric, mode categorical)"):
            df_clean = preprocess_dataframe(df)
            st.session_state['uploaded_data'] = df_clean
            st.session_state['original_df'] = df_clean.copy()
            st.success("Missing values filled. Use Visualization & Modeling pages next.")

# ---------------------------
# VISUALIZATION
# ---------------------------
elif page == "Visualization":
    st.header("📈 Visualizations")
    if 'uploaded_data' not in st.session_state:
        st.warning("Please upload data first (Home).")
    else:
        df = st.session_state['uploaded_data']
        st.subheader("Select visualization")
        viz = st.selectbox("Choose chart", ["Histogram", "Box Plot", "Correlation Heatmap", "Category Counts", "Category vs Numeric Mean", "Pie Chart"])

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        if viz == "Histogram":
            if not numeric_cols:
                st.warning("No numeric columns found.")
            else:
                col = st.selectbox("Numeric column", numeric_cols)
                fig = px.histogram(df.sample(min(len(df), 20000)), x=col, nbins=40, title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)

        elif viz == "Box Plot":
            if not numeric_cols:
                st.warning("No numeric columns found.")
            else:
                col = st.selectbox("Numeric column", numeric_cols)
                fig = px.box(df.sample(min(len(df), 20000)), y=col, title=f"Box plot of {col}")
                st.plotly_chart(fig, use_container_width=True)

        elif viz == "Correlation Heatmap":
            if len(numeric_cols) < 2:
                st.warning("Need at least two numeric columns for correlation.")
            else:
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)

        elif viz == "Category Counts":
            if not cat_cols:
                st.warning("No categorical columns found.")
            else:
                col = st.selectbox("Categorical column", cat_cols)
                vc = df[col].value_counts().reset_index()
                vc.columns = [col, "count"]
                fig = px.bar(vc, x=col, y="count", title=f"Counts of {col}", color_discrete_sequence=["#D4AF37"])
                st.plotly_chart(fig, use_container_width=True)

        elif viz == "Category vs Numeric Mean":
            if not cat_cols or not numeric_cols:
                st.warning("Require at least one categorical and one numeric column.")
            else:
                cat = st.selectbox("Categorical", cat_cols)
                num = st.selectbox("Numeric", numeric_cols)
                g = df.groupby(cat)[num].mean().reset_index().sort_values(num, ascending=False)
                fig = px.bar(g, x=cat, y=num, title=f"Average {num} by {cat}", color_discrete_sequence=["#1F4E79"])
                st.plotly_chart(fig, use_container_width=True)

        elif viz == "Pie Chart":
            if not cat_cols:
                st.warning("No categorical columns found.")
            else:
                col = st.selectbox("Categorical column", cat_cols)
                dfc = df[col].value_counts().reset_index()
                dfc.columns = [col, "count"]
                fig = px.pie(dfc, names=col, values="count", title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# MODELING
# ---------------------------
elif page == "Modeling":
    st.header("⚙️ Model Training")
    if 'uploaded_data' not in st.session_state:
        st.warning("Upload dataset first (Home).")
    else:
        df = st.session_state['uploaded_data']

        # choose target & features
        target_col = st.selectbox("Select target column (binary 0/1)", df.columns)
        feature_cols = st.multiselect("Select feature columns (features used for training)", [c for c in df.columns if c != target_col], default=[c for c in df.columns if c != target_col])

        if feature_cols:
            st.markdown("### Split & Preprocess")
            test_size = st.slider("Test set size (%)", 10, 50, 20)

            X = df[feature_cols].copy()
            y = df[target_col].copy()

            # preprocess (fill missing)
            X = preprocess_dataframe(X)

            # one-hot encode categorical
            cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
            if cat_cols:
                X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

            # split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42, stratify=y)

            # scale numeric columns
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            scaler = StandardScaler()
            X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
            X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

            # choose model
            model_choice = st.selectbox("Model", ["Logistic Regression", "Decision Tree", "Random Forest"])
            if st.button("Train Model"):
                with st.spinner("Training..."):
                    model = train_model_cached(model_choice, X_train, y_train)
                    metrics = compute_metrics(model, X_test, y_test)

                    # store trained model and associated objects
                    if 'model_results' not in st.session_state:
                        st.session_state['model_results'] = []
                        st.session_state['roc_data'] = []
                        st.session_state['conf_matrices'] = []
                        st.session_state['trained_models'] = {}

                    st.session_state['model_results'].append({
                        "Model": model_choice,
                        "Accuracy": metrics['accuracy'],
                        "Precision": metrics['precision'],
                        "Recall": metrics['recall'],
                        "F1 Score": metrics['f1'],
                        "ROC AUC": metrics['roc_auc']
                    })
                    st.session_state['roc_data'].append({
                        "model": model_choice,
                        "fpr": metrics['fpr'],
                        "tpr": metrics['tpr'],
                        "roc_auc": metrics['roc_auc']
                    })
                    st.session_state['conf_matrices'].append({
                        "model": model_choice,
                        "matrix": metrics['cm'],
                        "labels": np.unique(y_test).tolist()
                    })

                    # store objects for prediction/eval
                    st.session_state['trained_models'][model_choice] = model
                    st.session_state['scaler'] = scaler
                    st.session_state['selected_features'] = X_train.columns.tolist()
                    st.session_state['X_train'] = X_train  # used for feature names & alignment
                    st.success(f"{model_choice} trained — ROC AUC: {metrics['roc_auc']:.3f}")

                    # show immediate metrics
                    st.markdown("#### Model Metrics")
                    st.write(pd.DataFrame([{
                        "Accuracy": metrics['accuracy'],
                        "Precision": metrics['precision'],
                        "Recall": metrics['recall'],
                        "F1 Score": metrics['f1'],
                        "ROC AUC": metrics['roc_auc']
                    }]).T)

                    # confusion matrix
                    st.markdown("#### Confusion Matrix")
                    fig_cm = make_confusion_plot(metrics['cm'], labels=st.session_state['conf_matrices'][-1]['labels'], title=f"{model_choice} - Confusion Matrix")
                    st.plotly_chart(fig_cm, use_container_width=False)

                    # ROC plot
                    if metrics['fpr'] is not None:
                        fig, ax = plt.subplots(figsize=(6,4))
                        ax.plot(metrics['fpr'], metrics['tpr'], label=f"AUC={metrics['roc_auc']:.3f}", color="#1F4E79")
                        ax.plot([0,1], [0,1], 'r--')
                        ax.set_xlabel("False Positive Rate")
                        ax.set_ylabel("True Positive Rate")
                        ax.set_title("ROC Curve")
                        ax.legend()
                        st.pyplot(fig)

                    # top features
                    fi = get_top_features(model, X_train, top_n=10)
                    if fi is not None:
                        st.markdown("#### Top 10 Features (this model)")
                        st.dataframe(fi.style.background_gradient(cmap="YlOrRd", subset=["Importance"]))
                        fig_bar = go.Figure(go.Bar(x=fi["Importance"][::-1], y=fi["Feature"][::-1], orientation='h', marker_color="#1F4E79"))
                        fig_bar.update_layout(title="Top 10 Feature Importances", height=380)
                        st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------
# MODEL EVALUATION
# ---------------------------
elif page == "Model Evaluation":
    st.header("📈 Model Evaluation & Comparison")

    if 'model_results' not in st.session_state or len(st.session_state['model_results']) == 0:
        st.warning("Train at least one model on the Modeling page.")
    else:
        results_df = pd.DataFrame(st.session_state['model_results'])
        metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
        results_df[metric_cols] = results_df[metric_cols].round(3)
        st.subheader("Performance Summary")
        st.dataframe(results_df.style.background_gradient(cmap="YlGnBu", subset=metric_cols))

        # best model (by ROC AUC)
        best_idx = results_df['ROC AUC'].idxmax()
        best_model_name = results_df.loc[best_idx, 'Model']
        st.success(f"Best model: {best_model_name} (ROC AUC: {results_df.loc[best_idx,'ROC AUC']:.3f})")

        # bar comparison
        st.markdown("### Metric Comparison (bar chart)")
        long = results_df.melt(id_vars=['Model'], value_vars=metric_cols, var_name='Metric', value_name='Score')
        fig = px.bar(long, x='Model', y='Score', color='Metric', barmode='group', text='Score', title="Metrics by Model")
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

        # ROC comparison
        st.markdown("### ROC Curve Comparison")
        fig, ax = plt.subplots(figsize=(8,5))
        for r in st.session_state['roc_data']:
            if r['fpr'] is not None:
                ax.plot(r['fpr'], r['tpr'], label=f"{r['model']} (AUC={r['roc_auc']:.2f})")
        ax.plot([0,1],[0,1],'r--')
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend(); st.pyplot(fig)

        # Confusion matrices
        st.markdown("### Confusion Matrices")
        for i, res in enumerate(st.session_state['conf_matrices']):
            st.markdown(f"**{res['model']}**")
            fig_cm = make_confusion_plot(res['matrix'], labels=res['labels'], title=f"{res['model']} Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=False, key=f"cm_eval_{i}")

        # Feature importance comparison (tabs)
        st.markdown("### 🧠 Top 10 Features per Model")
        model_names = [r['Model'] for r in st.session_state['model_results']]
        tabs = st.tabs(model_names)
        for tab, m in zip(tabs, model_names):
            with tab:
                try:
                    model = st.session_state['trained_models'].get(m)
                    X_train = st.session_state.get('X_train')
                    fi = get_top_features(model, X_train, top_n=10)
                    if fi is None:
                        st.info("No feature importances/coefficients available for this model.")
                    else:
                        st.dataframe(fi.style.background_gradient(cmap="YlGnBu", subset=["Importance"]))
                        fig_bar = go.Figure(go.Bar(x=fi["Importance"][::-1], y=fi["Feature"][::-1], orientation='h', marker_color="#1F4E79"))
                        fig_bar.update_layout(title=f"Top 10 Features — {m}", height=420)
                        st.plotly_chart(fig_bar, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not show features for model {m}: {e}")

        # download results CSV
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download results (CSV)", csv, file_name="model_results.csv", mime="text/csv")

        # download best model
        if st.button("Download best model (.pkl)"):
            best_model = st.session_state['trained_models'][best_model_name]
            bio = model_to_bytes(best_model)
            st.download_button("Click to download model file", data=bio, file_name=f"{best_model_name.replace(' ', '_')}.pkl", mime="application/octet-stream")

# ---------------------------
# PREDICTION
# ---------------------------
elif page == "Prediction":
    st.header("🔮 Predict Loan Default (uses best model)")

    if 'model_results' not in st.session_state or 'trained_models' not in st.session_state:
        st.warning("Train models first (Modeling).")
    else:
        results_df = pd.DataFrame(st.session_state['model_results'])
        best_idx = results_df['ROC AUC'].idxmax()
        best_model_name = results_df.loc[best_idx, 'Model']
        best_model = st.session_state['trained_models'][best_model_name]
        scaler = st.session_state.get('scaler', None)
        selected_features = st.session_state.get('selected_features', None)
        X_train_cols = st.session_state.get('X_train').columns if st.session_state.get('X_train') is not None else None

        st.info(f"Predictions will use the best model: **{best_model_name}**")

        input_mode = st.radio("Input mode", ["Upload CSV", "Manual Input"])

        if input_mode == "Upload CSV":
            up = st.file_uploader("Upload CSV for prediction", type=["csv"])
            if up:
                df_in = pd.read_csv(up)
                st.write("Preview:")
                st.dataframe(df_in.head())

                # one-hot + alignment
                df_enc = pd.get_dummies(df_in)
                if X_train_cols is None:
                    st.error("Training feature set not found.")
                else:
                    df_enc = df_enc.reindex(columns=X_train_cols, fill_value=0)

                    # scale numeric columns (X_train_cols numeric subset)
                    if scaler:
                        numeric_cols = st.session_state['X_train'].select_dtypes(include=[np.number]).columns
                        try:
                            df_enc[numeric_cols] = scaler.transform(df_enc[numeric_cols])
                        except Exception as e:
                            st.error(f"Scaling problem: {e}")

                    preds = best_model.predict(df_enc)
                    probs = best_model.predict_proba(df_enc)[:,1] if hasattr(best_model, "predict_proba") else np.zeros(len(preds))

                    out = pd.DataFrame({"Predicted": preds, "Probability": probs})
                    st.dataframe(out.style.background_gradient(cmap="YlOrRd", subset=["Probability"]))

        else:  # Manual input
            original_df = st.session_state.get('original_df')
            if original_df is None:
                st.warning("Original data structure not found. Please re-run preprocessing or upload sample dataset.")
            else:
                user_input = {}
                for col in original_df.columns:
                    if original_df[col].dtype in [np.int64, np.float64, np.int32, np.float32]:
                        user_input[col] = st.number_input(col, value=float(original_df[col].median()))
                    else:
                        choices = original_df[col].dropna().unique().tolist()
                        if not choices:
                            user_input[col] = st.text_input(col, value="")
                        else:
                            user_input[col] = st.selectbox(col, options=choices)

                if st.button("Predict"):
                    user_df = pd.DataFrame([user_input])
                    user_df = pd.get_dummies(user_df)
                    if X_train_cols is None:
                        st.error("Training feature set not found.")
                    else:
                        user_df = user_df.reindex(columns=X_train_cols, fill_value=0)
                        if scaler:
                            numeric_cols = st.session_state['X_train'].select_dtypes(include=[np.number]).columns
                            try:
                                user_df[numeric_cols] = scaler.transform(user_df[numeric_cols])
                            except Exception as e:
                                st.error(f"Scaling problem: {e}")

                        pred = best_model.predict(user_df)[0]
                        prob = best_model.predict_proba(user_df)[0,1] if hasattr(best_model, "predict_proba") else 0.0

                        color = "#28a745" if pred == 0 else "#dc3545"
                        text = "NO DEFAULT (Low Risk)" if pred == 0 else "DEFAULT (High Risk)"
                        st.markdown(f"""
                        <div style='padding:18px; border-radius:8px; background-color:{color}; color:white; text-align:center;'>
                        <h3 style='margin:5px'>{text}</h3>
                        <p>Predicted probability of default: <b>{prob:.2f}</b></p>
                        </div>
                        """, unsafe_allow_html=True)

# ---------------------------
# End of app
# ---------------------------

