import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import zipfile
import json
import base64
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.base import clone
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve


# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Loan Default App", layout="wide")

# --- SIDEBAR MENU ---
st.sidebar.title("📊 Navigation Menu")

menu = st.sidebar.radio(
    "Go to",
    ["Home", "Exploratory Data Analysis","Visualisation", "Modeling", "Model Evaluation", "Prediction"]
)

if menu == "Home":
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

    # --- FILE UPLOAD SECTION ---
    st.markdown("### 📁 Upload Your Loan Dataset")
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drag and drop or select a CSV file", type=["csv"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- LOAD DATA BUTTON ---
    if uploaded_file is not None:
        if st.button("📊 Load Data", use_container_width=True):
            df = pd.read_csv(uploaded_file)
            st.session_state['uploaded_data'] = df  # ✅ store in session_state
            st.success("✅ Data loaded successfully!")
            st.write("### Preview of Uploaded Data")
            st.dataframe(df.head(10))
    else:
        st.info("Please upload a CSV file to get started.")


elif menu == 'Exploratory Data Analysis':
    # --- EDA PAGE STYLE ---
    st.markdown("""
    <style>
    /* --- EDA CONTAINER STYLING --- */
    .eda-box {
        background-color: #fefcf6;
        padding: 25px;
        border-radius: 14px;
        box-shadow: 0 2px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 30px;
        transition: all 0.3s ease;
    }
    .eda-box:hover {
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.15);
        transform: scale(1.01);
    }

    /* --- HEADER AND SUBHEADER --- */
    .eda-header {
        color: #1F4E79;
        font-weight: 800;
        font-size: 28px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
    }
    .eda-header span {
        margin-right: 10px;
        font-size: 30px;
        transition: all 0.3s ease;
    }
    .eda-header:hover span {
        color: #D4AF37;
        transform: rotate(10deg);
    }
    .eda-subheader {
        color: #D4AF37;
        font-weight: 600;
        font-size: 19px;
        margin-top: 25px;
    }

    /* --- BUTTON ENHANCEMENTS --- */
    div[data-testid="stButton"] > button {
        background: linear-gradient(90deg, #1F4E79, #D4AF37);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    }
    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(90deg, #D4AF37, #1F4E79);
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="eda-box">', unsafe_allow_html=True)
    st.markdown('<div class="eda-header"><span>📊</span>Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)

    if 'uploaded_data' not in st.session_state:
        st.warning("⚠️ Please upload your dataset on the Home page first to begin analysis.")
    else:
        df = st.session_state['uploaded_data']
        st.success("✅ Dataset successfully loaded for analysis!")

        # --- BASIC DATA OVERVIEW ---
        st.markdown('<div class="eda-subheader">🔍 Data Overview</div>', unsafe_allow_html=True)
        st.write(df.describe().T)

        # --- DATA TYPES & MISSING VALUES ---
        st.markdown('<div class="eda-subheader">🧾 Data Information</div>', unsafe_allow_html=True)
        st.write(df.dtypes)

        st.markdown('<div class="eda-subheader">🚨 Missing Values Summary</div>', unsafe_allow_html=True)
        st.write(df.isnull().sum())

        # --- IMPUTE BUTTON ---
        if st.button("🧹 Impute Missing Values (Median for Numeric, Mode for Categorical)", use_container_width=True):
            num_cols = df.select_dtypes(include=np.number).columns
            cat_cols = df.select_dtypes(exclude=np.number).columns

            for col in num_cols:
                df[col] = df[col].fillna(df[col].median())
            for col in cat_cols:
                df[col] = df[col].fillna(df[col].mode()[0])

            st.session_state['uploaded_data'] = df
            st.success("✅ Missing values imputed successfully!")

        # --- CLEANED DATA PREVIEW ---
        st.markdown('<div class="eda-subheader">✅ Cleaned Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head())

        # --- INSIGHTS ---
        st.markdown('<div class="eda-subheader">💡 Quick Insights & Recommendations</div>', unsafe_allow_html=True)
        st.markdown("""
        - Proceed to the **Visualisation** page to explore patterns and correlations.  
        - Focus on columns that may strongly influence **loan defaults** such as income, credit score, or loan amount.  
        - If categorical variables dominate, explore **group-wise defaults** for deeper insights.  
        """)

    st.markdown('</div>', unsafe_allow_html=True)


elif menu == 'Visualisation':
    st.markdown("""
    <style>
    .viz-box {
        background-color: #fefcf6;
        padding: 25px;
        border-radius: 14px;
        box-shadow: 0 2px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 30px;
        transition: all 0.3s ease;
    }
    .viz-box:hover {
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.15);
        transform: scale(1.01);
    }
    .viz-header {
        color: #1F4E79;
        font-weight: 800;
        font-size: 28px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
    }
    .viz-header span {
        margin-right: 10px;
        font-size: 30px;
        transition: all 0.3s ease;
    }
    .viz-header:hover span {
        color: #D4AF37;
        transform: rotate(10deg);
    }
    .viz-subheader {
        color: #D4AF37;
        font-weight: 600;
        font-size: 19px;
        margin-top: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="viz-box">', unsafe_allow_html=True)
    st.markdown('<div class="viz-header"><span>📈</span>Data Visualisations</div>', unsafe_allow_html=True)

    if 'uploaded_data' not in st.session_state:
        st.warning("⚠️ Please upload your dataset on the Home page first to visualize data.")
    else:
        df = st.session_state['uploaded_data']
        st.session_state['original_df'] = df.copy()

        st.markdown('<div class="viz-subheader">🎨 Select a Visualization Type</div>', unsafe_allow_html=True)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        viz_type = st.radio("Choose visualization type:", ["Histogram", "Box Plot", "Correlation Heatmap", "Category Counts",
                                                           "Category vs Numeric Mean", "Category Distribution (Pie Chart)"])

        # --- FIGURE DIMENSIONS ---
        fig_width = 1100
        fig_height = 600

        if viz_type == "Histogram":
            col = st.selectbox("Select numeric column:", numeric_cols)
            fig = px.histogram(df, x=col, nbins=30, color_discrete_sequence=["#1F4E79"])
            fig.update_layout(
                title=f"Distribution of {col}",
                title_font=dict(size=20, color="#D4AF37"),
                width=fig_width,
                height=fig_height,
                paper_bgcolor="#fefcf6",
                plot_bgcolor="white"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Box Plot":
            col = st.selectbox("Select numeric column:", numeric_cols)
            fig = px.box(df, y=col, color_discrete_sequence=["#D4AF37"])
            fig.update_layout(
                title=f"Box Plot of {col}",
                title_font=dict(size=20, color="#1F4E79"),
                width=fig_width,
                height=fig_height,
                paper_bgcolor="#fefcf6",
                plot_bgcolor="white"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Correlation Heatmap":
            corr = df.corr(numeric_only=True)
            fig = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="blues",
                title="Correlation Heatmap"
            )
            fig.update_layout(
                title_font=dict(size=20, color="#D4AF37"),
                width=fig_width,
                height=fig_height,
                paper_bgcolor="#fefcf6"
            )
            st.plotly_chart(fig, use_container_width=True)


        elif viz_type == "Category Counts":
            col = st.selectbox("Select categorical column:", categorical_cols)
            df_counts = df[col].value_counts().reset_index()
            df_counts.columns = [col, "Count"]  # Rename for clarity
            fig = px.bar(df_counts,x=col,y="Count",color_discrete_sequence=["#D4AF37"],width=900,height=600)
            fig.update_layout(title=f"Count of Each Category in {col}",xaxis_title=col,yaxis_title="Count",
                              title_font=dict(size=20, color="#1F4E79"),font=dict(size=14),plot_bgcolor="white")

            st.plotly_chart(fig, use_container_width=True)

            # === CATEGORY VS NUMERIC MEAN ===
        elif viz_type == "Category vs Numeric Mean":
            col1 = st.selectbox("Select categorical column:", categorical_cols)
            col2 = st.selectbox("Select numeric column:", numeric_cols)

            df_grouped = df.groupby(col1)[col2].mean().reset_index()

            fig = px.bar(
                df_grouped,
                x=col1,
                y=col2,
                color_discrete_sequence=["#1F4E79"],
                width=900,
                height=600
            )
            fig.update_layout(
                title=f"Average {col2} by {col1}",
                xaxis_title=col1,
                yaxis_title=f"Average {col2}",
                title_font=dict(size=20, color="#1F4E79"),
                font=dict(size=14),
                plot_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

        # === CATEGORY DISTRIBUTION (PIE CHART) ===
        elif viz_type == "Category Distribution (Pie Chart)":
            col = st.selectbox("Select categorical column:", categorical_cols)

            df_counts = df[col].value_counts().reset_index()
            df_counts.columns = [col, "Count"]

            fig = px.pie(
                df_counts,
                names=col,
                values="Count",
                color_discrete_sequence=px.colors.sequential.Blues_r,
                width=800,
                height=600
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(
                title=f"Distribution of {col}",
                title_font=dict(size=20, color="#1F4E79"),
                font=dict(size=14)
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- INSIGHTS ---
        st.markdown('<div class="viz-subheader">💡 Visualization Insights</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Histograms** help detect skewness and data spread.  
        - **Box Plots** show outliers and feature variation.  
        - **Heatmaps** reveal feature relationships that can improve your models.  
        - **Category Counts** highlight class balance and distribution patterns.  
        """)
    st.markdown('</div>', unsafe_allow_html=True)



# ============================================================
# ⚙️ MODELING PAGE
# ============================================================
elif menu == "Modeling":
    st.title("⚙️ Model Training & Evaluation")

    if 'uploaded_data' not in st.session_state:
        st.warning("⚠️ Please upload and clean your dataset on the Home or EDA page first.")
    else:
        df = st.session_state['uploaded_data']

        st.markdown("""
        <div style='background-color:#fefcf6; padding:20px; border-radius:10px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.08);'>
            <h4 style='color:#1F4E79;'>🧩 Model Setup</h4>
            <p style='color:#333;'>Select your target variable and features, then choose a model to train and view its performance instantly.</p>
        </div>
        """, unsafe_allow_html=True)

        # --- Target and Feature Selection ---
        target_col = st.selectbox("🎯 Target Variable", df.columns)
        feature_cols = st.multiselect(
            "🔍 Feature Columns (Independent Variables)",
            [col for col in df.columns if col != target_col]
        )

        if not feature_cols:
            st.info("Please select at least one feature column to continue.")
        else:
            X = df[feature_cols]
            y = df[target_col]

            # Split dataset
            test_size = st.slider("Select Test Size (%)", 10, 50, 20)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=42
            )

            # Preprocessing
            numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = X_train.select_dtypes(exclude=['int64', 'float64']).columns

            scaler = None
            if len(categorical_cols) > 0:
                X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
                X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
                X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

            if len(numeric_cols) > 0:
                scaler = StandardScaler()
                X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
                X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

            # --- Model Selection ---
            st.markdown("### 🧠 Choose a Model")
            model_type = st.selectbox(
                "Select a Model to Train",
                ["Logistic Regression", "Decision Tree", "Random Forest"]
            )

            if st.button("🚀 Train Model", use_container_width=True):
                with st.spinner("Training your model... ⏳"):

                    # --- Train the Model ---
                    if model_type == "Logistic Regression":
                        model = LogisticRegression(max_iter=1000, random_state=42)
                    elif model_type == "Decision Tree":
                        model = DecisionTreeClassifier(random_state=42)
                    elif model_type == "Random Forest":
                        model = RandomForestClassifier(n_estimators=100, random_state=42)

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]

                    # --- Metrics ---
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    roc_auc = roc_auc_score(y_test, y_prob)
                    cm = confusion_matrix(y_test, y_pred)
                    fpr, tpr, _ = roc_curve(y_test, y_prob)

                    # --- Save for Evaluation Page ---
                    if 'model_results' not in st.session_state:
                        st.session_state['model_results'] = []
                        st.session_state['roc_data'] = []
                        st.session_state['conf_matrices'] = []

                    st.session_state['model_results'].append({
                        "Model": model_type,
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1 Score": f1,
                        "ROC AUC": roc_auc
                    })

                    st.session_state['roc_data'].append({
                        "model": model_type,
                        "fpr": fpr,
                        "tpr": tpr,
                        "roc_auc": roc_auc
                    })

                    st.session_state['conf_matrices'].append({
                        "model": model_type,
                        "matrix": cm,
                        "labels": y.unique().tolist()
                    })

                    st.session_state['trained_model'] = model
                    st.session_state['scaler'] = scaler
                    st.session_state['X_train'] = X_train
                    st.session_state['X_test'] = X_test
                    st.session_state['original_df'] = df  # Save for prediction inputs

                # --- Display Results ---
                st.success(f"✅ {model_type} trained successfully!")

                # --- Metrics Table ---
                st.markdown("### 📊 Model Performance Metrics")
                metrics_df = pd.DataFrame([{
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "ROC AUC": roc_auc
                }])
                st.dataframe(
                    metrics_df.style.background_gradient(cmap="YlGnBu").format("{:.3f}"),
                    use_container_width=True
                )

                # --- Confusion Matrix ---
                st.markdown("### 🔲 Confusion Matrix")
                fig_cm = ff.create_annotated_heatmap(
                    z=cm[::-1],
                    x=y.unique().tolist(),
                    y=list(reversed(y.unique().tolist())),
                    colorscale='Blues'
                )
                fig_cm.update_layout(
                    width=500, height=400, title=f"Confusion Matrix - {model_type}",
                    title_font=dict(size=16, color="#1F4E79")
                )
                st.plotly_chart(fig_cm, use_container_width=True)

                # --- ROC Curve ---
                st.markdown("### 📉 ROC Curve")
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(fpr, tpr, color="#1F4E79", lw=2, label=f"{model_type} (AUC = {roc_auc:.2f})")
                ax.plot([0, 1], [0, 1], 'r--', label="Random Classifier")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend()
                st.pyplot(fig)

                # --- 🔝 Top 10 Feature Importances ---
                st.markdown("### 🔝 Top 10 Most Important Features")
                if model_type == "Logistic Regression":
                    importances = abs(model.coef_[0])
                    feature_importance = pd.DataFrame({
                        "Feature": X_train.columns,
                        "Importance": importances
                    }).sort_values(by="Importance", ascending=False).head(10)
                else:
                    importances = model.feature_importances_
                    feature_importance = pd.DataFrame({
                        "Feature": X_train.columns,
                        "Importance": importances
                    }).sort_values(by="Importance", ascending=False).head(10)

                st.dataframe(
                    feature_importance.style.background_gradient(cmap="YlOrRd", subset=["Importance"]),
                    use_container_width=True
                )

                fig_imp = go.Figure(
                    go.Bar(
                        x=feature_importance["Importance"][::-1],
                        y=feature_importance["Feature"][::-1],
                        orientation='h',
                        marker_color='#1F4E79'
                    )
                )
                fig_imp.update_layout(
                    title="Top 10 Feature Importances",
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    height=500,
                    template="plotly_white"
                )
                st.plotly_chart(fig_imp, use_container_width=True)


# ============================================================
# 📈 MODEL EVALUATION PAGE (Enhanced with Bar Chart Comparison)
# ============================================================
elif menu == "Model Evaluation":
    st.title("📈 Model Evaluation & Comparison")

    if 'model_results' not in st.session_state or len(st.session_state['model_results']) == 0:
        st.warning("⚠️ Please train at least one model first under the 'Modeling' page.")
    else:
        results_df = pd.DataFrame(st.session_state['model_results'])

        # --- Format numeric columns ---
        numeric_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
        for col in numeric_cols:
            results_df[col] = results_df[col].astype(float).round(3)

        # --- Metrics Table ---
        st.markdown("""
        <div style='background-color:#fefcf6; padding:20px; border-radius:10px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.08); margin-bottom:20px;'>
            <h4 style='color:#1F4E79;'>📊 Performance Summary Table</h4>
            <p style='color:#333;'>Compare the metrics of all trained models below.</p>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            results_df.style.background_gradient(cmap="YlGnBu", subset=numeric_cols)
                            .format("{:.3f}", subset=numeric_cols),
            use_container_width=True
        )

        # --- Identify Best Model ---
        best_model = results_df.loc[results_df['ROC AUC'].idxmax()]
        st.success(
            f"🏆 **Best Model:** {best_model['Model']} "
            f"— ROC AUC: {best_model['ROC AUC']:.3f}, "
            f"Accuracy: {best_model['Accuracy']:.3f}"
        )

        # --- Bar Chart Comparison ---
        st.markdown("### 📊 Metric Comparison Across Models")

        fig_bar = go.Figure()
        for metric in numeric_cols:
            fig_bar.add_trace(go.Bar(
                x=results_df['Model'],
                y=results_df[metric],
                name=metric,
                text=results_df[metric],
                textposition='auto'
            ))

        fig_bar.update_layout(
            barmode='group',
            title="Model Performance Comparison",
            xaxis_title="Model Type",
            yaxis_title="Score",
            template="plotly_white",
            title_font=dict(size=20, color="#1F4E79"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_bar, use_container_width=True, key="metric_bar_chart")

        # --- ROC Curve Comparison ---
        st.markdown("### 📉 ROC Curve Comparison")
        fig, ax = plt.subplots(figsize=(8, 6))
        for res in st.session_state['roc_data']:
            ax.plot(res['fpr'], res['tpr'], label=f"{res['model']} (AUC = {res['roc_auc']:.2f})")
        ax.plot([0, 1], [0, 1], 'r--', label="Random Classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve Comparison")
        ax.legend()
        st.pyplot(fig)

        # --- Confusion Matrices ---
        st.markdown("### 🔲 Confusion Matrices for Each Model")
        for i, res in enumerate(st.session_state['conf_matrices']):
            st.markdown(f"#### {res['model']}")
            cm = res['matrix']
            fig_cm = ff.create_annotated_heatmap(
                z=cm[::-1],
                x=res['labels'],
                y=list(reversed(res['labels'])),
                colorscale='Blues'
            )
            fig_cm.update_layout(
                title=f"Confusion Matrix - {res['model']}",
                width=600,
                height=450,
                title_font=dict(size=18, color="#1F4E79")
            )
            # ✅ Unique key for each chart
            st.plotly_chart(fig_cm, use_container_width=True, key=f"conf_matrix_{i}_{res['model']}")

        # --- Download Model Results ---
        st.markdown("### 💾 Export Results")
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Model Results (CSV)",
            data=csv,
            file_name="model_evaluation_results.csv",
            mime="text/csv",
            use_container_width=True
        )




elif menu == "Prediction":
    st.title("🔮 Loan Default Prediction")

    st.markdown("""
    <div style='background-color:#fefcf6; padding:20px; border-radius:10px; 
                box-shadow:0 2px 8px rgba(0,0,0,0.08);'>
        <h4 style='color:#1F4E79;'>💡 Predict Loan Default</h4>
        <p style='color:#333;'>Use the trained model to predict whether a borrower is likely to default on a loan.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Check for trained model ---
    if 'trained_model' not in st.session_state:
        st.warning("⚠️ Please train a model in the 'Modeling' section before making predictions.")
    else:
        model = st.session_state['trained_model']
        scaler = st.session_state.get('scaler', None)
        X_train = st.session_state['X_train']

        st.subheader("📁 Upload Data or Enter Manually")
        option = st.radio("Choose Input Method:", ["Upload CSV File", "Manual Input"])

        # --- Option 1: Upload CSV File ---
        if option == "Upload CSV File":
            uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])
            if uploaded_file is not None:
                input_df = pd.read_csv(uploaded_file)
                st.write("✅ Uploaded Data Preview:")
                st.dataframe(input_df.head())

                # Encode & align columns
                input_df = pd.get_dummies(input_df)
                missing_cols = [col for col in X_train.columns if col not in input_df.columns]
                for col in missing_cols:
                    input_df[col] = 0
                input_df = input_df[X_train.columns]

                # Scale numeric columns
                if scaler:
                    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
                    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

                # Predict
                preds = model.predict(input_df)
                probs = model.predict_proba(input_df)[:, 1]

                results = pd.DataFrame({
                    "Predicted_Default": preds,
                    "Default_Probability": probs
                })
                st.success("✅ Predictions Generated Successfully!")
                st.dataframe(results.style.background_gradient(cmap="YlOrRd", subset=["Default_Probability"]))

        # --- Option 2: Manual Input ---
        else:
            st.write("Please input borrower details below:")

            # --- Retrieve preprocessing info ---
            if 'original_df' in st.session_state:
                original_df = st.session_state['original_df']
            else:
                st.warning("⚠️ Original data structure not found. Please process data first in the EDA/Data Processing section.")
                st.stop()

            input_data = {}
            for col in original_df.columns:
                if original_df[col].dtype in ['int64', 'float64']:
                    input_data[col] = st.number_input(
                        f"{col}",
                        value=float(original_df[col].mean())
                    )
                else:
                    input_data[col] = st.selectbox(
                        f"{col}",
                        options=sorted(original_df[col].dropna().unique())
                    )

            if st.button("🔍 Predict Loan Default", use_container_width=True):
                user_df = pd.DataFrame([input_data])

                # --- Encode and align columns with training data ---
                user_df = pd.get_dummies(user_df)
                missing_cols = [col for col in X_train.columns if col not in user_df.columns]
                for col in missing_cols:
                    user_df[col] = 0
                user_df = user_df[X_train.columns]

                # --- Scale numeric columns (after alignment) ---
                if scaler:
                    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
                    user_df[numeric_cols] = scaler.transform(user_df[numeric_cols])

                # --- Predict ---
                prediction = model.predict(user_df)[0]
                probability = model.predict_proba(user_df)[0][1]

                # --- Display result ---
                st.markdown("---")
                st.subheader("🎯 Prediction Result")
                color = "#28a745" if prediction == 0 else "#dc3545"
                result_text = "NO DEFAULT (Low Risk)" if prediction == 0 else "DEFAULT (High Risk)"

                st.markdown(f"""
                <div style='padding:20px; border-radius:10px; background-color:{color}; color:white; text-align:center;'>
                    <h3>{result_text}</h3>
                    <p>Predicted Probability of Default: <b>{probability:.2f}</b></p>
                </div>
                """, unsafe_allow_html=True)
