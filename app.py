# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Traffic Volume Predictor", page_icon="🚦")

# ---------------------------
# Helper utilities
# ---------------------------
@st.cache_resource
def load_model(path="best_random_forest_model.pkl"):
    if not os.path.exists(path):
        st.error(f"Model file not found at: {path}. Please upload or place the model file there.")
        return None
    model = joblib.load(path)
    return model

@st.cache_data
def load_encoders(path="encoders.pkl"):
    if os.path.exists(path):
        enc = joblib.load(path)
        return enc
    return None

@st.cache_data
def load_scaler(path="scaler.pkl"):
    if os.path.exists(path):
        scaler = joblib.load(path)
        return scaler
    return None

def safe_label_encode_df(df, encoders=None, fit_on_missing=False):
    """
    Encode object columns. If encoders dict supplied, use them.
    If not, either fit new LabelEncoders (if fit_on_missing True) or raise.
    Returns encoded df and encoders used.
    """
    df2 = df.copy()
    used_encoders = {} if encoders is None else dict(encoders)
    for col in df2.columns:
        if df2[col].dtype == 'object' or df2[col].dtype.name == 'category':
            if encoders and col in encoders:
                le = encoders[col]
                # map unknown values to new integer by fit on combined classes if necessary
                try:
                    df2[col] = le.transform(df2[col].astype(str))
                except Exception:
                    # fallback: fit a new encoder combining existing classes and new ones
                    all_labels = np.unique(np.concatenate([le.classes_, df2[col].astype(str).unique()]))
                    new_le = LabelEncoder()
                    new_le.fit(all_labels)
                    df2[col] = new_le.transform(df2[col].astype(str))
                    used_encoders[col] = new_le
            else:
                if fit_on_missing:
                    le = LabelEncoder()
                    df2[col] = le.fit_transform(df2[col].astype(str))
                    used_encoders[col] = le
                else:
                    raise ValueError(f"No encoder supplied for column '{col}'. Provide encoders or set fit_on_missing=True.")
    return df2, used_encoders

def ensure_numeric_columns(df):
    """Convert booleans and other convertible columns to numeric; raise if non-numeric remains."""
    df2 = df.copy()
    # convert boolean to int
    for col in df2.select_dtypes(include=['bool']).columns:
        df2[col] = df2[col].astype(int)
    # try to coerce any remaining object columns
    for col in df2.columns:
        if df2[col].dtype == 'object':
            try:
                df2[col] = pd.to_numeric(df2[col])
            except Exception:
                pass
    non_numeric = df2.select_dtypes(include=['object', 'category']).columns.tolist()
    return df2, non_numeric

def preprocess_for_model(df, model_columns, encoders=None, scaler=None, fit_on_missing_encoders=False):
    """
    Preprocess df to match model_columns.
    - Checks for missing columns.
    - Fills missing numeric columns with median.
    - Encodes categorical using encoders (or fits if allowed).
    - Applies scaler if provided.
    Returns preprocessed array and used encoders.
    """
    df_proc = df.copy()

    # Ensure same columns: add missing columns with zeros
    for col in model_columns:
        if col not in df_proc.columns:
            df_proc[col] = 0.0  # placeholder; better to warn

    # Keep only model_columns order
    df_proc = df_proc[model_columns]

    # Convert booleans -> int, attempt numeric conversion
    df_proc, non_numeric = ensure_numeric_columns(df_proc)

    # If any non-numeric columns remain, try encoding
    if len(non_numeric) > 0:
        try:
            df_proc, used_encoders = safe_label_encode_df(df_proc, encoders=encoders, fit_on_missing=fit_on_missing_encoders)
        except Exception as e:
            raise ValueError(f"Preprocessing failed: {e}")

    # Fill missing numeric values with median
    for col in df_proc.columns:
        if df_proc[col].isnull().any():
            median = df_proc[col].median()
            df_proc[col].fillna(median, inplace=True)

    # Apply scaler if exists
    if scaler is not None:
        try:
            df_scaled = scaler.transform(df_proc)
            return df_scaled, used_encoders if 'used_encoders' in locals() else encoders
        except Exception as e:
            st.warning(f"Scaler application failed: {e}. Returning unscaled data.")
            return df_proc.values, used_encoders if 'used_encoders' in locals() else encoders

    return df_proc.values, (used_encoders if 'used_encoders' in locals() else encoders)

# ---------------------------
# App layout and main logic
# ---------------------------
st.sidebar.title("Model & Data")
st.sidebar.write("Load model and optionally encoders/scaler (if you saved them).")
model_file = st.sidebar.file_uploader("Upload RandomForest model (.pkl/.joblib)", type=["pkl", "joblib"], accept_multiple_files=False)
encoders_file = st.sidebar.file_uploader("Upload encoders (encoders.pkl) (optional)", type=["pkl", "joblib"], accept_multiple_files=False)
scaler_file = st.sidebar.file_uploader("Upload scaler (scaler.pkl) (optional)", type=["pkl", "joblib"], accept_multiple_files=False)

# load model (from upload or disk)
if model_file is not None:
    try:
        model = joblib.load(model_file)
        st.sidebar.success("Model loaded from uploaded file.")
    except Exception as e:
        st.sidebar.error(f"Could not load uploaded model: {e}")
        model = None
else:
    model = load_model("best_random_forest_model.pkl")
    if model is not None:
        st.sidebar.success("Model loaded from best_random_forest_model.pkl")

# load encoders & scaler if provided
encoders = None
scaler = None
if encoders_file is not None:
    try:
        encoders = joblib.load(encoders_file)
        st.sidebar.success("Encoders loaded.")
    except Exception as e:
        st.sidebar.warning(f"Failed to load encoders: {e}")

if scaler_file is not None:
    try:
        scaler = joblib.load(scaler_file)
        st.sidebar.success("Scaler loaded.")
    except Exception as e:
        st.sidebar.warning(f"Failed to load scaler: {e}")

st.title("Bengaluru Traffic — Traffic Volume Prediction")
st.markdown("""
This app uses a RandomForestRegressor model you trained.
- Upload a CSV to run batch predictions
- Or use the manual form for a single prediction
- If you saved your encoders/scaler during training, upload them so predictions match training preprocessing exactly.
""")

# -------------
# Upload data
# -------------
st.header("Data Upload & Preview")
uploaded = st.file_uploader("Upload a CSV file with features (or leave empty to use sample)", type=['csv'])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.subheader("Uploaded data (first 5 rows)")
    st.dataframe(df.head())
else:
    st.info("No file uploaded. You can still run manual prediction below or test using sample synthetic rows.")
    df = None

# show model input columns if model is loaded
if model is not None:
    if hasattr(model, "feature_names_in_"):
        model_cols = list(model.feature_names_in_)
        st.sidebar.write("Model expects columns (feature_names_in_):")
        st.sidebar.write(model_cols)
    else:
        model_cols = None
        st.sidebar.write("Model does not expose `feature_names_in_`. Ensure your input columns match training X order.")
else:
    model_cols = None

# -------------
# Manual prediction form (single row)
# -------------
st.header("Single Prediction (Manual Input)")
with st.form("manual_input_form"):
    st.write("Enter feature values. Column names must match training columns.")
    if model_cols:
        # create inputs in a 2-column grid for convenience
        cols = st.columns(2)
        input_values = {}
        for i, col in enumerate(model_cols):
            # show only first 12 to avoid overly long form; provide toggle to show all cols
            if i < 12:
                if df is not None and col in df.columns:
                    default = float(df[col].iloc[0]) if np.issubdtype(df[col].dtype, np.number) else df[col].iloc[0]
                else:
                    default = 0.0
                # choose widget type
                if isinstance(default, (int, np.integer)) or 'int' in str(type(default)).lower():
                    input_values[col] = cols[i % 2].number_input(col, value=int(default))
                else:
                    input_values[col] = cols[i % 2].number_input(col, value=float(default))
            else:
                # hidden inputs for the rest; show if user toggles
                pass

        show_more = st.checkbox("Show all model columns (if not shown above)")
        if show_more:
            for col in model_cols[12:]:
                input_values[col] = st.number_input(col, value=0.0)
    else:
        st.write("Model column names not available. Provide input CSV or contact developer.")
        input_values = {}
        # offer simple manual inputs
        num1 = st.number_input("Feature A", value=0.0)
        input_values["Feature A"] = num1

    submit_button = st.form_submit_button(label="Predict (single)")

if submit_button:
    if model is None:
        st.error("Model not loaded. Upload your model file in the left sidebar or place it as best_random_forest_model.pkl.")
    else:
        input_df = pd.DataFrame([input_values])
        try:
            X_for_pred, used_encoders = preprocess_for_model(input_df, model_cols if model_cols else input_df.columns.tolist(), encoders=encoders, scaler=scaler, fit_on_missing_encoders=True)
            preds = model.predict(X_for_pred)
            st.success(f"Predicted Traffic Volume: {preds[0]:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------------
# Batch prediction & EDA
# -------------
st.header("Batch Prediction & Exploratory Data Analysis")
if df is not None:
    st.subheader("Preview of uploaded dataset")
    st.dataframe(df.head())

    # simple EDA toggles
    if st.checkbox("Show Data Info (dtypes & nulls)"):
        buf = io.StringIO()
        df.info(buf=buf)
        s = buf.getvalue()
        st.text(s)
        st.write("Null counts:")
        st.write(df.isnull().sum())

    if st.checkbox("Show Descriptive Statistics"):
        st.write(df.describe(include='all').T)

    # Visualization panels
    st.subheader("Visualizations")
    viz_col1, viz_col2 = st.columns([2, 1])
    with viz_col1:
        # correlation heatmap (numeric only)
        if st.checkbox("Show Correlation Heatmap"):
            num_df = df.select_dtypes(include=[np.number])
            if num_df.shape[1] >= 2:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            else:
                st.info("Need at least 2 numeric columns for correlation.")

        # histograms
        if st.checkbox("Show Histograms for numeric features"):
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            sel = st.multiselect("Select numeric columns", options=num_cols, default=num_cols[:6])
            fig, axs = plt.subplots(nrows=len(sel)//3 + 1, ncols=3, figsize=(15, 4*(len(sel)//3 + 1)))
            axs = axs.flatten()
            for i, c in enumerate(sel):
                sns.histplot(df[c].dropna(), bins=30, ax=axs[i], kde=True)
                axs[i].set_title(c)
            for j in range(i+1, len(axs)):
                axs[j].axis('off')
            st.pyplot(fig)

        # pairwise scatter (sampled to avoid heavy plotting)
        if st.checkbox("Show Pairwise Scatter (sampled)"):
            sample = df.select_dtypes(include=[np.number]).sample(min(500, len(df)))
            fig = sns.pairplot(sample.iloc[:, :6], diag_kind='kde')
            st.pyplot(fig)

    with viz_col2:
        # top value counts for categorical
        if st.checkbox("Show top categories for object columns"):
            obj_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in obj_cols:
                st.write(f"**{col}**")
                st.write(df[col].value_counts().head(10))

    # ----------------
    # Run batch prediction
    # ----------------
    st.subheader("Run Batch Prediction")
    if st.button("Run model prediction on uploaded data"):
        if model is None:
            st.error("Model not loaded.")
        else:
            # Attempt to preprocess and predict
            # If model exposes feature names, use them
            if model_cols is None:
                # assume df columns align exactly
                model_cols_local = df.columns.tolist()
            else:
                model_cols_local = model_cols

            # Try preprocessing and predict
            try:
                X_vals, used_enc = preprocess_for_model(df.copy(), model_cols_local, encoders=encoders, scaler=scaler, fit_on_missing_encoders=True)
                preds = model.predict(X_vals)
                result_df = df.copy()
                result_df["Predicted_Traffic_Volume"] = preds
                st.success("Batch prediction completed.")
                st.dataframe(result_df.head())
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

    # -------------
    # Model evaluation if ground truth present
    # -------------
    if "Traffic Volume" in df.columns and st.checkbox("Evaluate model on uploaded data (if it contains ground truth 'Traffic Volume')"):
        if model is None:
            st.error("Model not loaded.")
        else:
            try:
                model_cols_local = model_cols if model_cols else [c for c in df.columns if c != "Traffic Volume"]
                X_vals, _ = preprocess_for_model(df.drop(columns=["Traffic Volume"]).copy(), model_cols_local, encoders=encoders, scaler=scaler, fit_on_missing_encoders=True)
                preds = model.predict(X_vals)
                y_true = df["Traffic Volume"].values
                mae = mean_absolute_error(y_true, preds)
                rmse = np.sqrt(mean_squared_error(y_true, preds))
                r2 = r2_score(y_true, preds)
                st.write("Evaluation on uploaded dataset")
                st.write(f"MAE: {mae:.2f}")
                st.write(f"RMSE: {rmse:.2f}")
                st.write(f"R²: {r2:.4f}")

                # show actual vs predicted scatter
                fig, ax = plt.subplots(figsize=(6,6))
                ax.scatter(y_true, preds, alpha=0.5)
                ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs Predicted")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

# -------------
# Advanced section: model introspection (feature importance & permutation importance)
# -------------
st.header("Model Introspection & Explainability")
if model is not None:
    try:
        # Feature importance plot
        if hasattr(model, "feature_importances_") and model_cols:
            imp = pd.Series(model.feature_importances_, index=model_cols).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(8, max(4, len(imp)*0.3)))
            imp.plot(kind='barh', ax=ax)
            ax.set_title("Feature Importances")
            st.pyplot(fig)

        # Permutation importance (slower; optional)
        if st.checkbox("Compute Permutation Importance (slower)"):
            if 'X_test' not in locals() and df is None:
                st.info("Upload a dataset or run a batch prediction so that a test set is available for permutation importance.")
            else:
                # create a test matrix
                sample_df = df if df is not None else pd.DataFrame(np.zeros((10, len(model_cols))), columns=model_cols)
                try:
                    X_vals, _ = preprocess_for_model(sample_df.copy(), model_cols, encoders=encoders, scaler=scaler, fit_on_missing_encoders=True)
                    r = permutation_importance(model, X_vals, model.predict(X_vals), n_repeats=10, n_jobs=-1, random_state=42)
                    perm_imp = pd.Series(r.importances_mean, index=model_cols).sort_values(ascending=True)
                    fig, ax = plt.subplots(figsize=(8, max(4, len(perm_imp)*0.3)))
                    perm_imp.plot(kind='barh', ax=ax)
                    ax.set_title("Permutation Importances")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Permutation importance failed: {e}")
    except Exception as e:
        st.error(f"Model introspection error: {e}")

# -------------
# Utilities & info
# -------------
st.header("Utilities & Developer Info")
with st.expander("Model & Preprocessing Files (Paths)"):
    st.write("Model file:", "best_random_forest_model.pkl (or uploaded file)")
    st.write("Optional encoders:", "encoders.pkl (if provided)")
    st.write("Optional scaler:", "scaler.pkl (if provided)")

with st.expander("Quick Troubleshooting"):
    st.write("""
    - If predictions look wrong, ensure the input columns match the exact names & order used during training.
    - If you used LabelEncoder/OneHotEncoder/scaler during training, upload the same artifacts (encoders.pkl, scaler.pkl) in the left sidebar.
    - If you get errors about non-numeric data, check the uploaded CSV for stray text columns.
    - To recreate encoders: in training, save a dict of {col: LabelEncoder()} via joblib.dump(encoders_dict, "encoders.pkl")
    """)

st.caption("App created to deploy your RandomForestRegressor model. Ensure you provide matching feature columns and preprocessing artifacts for exact parity with training.")
