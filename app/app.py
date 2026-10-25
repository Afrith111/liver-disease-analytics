
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Liver Intelligence",
    page_icon=":material/health_and_safety:",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "cleaned_liver_patient.csv"
MODEL_PATH = ROOT / "models" / "liver_model.pkl"

# ---------- INTERACTIVE DASHBOARD THEME ----------
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(ellipse at 12% 0%, rgba(12, 113, 128, 0.20), transparent 38%),
        radial-gradient(ellipse at 92% 18%, rgba(205, 91, 75, 0.14), transparent 32%),
        linear-gradient(145deg, #07171c 0%, #0b1b22 48%, #111d25 100%);
    color: #EAF6FF;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #081519, #0c2025);
    border-right: 1px solid #28535a;
}
.block-container {
    padding-top: 1.2rem;
    max-width: 1600px;
}
h1, h2, h3 {
    color: #8ce5d2 !important;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(16, 49, 53, 0.92), rgba(14, 31, 39, 0.94));
    border: 1px solid #326b69;
    padding: 18px;
    border-radius: 8px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.16);
    transition: transform 180ms ease, border-color 180ms ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: #83d9c6;
}
[data-testid="stMetricValue"] {
    color: #8ce5d2;
}
[data-testid="stMetricLabel"] {
    color: #c0d7d4;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #28535a;
    border-radius: 8px;
}
.stButton > button {
    background: linear-gradient(120deg, #176b6b, #287f74);
    color: white;
    border: 1px solid #6dd4bd;
    border-radius: 8px;
    min-height: 2.8rem;
    transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(61, 185, 157, 0.2);
    filter: brightness(1.12);
}
div[data-testid="stRadio"] label {
    color: #c5dedb;
}
hr {
    border-color: #28535a;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.sidebar.markdown("# 🩺 LIVER AI")
st.sidebar.caption("HEALTH ANALYTICS SYSTEM")
st.sidebar.markdown("---")
st.sidebar.caption("PROJECT STATUS")
st.sidebar.success("Analytics Dashboard")

try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error(f"Unable to load data or model: {e}")
    st.stop()

target_col = next(
    (c for c in ["dataset", "selector"]
     if c in df.columns),
    None
)

if target_col is None:
    st.error("Target column not found. Check your dataset columns.")
    st.stop()

features = df.drop(columns=[target_col]).copy()
y = df[target_col].map({1: 1, 2: 0})

if y.isna().any():
    st.error("Unexpected target labels. Verify your dataset encoding.")
    st.stop()

# Shared chart styling
def style_chart(fig, height=320):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(4,15,32,0.7)",
        font=dict(color="#CDEBFF"),
        margin=dict(l=20, r=20, t=50, b=25),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CDEBFF")
        ),
        title_font=dict(color="#64E8FF", size=16)
    )
    fig.update_xaxes(
        gridcolor="#183550",
        linecolor="#245274"
    )
    fig.update_yaxes(
        gridcolor="#183550",
        linecolor="#245274"
    )
    return fig

st.title("LIVER INTELLIGENCE", icon=":material/health_and_safety:")
st.caption("PATIENT DATA • BLOOD TEST ANALYTICS • MACHINE LEARNING")
pages = ["Overview", "EDA", "Model Performance", "Predict"]
page_icons = [
    ":material/dashboard:",
    ":material/query_stats:",
    ":material/monitoring:",
    ":material/person_search:"
]
if "active_page" not in st.session_state:
    st.session_state.active_page = "Overview"

nav_columns = st.columns(len(pages))
for index, (column, page_name, page_icon) in enumerate(zip(nav_columns, pages, page_icons)):
    with column:
        if st.button(
            f"{index + 1:02d}  {page_name}",
            icon=page_icon,
            key=f"nav_{page_name}",
            type="primary" if st.session_state.active_page == page_name else "secondary",
            width="stretch"
        ):
            st.session_state.active_page = page_name
            st.rerun()

page = st.session_state.active_page
st.markdown("---")

# ---------- OVERVIEW ----------
if page == "Overview":
    st.subheader("SYSTEM OVERVIEW")

    total = len(df)
    disease_count = int((y == 1).sum())
    no_disease_count = int((y == 0).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PATIENT RECORDS", f"{total:,}")
    c2.metric("FEATURES", features.shape[1])
    c3.metric("CLASS 1 RECORDS", disease_count)
    c4.metric("MISSING VALUES", int(df.isna().sum().sum()))

    st.markdown("### PATIENT DATA MONITOR")

    left, right = st.columns([1.35, 1])

    with left:
        st.markdown("#### Blood Test Feature Distribution")

        numeric_cols = features.select_dtypes(
            include=np.number
        ).columns.tolist()

        selected = st.selectbox(
            "Select blood-test measurement",
            numeric_cols
        )
        chart_mode = st.segmented_control(
            "Chart style",
            ["Histogram", "Box", "Violin"],
            key="overview_chart_mode"
        )
        chart_data = df.copy()
        chart_data["Patient class"] = y.map({1: "Class 1", 0: "Class 0"})

        if chart_mode == "Box":
            fig = px.box(
                chart_data,
                x="Patient class",
                y=selected,
                color="Patient class",
                title=f"{selected} by patient class",
                color_discrete_map={"Class 1": "#ef8c70", "Class 0": "#80d7bf"}
            )
        elif chart_mode == "Violin":
            fig = px.violin(
                chart_data,
                x="Patient class",
                y=selected,
                color="Patient class",
                box=True,
                points="outliers",
                title=f"{selected} by patient class",
                color_discrete_map={"Class 1": "#ef8c70", "Class 0": "#80d7bf"}
            )
        else:
            fig = px.histogram(
                chart_data,
                x=selected,
                color="Patient class",
                nbins=25,
                barmode="overlay",
                opacity=0.78,
                title=f"Distribution of {selected}",
                color_discrete_map={"Class 1": "#ef8c70", "Class 0": "#80d7bf"}
            )
        st.plotly_chart(
            style_chart(fig),
            width="stretch",
            alt=f"{chart_mode} chart showing {selected} across patient classes"
        )

    with right:
        st.markdown("#### Patient Class Distribution")

        counts = pd.Series({
            "Class 1": disease_count,
            "Class 0": no_disease_count
        })

        fig = go.Figure(go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.65,
            marker=dict(
                colors=["#ef8c70", "#80d7bf"],
                line=dict(color="#071326", width=3)
            ),
            textinfo="label+percent"
        ))
        fig.update_layout(title="Dataset Class Split")
        st.plotly_chart(
            style_chart(fig),
            width="stretch",
            alt="Donut chart showing the share of each patient class"
        )

    st.markdown("### DATA RECORDS")
    record_controls = st.columns([1.4, 1])
    with record_controls[0]:
        record_class = st.segmented_control(
            "Patient cohort",
            ["All patients", "Class 1", "Class 0"],
            key="record_cohort"
        )
    with record_controls[1]:
        row_limit = max(1, min(100, len(df)))
        row_count = st.slider(
            "Rows to display",
            min_value=1,
            max_value=row_limit,
            value=min(15, row_limit),
            key="record_count"
        )

    record_data = df.copy()
    if record_class != "All patients":
        class_value = 1 if record_class == "Class 1" else 0
        record_data = record_data.loc[y == class_value]
    st.dataframe(
        record_data.head(row_count),
        width="stretch",
        alt="Patient records filtered by the selected cohort"
    )

# ---------- EDA ----------
elif page == "EDA":
    st.subheader("EXPLORATORY DATA ANALYSIS")

    numeric = features.select_dtypes(
        include=np.number
    )

    analysis_view = st.segmented_control(
        "Explore",
        ["Feature distribution", "Feature relationship"],
        key="eda_view"
    )
    class_data = df.copy()
    class_data["Patient class"] = y.map({1: "Class 1", 0: "Class 0"})

    if analysis_view == "Feature distribution":
        controls = st.columns([1, 1.4])
        with controls[0]:
            col = st.selectbox("Feature", numeric.columns.tolist(), key="eda_feature")
        with controls[1]:
            distribution = st.segmented_control(
                "Display",
                ["Histogram", "Box", "Violin"],
                key="eda_distribution"
            )

        if distribution == "Box":
            fig = px.box(
                class_data,
                x="Patient class",
                y=col,
                color="Patient class",
                points="outliers",
                color_discrete_map={"Class 1": "#ef8c70", "Class 0": "#80d7bf"},
                title=f"{col} by patient class"
            )
        elif distribution == "Violin":
            fig = px.violin(
                class_data,
                x="Patient class",
                y=col,
                color="Patient class",
                box=True,
                points="outliers",
                color_discrete_map={"Class 1": "#ef8c70", "Class 0": "#80d7bf"},
                title=f"{col} by patient class"
            )
        else:
            fig = px.histogram(
                class_data,
                x=col,
                color="Patient class",
                marginal="box",
                barmode="overlay",
                opacity=0.78,
                color_discrete_map={"Class 1": "#ef8c70", "Class 0": "#80d7bf"},
                title=f"{col} distribution by patient class"
            )
        st.plotly_chart(
            style_chart(fig),
            width="stretch",
            alt=f"{distribution} chart comparing {col} by patient class"
        )
    else:
        controls = st.columns(2)
        with controls[0]:
            x_feature = st.selectbox("Horizontal axis", numeric.columns.tolist(), key="eda_x")
        with controls[1]:
            y_options = [name for name in numeric.columns if name != x_feature]
            y_feature = st.selectbox("Vertical axis", y_options, key="eda_y")

        fig = px.scatter(
            class_data,
            x=x_feature,
            y=y_feature,
            color="Patient class",
            hover_data=[name for name in ["age", "gender"] if name in class_data.columns],
            color_discrete_map={"Class 1": "#ef8c70", "Class 0": "#80d7bf"},
            opacity=0.78,
            title=f"{y_feature} versus {x_feature}"
        )
        fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="#d8eee7")))
        st.plotly_chart(
            style_chart(fig, 430),
            width="stretch",
            alt=f"Scatter plot of {y_feature} against {x_feature}, colored by patient class"
        )

    st.markdown("### FEATURE CORRELATION MATRIX")
    corr = numeric.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Tealrose",
        title="Blood Test Feature Correlations"
    )
    st.plotly_chart(
        style_chart(fig, 500),
        width="stretch",
        alt="Correlation heatmap for numeric blood-test features"
    )

# ---------- MODEL PERFORMANCE ----------
elif page == "Model Performance":
    st.subheader("MACHINE LEARNING PERFORMANCE")

    st.info(
        "This evaluation uses a fixed 80/20 split. "
        "Confirm that your saved model was trained with the same "
        "features, preprocessing, and random seed."
    )

    if st.button("RUN MODEL EVALUATION"):
        X = features.copy()

        if "gender" in X.columns:
            if X["gender"].dtype == "object":
                X["gender"] = X["gender"].map(
                    {"Male": 1, "Female": 0}
                )

        if X.isna().any().any():
            st.error("Features contain missing values.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        try:
            predictions = model.predict(X_test)

            a, b, c = st.columns(3)
            a.metric(
                "ACCURACY",
                f"{accuracy_score(y_test, predictions):.1%}"
            )

            report = classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0
            )

            if "1" in report:
                b.metric(
                    "DISEASE CLASS RECALL",
                    f"{report['1']['recall']:.1%}"
                )
                c.metric(
                    "DISEASE CLASS F1",
                    f"{report['1']['f1-score']:.1%}"
                )

            st.markdown("### CONFUSION MATRIX")

            cm = confusion_matrix(
                y_test, predictions, labels=[0, 1]
            )

            fig = px.imshow(
                cm,
                text_auto=True,
                x=["No Disease", "Disease"],
                y=["No Disease", "Disease"],
                color_continuous_scale="Blues",
                title="Actual vs Predicted"
            )
            st.plotly_chart(
                style_chart(fig),
                width="stretch",
                alt="Confusion matrix comparing actual and predicted patient classes"
            )

            st.markdown("### CLASSIFICATION REPORT")
            st.dataframe(
                pd.DataFrame(report).transpose().round(3),
                width="stretch",
                alt="Precision, recall, and F1 scores by class"
            )

        except Exception as e:
            st.error(f"Evaluation failed: {e}")

# ---------- PREDICTION ----------
elif page == "Predict":
    st.subheader("PATIENT FEATURE EXPLORER")
    st.warning(
        "Educational demonstration only. Not a medical diagnosis. "
        "Do not use this output for healthcare decisions."
    )

    feature_names = list(features.columns)

    with st.form("prediction_form"):
        values = {}

        for feature in feature_names:
            if feature == "gender":
                if features[feature].dtype == "object":
                    values[feature] = st.selectbox(
                        "Gender", ["Male", "Female"]
                    )
                else:
                    values[feature] = st.selectbox(
                        "Gender", [0, 1],
                        format_func=lambda x:
                        "Female" if x == 0 else "Male"
                    )
            else:
                series = pd.to_numeric(
                    features[feature], errors="coerce"
                ).dropna()

                default = float(series.median()) if len(series) else 0.0
                values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=default,
                    format="%.2f"
                )

        submitted = st.form_submit_button("ANALYZE SAMPLE")

    if submitted:
        input_df = pd.DataFrame(
            [values],
            columns=feature_names
        )

        if "gender" in input_df.columns:
            if input_df["gender"].dtype == "object":
                input_df["gender"] = input_df["gender"].map(
                    {"Male": 1, "Female": 0}
                )

        try:
            prediction = int(model.predict(input_df)[0])

            st.markdown("### MODEL OUTPUT")
            if prediction == 1:
                st.error("Predicted class: 1")
            else:
                st.success("Predicted class: 0")

            st.caption(
                "Class labels must be verified against the dataset. "
                "This model is for education and is not clinically validated."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption(
    "LIVER INTELLIGENCE • EDUCATIONAL DATA ANALYTICS PROJECT"
)
