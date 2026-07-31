"""
EduPro – Predictive Modeling Dashboard
Course Demand & Revenue Forecasting | Atlantic Recording Corporation / Unified Mentor
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduPro – Course Demand & Revenue Forecasting",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f172a 40%, #1a1040 100%);
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    border-right: 1px solid rgba(139, 92, 246, 0.2);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 50%, #0ea5e9 100%);
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 32px;
    box-shadow: 0 25px 60px rgba(124, 58, 237, 0.35);
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
}
.hero-header h1 {
    font-size: 2.6rem;
    font-weight: 800;
    color: white !important;
    margin: 0 0 8px 0;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
}
.hero-header p {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.85) !important;
    margin: 0;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, rgba(30,27,75,0.8), rgba(15,23,42,0.9));
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}
.metric-card:hover {
    border-color: rgba(139, 92, 246, 0.6);
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(124, 58, 237, 0.25);
}
.metric-card .metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card .metric-label {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: 4px;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.metric-card .metric-icon {
    font-size: 1.8rem;
    margin-bottom: 8px;
}

/* Section headers */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e2e8f0;
    border-left: 4px solid #7c3aed;
    padding-left: 16px;
    margin: 32px 0 20px 0;
}

/* Prediction result card */
.pred-card {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.15), rgba(14, 165, 233, 0.15));
    border: 1px solid rgba(124, 58, 237, 0.4);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(124, 58, 237, 0.2);
}
.pred-card .pred-value {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.pred-card .pred-label {
    font-size: 1rem;
    color: #94a3b8;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15, 23, 42, 0.6);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(139, 92, 246, 0.2);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8 !important;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
}

/* Info box */
.info-box {
    background: rgba(14, 165, 233, 0.1);
    border: 1px solid rgba(14, 165, 233, 0.3);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #7dd3fc;
    font-size: 0.9rem;
}

/* Table */
.dataframe { font-size: 0.85rem !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #7c3aed; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

ARTIFACT_PATH = "Data/edupro_artifacts.pkl"
PLOTLY_TEMPLATE = "plotly_dark"
PURPLE = "#7c3aed"
BLUE = "#4f46e5"
CYAN = "#0ea5e9"
PALETTE = ["#7c3aed", "#4f46e5", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444",
           "#ec4899", "#8b5cf6", "#06b6d4", "#84cc16", "#f97316", "#6366f1"]


# ─────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if not os.path.exists(ARTIFACT_PATH):
        return None
    return joblib.load(ARTIFACT_PATH)


def run_preprocessing():
    """Run preprocess.py to generate artifacts."""
    import subprocess
    result = subprocess.run(
        ["python", "preprocess.py"],
        capture_output=True, text=True
    )
    return result.returncode == 0, result.stdout, result.stderr


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def fmt_number(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:.1f}"


def fmt_currency(n):
    return f"${n:,.2f}"


def make_fig(fig, height=400):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(bgcolor="rgba(15,23,42,0.8)", bordercolor="rgba(139,92,246,0.3)", borderwidth=1),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    return fig


def encode_input(arts, category, course_type, level, price, duration, rating, exp, teacher_rating):
    """Encode a single prediction input using saved label encoders."""
    le = arts["label_encoders"]
    feature_cols = arts["feature_cols"]

    # Price band
    if price <= 150:
        price_band = "Low"
    elif price <= 350:
        price_band = "Medium"
    else:
        price_band = "High"

    if duration <= 15:
        dur_bucket = "Short"
    elif duration <= 30:
        dur_bucket = "Medium"
    else:
        dur_bucket = "Long"

    if rating <= 2.5:
        rat_tier = "Low"
    elif rating <= 3.75:
        rat_tier = "Mid"
    else:
        rat_tier = "High"

    if exp <= 5:
        exp_bucket = "Junior"
    elif exp <= 12:
        exp_bucket = "Mid"
    else:
        exp_bucket = "Senior"

    def safe_encode(col, val):
        classes = list(le[col].classes_)
        if val in classes:
            return le[col].transform([val])[0]
        return 0

    row = {
        "CoursePrice": price,
        "CourseDuration": duration,
        "CourseRating": rating,
        "YearsOfExperience": exp,
        "TeacherRating": teacher_rating,
        "expertise_match": 0,
        "CourseCategory_enc": safe_encode("CourseCategory", category),
        "CourseType_enc": safe_encode("CourseType", course_type),
        "CourseLevel_enc": safe_encode("CourseLevel", level),
        "price_band_enc": safe_encode("price_band", price_band),
        "duration_bucket_enc": safe_encode("duration_bucket", dur_bucket),
        "rating_tier_enc": safe_encode("rating_tier", rat_tier),
        "experience_bucket_enc": safe_encode("experience_bucket", exp_bucket),
    }

    X = pd.DataFrame([row])[feature_cols]
    return X


# ─────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🎓 EduPro Forecasting Intelligence</h1>
    <p>Predictive Modeling for Course Demand & Revenue Forecasting · Unified Mentor × Atlantic Recording Corporation</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD OR GENERATE ARTIFACTS
# ─────────────────────────────────────────────────────────────
arts = load_artifacts()

if arts is None:
    st.warning("⚙️ Model artifacts not found. Generating now — this takes ~10 seconds...")
    with st.spinner("Training models and preparing data..."):
        success, stdout, stderr = run_preprocessing()
    if success:
        st.cache_resource.clear()
        arts = load_artifacts()
        st.success("Models trained successfully!")
    else:
        st.error(f"Preprocessing failed.\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}")
        st.stop()

df = arts["df"]
transactions = arts["transactions"]
courses = arts["courses"]
teachers = arts["teachers"]
users = arts["users"]
monthly_trend = arts["monthly_trend"]
cat_revenue = arts["cat_revenue"]
enroll_results = arts["enroll_results"]
rev_results = arts["rev_results"]
feature_cols = arts["feature_cols"]

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Course Predictor")
    st.markdown("Configure parameters to forecast demand & revenue.")
    st.divider()

    categories = sorted(df["CourseCategory"].unique())
    levels = ["Beginner", "Intermediate", "Advanced"]
    course_types = ["Paid", "Free"]

    sel_category = st.selectbox("Course Category", categories)
    sel_level = st.selectbox("Course Level", levels)
    sel_type = st.selectbox("Course Type", course_types)

    if sel_type == "Free":
        sel_price = 0.0
        st.info("Price is $0 for Free courses.")
    else:
        sel_price = st.slider("Course Price ($)", 50.0, 500.0, 250.0, step=10.0)

    sel_duration = st.slider("Duration (hours)", 1.0, 50.0, 20.0, step=0.5)
    sel_rating = st.slider("Course Rating", 1.0, 5.0, 3.8, step=0.1)

    st.divider()
    st.markdown("#### Instructor Profile")
    sel_exp = st.slider("Years of Experience", 1, 20, 8)
    sel_teacher_rating = st.slider("Teacher Rating", 1.0, 5.0, 3.5, step=0.1)

    st.divider()
    predict_btn = st.button("Run Prediction", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────
# GLOBAL KPI ROW
# ─────────────────────────────────────────────────────────────
total_enrollments = int(transactions.shape[0])
total_revenue = transactions["Amount"].sum()
avg_rating = df["CourseRating"].mean()
n_courses = len(courses)
n_categories = df["CourseCategory"].nunique()
paid_courses = len(df[df["CourseType"] == "Paid"])

col1, col2, col3, col4, col5, col6 = st.columns(6)
kpis = [
    ("🎓", f"{total_enrollments:,}", "Total Enrollments"),
    ("💰", f"${total_revenue:,.0f}", "Total Revenue"),
    ("📚", str(n_courses), "Total Courses"),
    ("🏷️", str(n_categories), "Categories"),
    ("⭐", f"{avg_rating:.2f}", "Avg Course Rating"),
    ("💳", str(paid_courses), "Paid Courses"),
]
for col, (icon, val, label) in zip([col1, col2, col3, col4, col5, col6], kpis):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "EDA Overview",
    "Demand Predictor",
    "Revenue Forecast",
    "Feature Importance",
    "Category Comparison",
])

# ═════════════════════════════════════════════════════════════
# TAB 1 – EDA OVERVIEW
# ═════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # Category enrollment distribution
        cat_enroll = df.groupby("CourseCategory")["enrollment_count"].sum().reset_index().sort_values("enrollment_count", ascending=True)
        fig = px.bar(cat_enroll, x="enrollment_count", y="CourseCategory", orientation="h",
                     title="Total Enrollments by Category",
                     color="enrollment_count", color_continuous_scale=["#4f46e5", "#7c3aed", "#0ea5e9"])
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(make_fig(fig, 420), use_container_width=True)

    with c2:
        # Course type pie
        type_counts = df["CourseType"].value_counts().reset_index()
        type_counts.columns = ["CourseType", "count"]
        fig2 = px.pie(type_counts, names="CourseType", values="count",
                      title="Paid vs Free Courses",
                      color_discrete_sequence=[PURPLE, CYAN],
                      hole=0.55)
        fig2.update_traces(textinfo="label+percent", pull=[0.04, 0])
        st.plotly_chart(make_fig(fig2, 420), use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        # Price vs Enrollment scatter
        fig3 = px.scatter(df, x="CoursePrice", y="enrollment_count",
                          color="CourseCategory", size="CourseRating",
                          hover_name="CourseName",
                          title="Course Price vs Enrollment Count",
                          color_discrete_sequence=PALETTE,
                          labels={"CoursePrice": "Price ($)", "enrollment_count": "Enrollments"})
        st.plotly_chart(make_fig(fig3, 420), use_container_width=True)

    with c4:
        # Rating distribution
        fig4 = px.histogram(df, x="CourseRating", nbins=20, color="CourseLevel",
                            title="Course Rating Distribution by Level",
                            color_discrete_sequence=[PURPLE, CYAN, "#10b981"],
                            barmode="overlay",
                            labels={"CourseRating": "Rating"})
        fig4.update_traces(opacity=0.75)
        st.plotly_chart(make_fig(fig4, 420), use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        # Revenue by category
        cat_rev_sorted = cat_revenue.sort_values("category_total_revenue", ascending=True)
        fig5 = px.bar(cat_rev_sorted, x="category_total_revenue", y="CourseCategory", orientation="h",
                      title="Total Revenue by Course Category",
                      color="category_total_revenue",
                      color_continuous_scale=["#4f46e5", "#7c3aed", "#f59e0b"])
        fig5.update_coloraxes(showscale=False)
        st.plotly_chart(make_fig(fig5, 420), use_container_width=True)

    with c6:
        # Level distribution
        level_df = df.groupby(["CourseCategory", "CourseLevel"]).size().reset_index(name="count")
        fig6 = px.bar(level_df, x="CourseCategory", y="count", color="CourseLevel",
                      title="Course Level Distribution by Category",
                      color_discrete_sequence=[PURPLE, CYAN, "#10b981"],
                      barmode="stack",
                      labels={"CourseCategory": "Category", "count": "Courses"})
        fig6.update_xaxes(tickangle=45)
        st.plotly_chart(make_fig(fig6, 420), use_container_width=True)

    # Dataset sample
    st.markdown('<div class="section-header">Dataset Sample</div>', unsafe_allow_html=True)
    display_cols = ["CourseName", "CourseCategory", "CourseType", "CourseLevel",
                    "CoursePrice", "CourseDuration", "CourseRating",
                    "enrollment_count", "total_revenue"]
    st.dataframe(
        df[display_cols].rename(columns={
            "CourseName": "Course", "CourseCategory": "Category",
            "CourseType": "Type", "CourseLevel": "Level",
            "CoursePrice": "Price ($)", "CourseDuration": "Duration (h)",
            "CourseRating": "Rating", "enrollment_count": "Enrollments",
            "total_revenue": "Revenue ($)"
        }).round(2),
        use_container_width=True, height=350
    )

# ═════════════════════════════════════════════════════════════
# TAB 2 – DEMAND PREDICTOR
# ═════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Course Demand & Revenue Predictor</div>', unsafe_allow_html=True)

    if predict_btn or True:
        X_input = encode_input(
            arts, sel_category, sel_type, sel_level,
            sel_price, sel_duration, sel_rating, sel_exp, sel_teacher_rating
        )

        best_enroll_model = arts["enroll_models"][arts["best_enroll_name"]]
        best_rev_model = arts["rev_models"][arts["best_rev_name"]]

        pred_enroll = max(0, best_enroll_model.predict(X_input)[0])
        pred_revenue = max(0, best_rev_model.predict(X_input)[0])

        # Show prediction cards
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.markdown(f"""
            <div class="pred-card">
                <div class="pred-label">Predicted Enrollments</div>
                <div class="pred-value">{int(pred_enroll):,}</div>
                <div class="pred-label">students expected</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="pred-card">
                <div class="pred-label">Predicted Revenue</div>
                <div class="pred-value">${pred_revenue:,.0f}</div>
                <div class="pred-label">total course revenue</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            rev_per_enroll = pred_revenue / pred_enroll if pred_enroll > 0 else 0
            st.markdown(f"""
            <div class="pred-card">
                <div class="pred-label">Revenue per Student</div>
                <div class="pred-value">${rev_per_enroll:,.2f}</div>
                <div class="pred-label">monetization efficiency</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Compare against similar courses
        st.markdown('<div class="section-header">How It Compares to Similar Courses</div>', unsafe_allow_html=True)
        similar = df[df["CourseCategory"] == sel_category].copy()

        if len(similar) > 0:
            c1, c2 = st.columns(2)
            with c1:
                avg_enroll = similar["enrollment_count"].mean()
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=["Category Avg", "Your Prediction"],
                    y=[avg_enroll, pred_enroll],
                    marker_color=[BLUE, PURPLE],
                    text=[f"{avg_enroll:.0f}", f"{pred_enroll:.0f}"],
                    textposition="outside"
                ))
                fig.update_layout(title=f"Enrollments vs {sel_category} Avg")
                st.plotly_chart(make_fig(fig, 350), use_container_width=True)
            with c2:
                avg_rev = similar["total_revenue"].mean()
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=["Category Avg", "Your Prediction"],
                    y=[avg_rev, pred_revenue],
                    marker_color=[CYAN, "#10b981"],
                    text=[f"${avg_rev:,.0f}", f"${pred_revenue:,.0f}"],
                    textposition="outside"
                ))
                fig2.update_layout(title=f"Revenue vs {sel_category} Avg")
                st.plotly_chart(make_fig(fig2, 350), use_container_width=True)

            # Scatter: show where prediction lands among real courses
            fig3 = px.scatter(similar, x="CoursePrice", y="enrollment_count",
                              hover_name="CourseName",
                              color="CourseLevel",
                              color_discrete_sequence=[PURPLE, CYAN, "#10b981"],
                              title=f"Your Course vs Existing {sel_category} Courses",
                              labels={"CoursePrice": "Price ($)", "enrollment_count": "Enrollments"})
            fig3.add_trace(go.Scatter(
                x=[sel_price], y=[pred_enroll],
                mode="markers",
                marker=dict(size=18, color="#f59e0b", symbol="star", line=dict(width=2, color="white")),
                name="Your Course (Predicted)",
            ))
            st.plotly_chart(make_fig(fig3, 400), use_container_width=True)

    # Model performance summary
    st.markdown('<div class="section-header">Model Performance Summary</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Enrollment Prediction Models**")
        st.dataframe(
            enroll_results[["Model", "MAE", "RMSE", "R2"]].sort_values("R2", ascending=False).reset_index(drop=True),
            use_container_width=True
        )
    with c2:
        st.markdown("**Revenue Prediction Models**")
        st.dataframe(
            rev_results[["Model", "MAE", "RMSE", "R2"]].sort_values("R2", ascending=False).reset_index(drop=True),
            use_container_width=True
        )

# ═════════════════════════════════════════════════════════════
# TAB 3 – REVENUE FORECAST
# ═════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Revenue Forecasting Analytics</div>', unsafe_allow_html=True)

    # Monthly trend
    monthly_trend["YearMonth"] = monthly_trend["YearMonth"].astype(str)
    monthly_trend_sorted = monthly_trend.sort_values("YearMonth")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=monthly_trend_sorted["YearMonth"], y=monthly_trend_sorted["monthly_revenue"],
        name="Monthly Revenue", marker_color=PURPLE, opacity=0.8
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=monthly_trend_sorted["YearMonth"], y=monthly_trend_sorted["monthly_enrollments"],
        name="Enrollments", line=dict(color=CYAN, width=3), mode="lines+markers",
        marker=dict(size=7)
    ), secondary_y=True)
    fig.update_layout(title="Monthly Revenue & Enrollment Trends (2025)", template=PLOTLY_TEMPLATE,
                      height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter", color="#e2e8f0"),
                      legend=dict(bgcolor="rgba(15,23,42,0.8)"),
                      margin=dict(l=20, r=20, t=50, b=20))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
    fig.update_yaxes(title_text="Enrollments", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        # Category revenue
        fig2 = px.bar(cat_revenue.sort_values("category_total_revenue", ascending=False),
                      x="CourseCategory", y="category_total_revenue",
                      title="Total Revenue by Category",
                      color="category_total_revenue",
                      color_continuous_scale=["#4f46e5", "#7c3aed", "#f59e0b"],
                      labels={"category_total_revenue": "Revenue ($)", "CourseCategory": "Category"})
        fig2.update_coloraxes(showscale=False)
        fig2.update_xaxes(tickangle=45)
        st.plotly_chart(make_fig(fig2, 420), use_container_width=True)

    with c2:
        # Top 10 courses by revenue
        top10_rev = df.nlargest(10, "total_revenue")[["CourseName", "CourseCategory", "total_revenue", "enrollment_count"]]
        fig3 = px.bar(top10_rev.sort_values("total_revenue"), x="total_revenue", y="CourseName",
                      orientation="h", title="Top 10 Courses by Revenue",
                      color="CourseCategory", color_discrete_sequence=PALETTE,
                      labels={"total_revenue": "Revenue ($)", "CourseName": "Course"})
        st.plotly_chart(make_fig(fig3, 420), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        # Revenue vs Enrollments scatter (all courses)
        fig4 = px.scatter(df, x="enrollment_count", y="total_revenue",
                          color="CourseCategory", hover_name="CourseName",
                          size="CoursePrice",
                          title="Revenue vs Enrollments (All Courses)",
                          color_discrete_sequence=PALETTE,
                          labels={"enrollment_count": "Enrollments", "total_revenue": "Revenue ($)"})
        st.plotly_chart(make_fig(fig4, 420), use_container_width=True)

    with c4:
        # Revenue per enrollment by level
        df["rev_per_enroll"] = np.where(df["enrollment_count"] > 0, df["total_revenue"] / df["enrollment_count"], 0)
        fig5 = px.box(df, x="CourseLevel", y="rev_per_enroll",
                      color="CourseType",
                      title="Revenue per Enrollment by Level & Type",
                      color_discrete_sequence=[PURPLE, CYAN],
                      labels={"rev_per_enroll": "Revenue/Enrollment ($)", "CourseLevel": "Level"})
        st.plotly_chart(make_fig(fig5, 420), use_container_width=True)

    # Payment method breakdown
    st.markdown('<div class="section-header">Transaction Analytics</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        pay_breakdown = transactions.groupby("PaymentMethod")["Amount"].sum().reset_index()
        pay_breakdown.columns = ["PaymentMethod", "TotalRevenue"]
        fig6 = px.pie(pay_breakdown, names="PaymentMethod", values="TotalRevenue",
                      title="Revenue by Payment Method",
                      color_discrete_sequence=PALETTE, hole=0.5)
        st.plotly_chart(make_fig(fig6, 380), use_container_width=True)
    with c6:
        transactions["Month"] = pd.to_datetime(transactions["TransactionDate"]).dt.month_name()
        month_order = ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]
        monthly_pay = transactions.groupby(["Month", "PaymentMethod"])["Amount"].sum().reset_index()
        monthly_pay["Month"] = pd.Categorical(monthly_pay["Month"], categories=month_order, ordered=True)
        monthly_pay = monthly_pay.sort_values("Month")
        fig7 = px.bar(monthly_pay, x="Month", y="Amount", color="PaymentMethod",
                      barmode="stack", title="Monthly Revenue by Payment Method",
                      color_discrete_sequence=PALETTE)
        fig7.update_xaxes(tickangle=45)
        st.plotly_chart(make_fig(fig7, 380), use_container_width=True)

# ═════════════════════════════════════════════════════════════
# TAB 4 – FEATURE IMPORTANCE
# ═════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Feature Importance Explorer</div>', unsafe_allow_html=True)

    friendly_names = {
        "CoursePrice": "Course Price",
        "CourseDuration": "Course Duration",
        "CourseRating": "Course Rating",
        "YearsOfExperience": "Instructor Experience",
        "TeacherRating": "Instructor Rating",
        "expertise_match": "Expertise Match",
        "CourseCategory_enc": "Course Category",
        "CourseType_enc": "Course Type",
        "CourseLevel_enc": "Course Level",
        "price_band_enc": "Price Band",
        "duration_bucket_enc": "Duration Bucket",
        "rating_tier_enc": "Rating Tier",
        "experience_bucket_enc": "Experience Bucket",
    }

    c1, c2 = st.columns(2)

    for col, imp_key, title, color_scale in [
        (c1, "enroll_feat_imp", "Enrollment Prediction Drivers", ["#4f46e5", "#a78bfa"]),
        (c2, "rev_feat_imp", "Revenue Prediction Drivers", ["#0ea5e9", "#10b981"]),
    ]:
        with col:
            imp = arts[imp_key]
            if imp is not None:
                feat_imp_df = pd.DataFrame({
                    "Feature": [friendly_names.get(f, f) for f in feature_cols],
                    "Importance": imp,
                }).sort_values("Importance", ascending=True).tail(13)

                fig = px.bar(feat_imp_df, x="Importance", y="Feature", orientation="h",
                             title=title,
                             color="Importance", color_continuous_scale=color_scale)
                fig.update_coloraxes(showscale=False)
                st.plotly_chart(make_fig(fig, 500), use_container_width=True)
            else:
                st.info(f"Feature importance not available for {title} (linear model selected as best).")

    # Correlation heatmap
    st.markdown('<div class="section-header">Feature Correlation Matrix</div>', unsafe_allow_html=True)
    num_cols = ["CoursePrice", "CourseDuration", "CourseRating",
                "YearsOfExperience", "TeacherRating", "enrollment_count", "total_revenue"]
    corr = df[num_cols].corr().round(3)
    friendly = {
        "CoursePrice": "Price", "CourseDuration": "Duration", "CourseRating": "Crs Rating",
        "YearsOfExperience": "Exp Years", "TeacherRating": "Tchr Rating",
        "enrollment_count": "Enrollments", "total_revenue": "Revenue"
    }
    corr.index = [friendly[c] for c in corr.index]
    corr.columns = [friendly[c] for c in corr.columns]

    fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                         color_continuous_scale=["#0f172a", "#4f46e5", "#7c3aed", "#a78bfa"],
                         title="Feature Correlation Heatmap")
    st.plotly_chart(make_fig(fig_corr, 500), use_container_width=True)

    # Business insight callouts
    st.markdown('<div class="section-header">Key Insights from Feature Importance</div>', unsafe_allow_html=True)
    insights = [
        ("💰 Price Sensitivity", "Course Price is among the most influential features. Higher-priced paid courses drive disproportionate revenue but can suppress enrollment volume."),
        ("⭐ Rating Matters", "Both Course Rating and Teacher Rating show strong predictive power — quality signals significantly affect expected demand."),
        ("🏷️ Level & Category Effects", "Course Level (Beginner/Intermediate/Advanced) and Category jointly shape baseline enrollment expectations — category-specific benchmarks are essential for planning."),
        ("⏱️ Duration Signal", "Course Duration moderately predicts enrollment; shorter courses attract higher volume, while longer courses generate higher per-student revenue."),
    ]
    cols = st.columns(2)
    for i, (title, body) in enumerate(insights):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="metric-card" style="text-align:left; padding:20px;">
                <div style="font-weight:700; font-size:1rem; color:#a78bfa; margin-bottom:8px;">{title}</div>
                <div style="font-size:0.88rem; color:#cbd5e1; line-height:1.6;">{body}</div>
            </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# TAB 5 – CATEGORY COMPARISON
# ═════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Category-Level Demand & Revenue Comparison</div>', unsafe_allow_html=True)

    cat_summary = df.groupby("CourseCategory").agg(
        total_enrollments=("enrollment_count", "sum"),
        total_revenue=("total_revenue", "sum"),
        avg_price=("CoursePrice", "mean"),
        avg_rating=("CourseRating", "mean"),
        n_courses=("CourseID", "count"),
        avg_duration=("CourseDuration", "mean"),
    ).reset_index().sort_values("total_enrollments", ascending=False)

    cat_summary["rev_per_enrollment"] = cat_summary["total_revenue"] / cat_summary["total_enrollments"]

    # Radar chart
    radar_cats = cat_summary["CourseCategory"].tolist()
    metrics_to_plot = ["total_enrollments", "total_revenue", "avg_price", "avg_rating"]
    metric_labels = ["Enrollments", "Revenue", "Avg Price", "Avg Rating"]

    # Normalize for radar
    radar_df = cat_summary.copy()
    for m in metrics_to_plot:
        radar_df[m + "_norm"] = (radar_df[m] - radar_df[m].min()) / (radar_df[m].max() - radar_df[m].min() + 1e-9)

    # Select top 6 categories for radar
    top6 = cat_summary.nlargest(6, "total_enrollments")["CourseCategory"].tolist()
    fig_radar = go.Figure()
    for i, cat in enumerate(top6):
        row = radar_df[radar_df["CourseCategory"] == cat].iloc[0]
        values = [row[m + "_norm"] for m in metrics_to_plot]
        values += [values[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_labels + [metric_labels[0]],
            fill="toself",
            name=cat,
            line_color=PALETTE[i],
            opacity=0.7
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(15,23,42,0.6)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="#94a3b8")),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="#e2e8f0")),
        ),
        title="Category Performance Radar (Top 6 by Enrollment)",
        template=PLOTLY_TEMPLATE,
        height=500, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        legend=dict(bgcolor="rgba(15,23,42,0.8)", bordercolor="rgba(139,92,246,0.3)", borderwidth=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(cat_summary.sort_values("total_enrollments", ascending=False),
                     x="CourseCategory", y="total_enrollments",
                     color="total_enrollments", title="Total Enrollments by Category",
                     color_continuous_scale=["#4f46e5", "#7c3aed", "#a78bfa"],
                     labels={"total_enrollments": "Enrollments", "CourseCategory": "Category"})
        fig.update_coloraxes(showscale=False)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(make_fig(fig, 400), use_container_width=True)

    with c2:
        fig2 = px.bar(cat_summary.sort_values("rev_per_enrollment", ascending=False),
                      x="CourseCategory", y="rev_per_enrollment",
                      color="rev_per_enrollment", title="Revenue per Enrollment by Category",
                      color_continuous_scale=["#0ea5e9", "#10b981", "#f59e0b"],
                      labels={"rev_per_enrollment": "Rev/Enrollment ($)", "CourseCategory": "Category"})
        fig2.update_coloraxes(showscale=False)
        fig2.update_xaxes(tickangle=45)
        st.plotly_chart(make_fig(fig2, 400), use_container_width=True)

    # Bubble chart
    fig3 = px.scatter(cat_summary, x="avg_price", y="total_enrollments",
                      size="total_revenue", color="CourseCategory",
                      hover_name="CourseCategory",
                      color_discrete_sequence=PALETTE,
                      title="Category Bubble Chart: Price vs Enrollments vs Revenue",
                      labels={"avg_price": "Avg Price ($)", "total_enrollments": "Total Enrollments",
                              "total_revenue": "Revenue (bubble size)"})
    st.plotly_chart(make_fig(fig3, 480), use_container_width=True)

    # Summary table
    st.markdown('<div class="section-header">Category Performance Summary Table</div>', unsafe_allow_html=True)
    display_cat = cat_summary.rename(columns={
        "CourseCategory": "Category", "total_enrollments": "Total Enrollments",
        "total_revenue": "Total Revenue ($)", "avg_price": "Avg Price ($)",
        "avg_rating": "Avg Rating", "n_courses": "# Courses",
        "avg_duration": "Avg Duration (h)", "rev_per_enrollment": "Rev/Student ($)"
    }).round(2)
    st.dataframe(display_cat, use_container_width=True, height=420)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#475569; font-size:0.82rem; padding:20px; border-top: 1px solid rgba(139,92,246,0.15);">
    EduPro Forecasting Intelligence · Unified Mentor × Atlantic Recording Corporation · Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
