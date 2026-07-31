"""
EduPro – Preprocessing & Model Training Pipeline
Merges all sheets, engineers features, trains models, saves artifacts.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "Data/EduPro Online Platform.xlsx"
OUTPUT_DIR = "Data"

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
def load_data():
    xl = pd.ExcelFile(DATA_PATH)
    users = xl.parse("Users")
    teachers = xl.parse("Teachers")
    courses = xl.parse("Courses")
    transactions = xl.parse("Transactions")
    transactions["TransactionDate"] = pd.to_datetime(transactions["TransactionDate"])
    return users, teachers, courses, transactions


# ─────────────────────────────────────────────────────────────
# 2. AGGREGATE TRANSACTIONS AT COURSE LEVEL
# ─────────────────────────────────────────────────────────────
def aggregate_transactions(transactions, courses):
    tx = transactions.copy()
    tx["Month"] = tx["TransactionDate"].dt.month
    tx["YearMonth"] = tx["TransactionDate"].dt.to_period("M").astype(str)

    agg = tx.groupby("CourseID").agg(
        enrollment_count=("TransactionID", "count"),
        total_revenue=("Amount", "sum"),
        avg_monthly_enrollments=("TransactionID", lambda x: x.count() / tx["Month"].nunique()),
        avg_monthly_revenue=("Amount", lambda x: x.sum() / tx["Month"].nunique()),
        unique_users=("UserID", "nunique"),
    ).reset_index()

    # Monthly breakdown (for trend charts)
    monthly = tx.groupby(["CourseID", "YearMonth"]).agg(
        monthly_enroll=("TransactionID", "count"),
        monthly_rev=("Amount", "sum"),
    ).reset_index()

    return agg, monthly, tx


# ─────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def engineer_features(courses, teachers, transactions, agg):
    # Join teacher to transactions (mode teacher per course)
    teacher_per_course = (
        transactions.groupby("CourseID")["TeacherID"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
        .rename(columns={"TeacherID": "TeacherID"})
    )

    df = courses.merge(agg, on="CourseID", how="left")
    df = df.merge(teacher_per_course, on="CourseID", how="left")
    df = df.merge(
        teachers[["TeacherID", "Expertise", "YearsOfExperience", "TeacherRating"]],
        on="TeacherID",
        how="left",
    )

    # Fill missing aggregates with 0
    for col in ["enrollment_count", "total_revenue", "avg_monthly_enrollments", "avg_monthly_revenue", "unique_users"]:
        df[col] = df[col].fillna(0)

    # Price band
    df["price_band"] = pd.cut(
        df["CoursePrice"],
        bins=[-1, 150, 350, 1000],
        labels=["Low", "Medium", "High"],
    )

    # Duration bucket
    df["duration_bucket"] = pd.cut(
        df["CourseDuration"],
        bins=[0, 15, 30, 200],
        labels=["Short", "Medium", "Long"],
    )

    # Rating tier
    df["rating_tier"] = pd.cut(
        df["CourseRating"],
        bins=[0, 2.5, 3.75, 5.1],
        labels=["Low", "Mid", "High"],
    )

    # Experience bucket
    df["experience_bucket"] = pd.cut(
        df["YearsOfExperience"],
        bins=[0, 5, 12, 100],
        labels=["Junior", "Mid", "Senior"],
    )

    # Expertise-category match
    df["expertise_match"] = (df["Expertise"] == df["CourseCategory"]).astype(int)

    # Revenue per enrollment
    df["revenue_per_enrollment"] = np.where(
        df["enrollment_count"] > 0,
        df["total_revenue"] / df["enrollment_count"],
        0,
    )

    return df


# ─────────────────────────────────────────────────────────────
# 4. ENCODE FOR MODELING
# ─────────────────────────────────────────────────────────────
def prepare_model_features(df):
    feature_cols = [
        "CoursePrice", "CourseDuration", "CourseRating",
        "YearsOfExperience", "TeacherRating", "expertise_match",
    ]
    cat_cols = ["CourseCategory", "CourseType", "CourseLevel", "price_band", "duration_bucket", "rating_tier", "experience_bucket"]

    df_enc = df.copy()
    label_encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col + "_enc"] = le.fit_transform(df_enc[col].astype(str))
        label_encoders[col] = le
        feature_cols.append(col + "_enc")

    return df_enc, feature_cols, label_encoders


# ─────────────────────────────────────────────────────────────
# 5. TRAIN MODELS
# ─────────────────────────────────────────────────────────────
def train_and_evaluate(X, y, target_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=5000),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    }

    results = []
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        results.append({"Model": name, "Target": target_name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)})
        trained[name] = model
        print(f"  [{target_name}] {name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    # Best model by R2
    best_name = max(results, key=lambda x: x["R2"])["Model"]
    best_model = trained[best_name]
    print(f"  >> Best model for {target_name}: {best_name} (R2={max(results, key=lambda x: x['R2'])['R2']})")

    # Feature importance
    feat_imp = None
    if hasattr(best_model, "feature_importances_"):
        feat_imp = best_model.feature_importances_

    return results, trained, best_name, feat_imp


# ─────────────────────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    users, teachers, courses, transactions = load_data()

    print("Aggregating transactions...")
    agg, monthly, tx = aggregate_transactions(transactions, courses)

    print("Engineering features...")
    df = engineer_features(courses, teachers, transactions, agg)

    print("Preparing model features...")
    df_enc, feature_cols, label_encoders = prepare_model_features(df)

    X = df_enc[feature_cols].fillna(0)

    # Target 1: Enrollment Count
    y_enroll = df_enc["enrollment_count"]
    print("\nTraining Enrollment models...")
    enroll_results, enroll_models, best_enroll, enroll_imp = train_and_evaluate(X, y_enroll, "Enrollment")

    # Target 2: Course Revenue
    y_revenue = df_enc["total_revenue"]
    print("\nTraining Revenue models...")
    rev_results, rev_models, best_rev, rev_imp = train_and_evaluate(X, y_revenue, "Revenue")

    # Category Revenue (aggregated)
    cat_revenue = df.groupby("CourseCategory")["total_revenue"].sum().reset_index()
    cat_revenue.columns = ["CourseCategory", "category_total_revenue"]

    # Monthly global trends
    monthly_global = transactions.copy()
    monthly_global["YearMonth"] = monthly_global["TransactionDate"].dt.to_period("M").astype(str)
    monthly_trend = monthly_global.groupby("YearMonth").agg(
        monthly_enrollments=("TransactionID", "count"),
        monthly_revenue=("Amount", "sum"),
    ).reset_index()

    # Save everything
    artifacts = {
        "df": df,
        "df_enc": df_enc,
        "feature_cols": feature_cols,
        "label_encoders": label_encoders,
        "X": X,
        "enroll_models": enroll_models,
        "rev_models": rev_models,
        "best_enroll_name": best_enroll,
        "best_rev_name": best_rev,
        "enroll_results": pd.DataFrame(enroll_results),
        "rev_results": pd.DataFrame(rev_results),
        "enroll_feat_imp": enroll_imp,
        "rev_feat_imp": rev_imp,
        "monthly": monthly,
        "monthly_trend": monthly_trend,
        "cat_revenue": cat_revenue,
        "transactions": transactions,
        "teachers": teachers,
        "courses": courses,
        "users": users,
    }

    out_path = os.path.join(OUTPUT_DIR, "edupro_artifacts.pkl")
    joblib.dump(artifacts, out_path)
    print(f"\nArtifacts saved to {out_path}")

    eval_df = pd.DataFrame(enroll_results + rev_results)
    eval_df.to_csv(os.path.join(OUTPUT_DIR, "model_evaluation.csv"), index=False)
    print("Model evaluation saved to Data/model_evaluation.csv")


if __name__ == "__main__":
    main()
