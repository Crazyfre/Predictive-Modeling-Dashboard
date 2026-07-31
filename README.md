# EduPro – Predictive Modeling Dashboard

An interactive Streamlit-based predictive modeling dashboard designed for course demand and revenue forecasting on the EduPro Online Platform. 

## Features

- **Course Demand Forecasting:** Predicts enrollment counts based on course duration, category, rating, and teacher metrics.
- **Revenue Estimation:** Projects potential earnings for current and new courses.
- **Model Evaluation Dashboard:** View details on model performance (Random Forest, Gradient Boosting, Linear Regression) including MAE, MSE, and R² scores.
- **Interactive Scenarios:** Simulation tools to see how changes in course pricing, rating, or duration affect expected revenue and enrollments.

## Repository Structure

- `app.py`: Streamlit dashboard application containing visualization and user interfaces.
- `preprocess.py`: Machine learning pipeline to clean raw data, perform feature engineering, train regressor models, and save evaluation metrics.
- `requirements.txt`: Python package dependencies.
- `.gitignore`: Configured to exclude data files and project documentation.

## Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies:**
   Make sure you have Python 3.8+ installed. Run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Provide Raw Data (Required):**
   Place your raw dataset file `EduPro Online Platform.xlsx` under a `Data/` directory:
   ```text
   Data/EduPro Online Platform.xlsx
   ```

4. **Run Preprocessing & Model Training:**
   To train models and generate required evaluation artifacts:
   ```bash
   python preprocess.py
   ```

5. **Launch the Dashboard:**
   Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
