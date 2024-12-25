import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set a more attractive Seaborn style/theme
sns.set_theme(style='whitegrid', palette='pastel')

# Helper Functions
def read_csv_data(uploaded_file) -> pd.DataFrame:
    """
    Reads a CSV file containing at least two columns: 'StudentID' and 'Score'.
    Returns a pandas DataFrame or raises an Exception if the file is missing columns.
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        raise Exception(f"An error occurred while reading the CSV: {e}")

    required_cols = {'StudentID', 'Score'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV file must contain columns: {required_cols}")

    return df


def assign_absolute_grade(score, thresholds=None):
    """
    Assigns an absolute letter grade based on fixed numeric thresholds.
    """
    if thresholds is None:
        thresholds = {'A': 90, 'B': 80, 'C': 70, 'D': 60}

    if score >= thresholds['A']:
        return 'A'
    elif score >= thresholds['B']:
        return 'B'
    elif score >= thresholds['C']:
        return 'C'
    elif score >= thresholds['D']:
        return 'D'
    else:
        return 'F'


def transform_scores_normal_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs z-score scaling (normal-curve approach) on 'Score'.
    """
    df_new = df.copy()
    mu = df['Score'].mean()
    sigma = df['Score'].std()

    if sigma == 0:
        # All scores are identical; no transformation
        df_new['AdjustedScore'] = df['Score']
        return df_new

    # Standard z-score
    z_scores = (df['Score'] - mu) / sigma
    df_new['AdjustedScore'] = z_scores

    return df_new


def assign_letter_grades_from_percentiles(df: pd.DataFrame, grade_col='FinalGrade') -> pd.DataFrame:
    """
    Assign letter grades based on the percentile of 'AdjustedScore' in a normal distribution.
    """
    df_new = df.copy()
    if 'AdjustedScore' not in df_new.columns:
        df_new['AdjustedScore'] = df_new['Score']

    z_scores = df_new['AdjustedScore']
    percentiles = norm.cdf(z_scores)

    letter_bins = {'A': 0.80, 'B': 0.50, 'C': 0.20, 'D': 0.10, 'F': 0.00}
    letter_grades = []
    for p in percentiles:
        if p >= letter_bins['A']:
            letter_grades.append('A')
        elif p >= letter_bins['B']:
            letter_grades.append('B')
        elif p >= letter_bins['C']:
            letter_grades.append('C')
        elif p >= letter_bins['D']:
            letter_grades.append('D')
        else:
            letter_grades.append('F')

    df_new[grade_col] = letter_grades
    return df_new


# Main Streamlit App
def main():
    st.title("Grading System: Absolute vs Relative Grading")

    st.markdown("""
    This app lets you choose between **absolute grading** and **relative grading**:
    - Upload a CSV file with at least two columns: `StudentID` and `Score`.
    - Select the grading method.
    - View the results and grade distribution.
    """)

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file with columns ['StudentID', 'Score']", type=["csv"])
    if not uploaded_file:
        st.warning("Please upload a CSV file to proceed.")
        return

    # Read CSV data
    df = read_csv_data(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Select Grading Method
    grading_method = st.selectbox("Choose a Grading Method", ["Absolute Grading", "Relative Grading"])
    
    if grading_method == "Absolute Grading":
        # Absolute Grading
        st.subheader("Absolute Grading")
        thresholds = {'A': 90, 'B': 80, 'C': 70, 'D': 60}
        df['Grade'] = df['Score'].apply(assign_absolute_grade, thresholds=thresholds)
        st.write("Grades assigned based on absolute thresholds.")
        st.dataframe(df[['StudentID', 'Score', 'Grade']].head())
        
        # Display Grade Distribution
        grade_counts = df['Grade'].value_counts().sort_index()
        st.bar_chart(grade_counts)

    elif grading_method == "Relative Grading":
        # Relative Grading
        st.subheader("Relative Grading")
        
        # Transform scores to z-scores
        df_transformed = transform_scores_normal_curve(df)
        df_grades = assign_letter_grades_from_percentiles(df_transformed)
        
        st.write("Grades assigned based on percentile ranks (relative grading).")
        st.dataframe(df_grades[['StudentID', 'Score', 'AdjustedScore', 'FinalGrade']].head())
        
        # Display Grade Distribution
        grade_counts = df_grades['FinalGrade'].value_counts().sort_index()
        st.bar_chart(grade_counts)

    st.success("Grading completed successfully!")


if __name__ == '__main__':
    main()
