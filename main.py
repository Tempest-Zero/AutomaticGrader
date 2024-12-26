import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Grading System App",
    page_icon="📊",
    layout="centered"
)
sns.set_theme(style='whitegrid', palette='pastel')

# -----------------------------
# Helper Functions
# -----------------------------
def read_csv_data(uploaded_file) -> pd.DataFrame:
    """
    Reads a CSV file containing 'StudentID' and 'Score'.
    Raises an exception if columns are missing or file read fails.
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        raise Exception(f"Error while reading the CSV: {e}")

    required_cols = {'StudentID', 'Score'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must have columns {required_cols}")
    return df

def assign_absolute_grade(score, thresholds=None):
    """
    Assign an absolute letter grade by numeric thresholds.
    Defaults: A>=90, B>=80, C>=70, D>=60, else F.
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
    Converts 'Score' into z-scores (AdjustedScore).
    If all scores are identical, no transformation is done.
    """
    df_new = df.copy()
    mean_ = df['Score'].mean()
    std_ = df['Score'].std()

    if std_ == 0:
        # All scores identical, no transformation
        df_new['AdjustedScore'] = df['Score']
    else:
        df_new['AdjustedScore'] = (df['Score'] - mean_) / std_
    return df_new

def assign_letter_grades_from_percentiles(df: pd.DataFrame, grade_col='FinalGrade') -> pd.DataFrame:
    """
    Assigns letter grades (A,B,C,D,F) by percentile cutoffs in a normal distribution.
    """
    df_new = df.copy()
    if 'AdjustedScore' not in df_new.columns:
        df_new['AdjustedScore'] = df_new['Score']

    z_scores = df_new['AdjustedScore']
    # Convert z-scores to percentiles
    percentiles = norm.cdf(z_scores)

    # Typical percentile cutoffs
    cutoffs = {'A': 0.80, 'B': 0.50, 'C': 0.20, 'D': 0.10, 'F': 0.00}
    letter_grades = []
    for p in percentiles:
        if p >= cutoffs['A']:
            letter_grades.append('A')
        elif p >= cutoffs['B']:
            letter_grades.append('B')
        elif p >= cutoffs['C']:
            letter_grades.append('C')
        elif p >= cutoffs['D']:
            letter_grades.append('D')
        else:
            letter_grades.append('F')

    df_new[grade_col] = letter_grades
    return df_new

def plot_distribution(df, col='Score', title='Score Distribution'):
    """
    Plots a histogram + KDE + (optional) normal PDF for the chosen column.
    """
    fig, ax = plt.subplots(figsize=(7,4))
    sns.histplot(df[col], bins=15, stat='density', color='skyblue',
                 alpha=0.6, edgecolor='black', label='Histogram', ax=ax)
    sns.kdeplot(df[col], color='blue', linewidth=2, label='KDE', ax=ax)

    mean_val = df[col].mean()
    std_val = df[col].std()
    if std_val > 0:
        # Theoretical Normal PDF
        x_vals = np.linspace(df[col].min(), df[col].max(), 200)
        pdf_vals = norm.pdf(x_vals, loc=mean_val, scale=std_val)
        ax.plot(x_vals, pdf_vals, 'r--', lw=2, label='Normal PDF')

    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)

def plot_grade_distribution(df, grade_col='Grade', title='Grade Distribution'):
    """
    Bar chart showing how many students got each grade.
    """
    fig, ax = plt.subplots(figsize=(6,4))
    order = sorted(df[grade_col].unique())
    sns.countplot(x=grade_col, data=df, order=order, color='salmon', ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Grade")
    ax.set_ylabel("Count")
    st.pyplot(fig)

def plot_grade_vs_score(df, grade_col='Grade', score_col='Score',
                        all_grades=None, title='Average Score by Grade'):
    """
    Plots a line chart: X=Grade, Y=Average Score.
    """
    if all_grades is None:
        all_grades = sorted(df[grade_col].unique())

    # Safely compute average score for each grade
    means = df.groupby(grade_col)[score_col].mean().reindex(all_grades)
    means = means.dropna()  # remove missing if a grade wasn't assigned

    fig, ax = plt.subplots(figsize=(6,4))
    sns.lineplot(
        x=means.index,
        y=means.values,
        marker='o',
        color='purple',
        linewidth=2,
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Grade")
    ax.set_ylabel(f"Average {score_col}")
    ax.set_ylim(0, 100)
    st.pyplot(fig)

def plot_iqr_boxplot(df, col='Score', title='Box Plot - Outliers based on IQR'):
    """
    Plots a boxplot based on the IQR for the chosen column,
    to visualize outliers and how they might be handled.
    """
    fig, ax = plt.subplots(figsize=(6,4))

    # Calculate IQR
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Annotate the plot with IQR details
    sns.boxplot(
        y=df[col],
        ax=ax,
        color='skyblue'
    )
    ax.axhline(lower_bound, color='red', linestyle='--', label=f'Lower Bound = {lower_bound:.2f}')
    ax.axhline(upper_bound, color='red', linestyle='--', label=f'Upper Bound = {upper_bound:.2f}')
    ax.set_title(title)
    ax.legend()

    st.pyplot(fig)

def convert_df_to_csv(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame to CSV for download.
    """
    return df.to_csv(index=False)

# -----------------------------
# Main Streamlit App
# -----------------------------
def main():
    st.title("📊 Grading System: Absolute vs. Relative Grading")

    st.markdown("""
    This app lets you:
    1. **Upload** a CSV with **StudentID** and **Score**.
    2. **Choose** Absolute or Relative grading.
    3. **View** distribution plots and final grade counts.
    4. **See** a line chart of *Grade vs. Average Score*.
    5. **Check** how many got a specific Grade at each Score.
    6. **Download** a CSV with final grades.
    7. **See** an IQR-based boxplot for outliers.
    """)

    # 1. File upload
    uploaded_file = st.file_uploader("Upload your CSV (StudentID, Score)", type=["csv"])
    if not uploaded_file:
        st.warning("Please upload a valid CSV.")
        return
    try:
        df = read_csv_data(uploaded_file)
    except Exception as ex:
        st.error(f"Error reading CSV: {ex}")
        return

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # 2. Choose Grading Method
    grading_method = st.selectbox(
        "Choose a Grading Method",
        ["Absolute Grading", "Relative Grading"]
    )

    # 3. Branch: Absolute vs. Relative
    if grading_method == "Absolute Grading":
        st.subheader("Absolute Grading")
        thresholds = {'A': 90, 'B': 80, 'C': 70, 'D': 60}
        df["Grade"] = df["Score"].apply(assign_absolute_grade, thresholds=thresholds)

        st.write("Grades assigned based on these **absolute thresholds**:")
        st.json(thresholds)

        # Show data with assigned grades
        st.dataframe(df[["StudentID","Score","Grade"]].head())

        # Plot Score Distribution
        plot_distribution(df, col="Score", title="Score Distribution (Absolute)")

        # Boxplot with IQR
        st.subheader("IQR Boxplot (Absolute Grading)")
        plot_iqr_boxplot(df, col="Score", title="Outlier Detection via IQR (Absolute)")

        # Grade Distribution
        plot_grade_distribution(df, grade_col="Grade", title="Grade Distribution (Absolute)")

        # Grade vs. Score Plot
        plot_grade_vs_score(
            df,
            grade_col="Grade",
            score_col="Score",
            all_grades=["A","B","C","D","F"],
            title="Average Score by Grade (Absolute)"
        )

        # Show how many got a specific grade at each score
        st.subheader("Grade vs. Score Details (Absolute)")
        abs_counts = df.groupby(["Grade", "Score"]).size().reset_index(name="Count")
        st.dataframe(abs_counts)

        # Provide a CSV download for final results
        st.subheader("Download Final Grades (Absolute)")
        abs_csv = convert_df_to_csv(df[["StudentID","Score","Grade"]])
        st.download_button(
            label="Download CSV",
            data=abs_csv,
            file_name="absolute_grades.csv",
            mime="text/csv"
        )

    else:
        st.subheader("Relative Grading")
        df_transformed = transform_scores_normal_curve(df)
        df_grades = assign_letter_grades_from_percentiles(df_transformed, grade_col="FinalGrade")

        # Let's rename "FinalGrade" -> "Grade" to keep column naming consistent
        df_grades.rename(columns={"FinalGrade": "Grade"}, inplace=True)

        st.dataframe(df_grades[["StudentID","Score","AdjustedScore","Grade"]].head())

        # Plot raw Score distribution
        plot_distribution(df_grades, col="Score", title="Raw Score Distribution (Relative)")

        # Boxplot with IQR
        st.subheader("IQR Boxplot (Relative Grading)")
        plot_iqr_boxplot(df_grades, col="Score", title="Outlier Detection via IQR (Relative)")

        # Plot Adjusted (z-score) distribution
        st.write("**Adjusted Score (Z-Score) Distribution**")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(df_grades["AdjustedScore"], bins=15, stat="density",
                     color="skyblue", alpha=0.6, edgecolor="black", ax=ax, label="Histogram")
        sns.kdeplot(df_grades["AdjustedScore"], color="blue", lw=2, ax=ax, label="KDE")
        x_vals = np.linspace(-4, 4, 200)
        ax.plot(x_vals, norm.pdf(x_vals, 0, 1), 'r--', lw=2, label="Std Normal PDF")
        ax.set_title("Distribution of Adjusted Scores (Z-Scores)")
        ax.set_xlabel("Adjusted Score")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)

        # Grade Distribution
        plot_grade_distribution(df_grades, grade_col="Grade", 
                                title="Final Grade Distribution (Relative)")

        # Grade vs. Score Plot
        plot_grade_vs_score(
            df_grades,
            grade_col="Grade",
            score_col="Score",
            all_grades=["A","B","C","D","F"],
            title="Average Score by Grade (Relative)"
        )

        # Show how many got a specific grade at each score
        st.subheader("Grade vs. Score Details (Relative)")
        rel_counts = df_grades.groupby(["Grade", "Score"]).size().reset_index(name="Count")
        st.dataframe(rel_counts)

        # Provide a CSV download for final results
        st.subheader("Download Final Grades (Relative)")
        rel_csv = convert_df_to_csv(df_grades[["StudentID","Score","AdjustedScore","Grade"]])
        st.download_button(
            label="Download CSV",
            data=rel_csv,
            file_name="relative_grades.csv",
            mime="text/csv"
        )

    st.success("Grading and analysis completed successfully!")

if __name__ == "__main__":
    main()
