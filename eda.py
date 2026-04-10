"""
Exploratory Data Analysis (EDA) for AKGEC Placement Dataset.
Generates visualization charts saved to ml/charts/ directory.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 12

CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv("dataset/students.csv")
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumn types:\n{df.dtypes}")
    print(f"\nBasic Statistics:\n{df.describe()}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    return df


def placement_rate_by_branch(df):
    """Bar chart: Placement rate by branch."""
    branch_stats = df.groupby('Branch')['Placement_Status'].apply(
        lambda x: (x == 'Placed').sum() / len(x) * 100
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(branch_stats))
    bars = ax.bar(branch_stats.index, branch_stats.values, color=colors,
                  edgecolor='white', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, branch_stats.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=13)

    ax.set_title('Placement Rate by Branch (AKGEC)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Branch', fontsize=13)
    ax.set_ylabel('Placement Rate (%)', fontsize=13)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'placement_rate_by_branch.png'), dpi=150)
    plt.close()
    print("✅ Chart saved: placement_rate_by_branch.png")


def skill_vs_placement(df):
    """Correlation heatmap: Skills vs Placement."""
    skill_cols = ['CGPA', 'DSA_Score', 'WebDev_Score', 'ML_Score',
                  'Aptitude_Score', 'Communication_Score', 'Internships', 'Projects', 'Hackathons']

    df_numeric = df.copy()
    df_numeric['Placed'] = (df_numeric['Placement_Status'] == 'Placed').astype(int)
    cols_for_corr = skill_cols + ['Placed', 'Salary_LPA']

    corr = df_numeric[cols_for_corr].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, square=True, linewidths=1, ax=ax,
                cbar_kws={'label': 'Correlation Coefficient'})
    ax.set_title('Skill vs Placement Correlation Heatmap', fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'skill_placement_correlation.png'), dpi=150)
    plt.close()
    print("✅ Chart saved: skill_placement_correlation.png")


def cgpa_vs_salary(df):
    """Scatter plot: CGPA vs Salary trend."""
    placed = df[df['Placement_Status'] == 'Placed'].copy()

    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(placed['CGPA'], placed['Salary_LPA'],
                         c=placed['Salary_LPA'], cmap='plasma',
                         alpha=0.6, s=40, edgecolors='white', linewidth=0.5)

    # Add trend line
    z = np.polyfit(placed['CGPA'], placed['Salary_LPA'], 2)
    p = np.poly1d(z)
    x_line = np.linspace(placed['CGPA'].min(), placed['CGPA'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend Line')

    plt.colorbar(scatter, label='Salary (LPA)')
    ax.set_title('CGPA vs Salary Trend (Placed Students)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('CGPA', fontsize=13)
    ax.set_ylabel('Salary (LPA)', fontsize=13)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'cgpa_vs_salary.png'), dpi=150)
    plt.close()
    print("✅ Chart saved: cgpa_vs_salary.png")


def most_demanded_skills(df):
    """Bar chart: Average skill scores for placed vs not placed."""
    skill_cols = ['DSA_Score', 'WebDev_Score', 'ML_Score', 'Aptitude_Score', 'Communication_Score']
    skill_labels = ['DSA', 'Web Dev', 'ML', 'Aptitude', 'Communication']

    placed_avg = df[df['Placement_Status'] == 'Placed'][skill_cols].mean()
    not_placed_avg = df[df['Placement_Status'] == 'Not Placed'][skill_cols].mean()

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(skill_labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, placed_avg.values, width, label='Placed',
                   color='#2ecc71', edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width / 2, not_placed_avg.values, width, label='Not Placed',
                   color='#e74c3c', edgecolor='white', linewidth=1)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{bar.get_height():.1f}', ha='center', fontsize=10, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{bar.get_height():.1f}', ha='center', fontsize=10, fontweight='bold')

    ax.set_title('Most Demanded Skills: Placed vs Not Placed', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Skill', fontsize=13)
    ax.set_ylabel('Average Score', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(skill_labels)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'most_demanded_skills.png'), dpi=150)
    plt.close()
    print("✅ Chart saved: most_demanded_skills.png")


def salary_distribution(df):
    """Histogram: Salary distribution of placed students."""
    placed = df[df['Placement_Status'] == 'Placed']

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(placed['Salary_LPA'], bins=30, color='#3498db', edgecolor='white',
            linewidth=1, alpha=0.8)

    ax.axvline(placed['Salary_LPA'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {placed["Salary_LPA"].mean():.2f} LPA')
    ax.axvline(placed['Salary_LPA'].median(), color='green', linestyle='--',
               linewidth=2, label=f'Median: {placed["Salary_LPA"].median():.2f} LPA')

    ax.set_title('Salary Distribution (Placed Students)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Salary (LPA)', fontsize=13)
    ax.set_ylabel('Number of Students', fontsize=13)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'salary_distribution.png'), dpi=150)
    plt.close()
    print("✅ Chart saved: salary_distribution.png")


def branch_wise_salary(df):
    """Box plot: Branch-wise salary distribution."""
    placed = df[df['Placement_Status'] == 'Placed']

    fig, ax = plt.subplots(figsize=(12, 7))
    branch_order = placed.groupby('Branch')['Salary_LPA'].median().sort_values(ascending=False).index
    sns.boxplot(data=placed, x='Branch', y='Salary_LPA', order=branch_order,
                palette='viridis', ax=ax)

    ax.set_title('Branch-wise Salary Distribution', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Branch', fontsize=13)
    ax.set_ylabel('Salary (LPA)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'branch_wise_salary.png'), dpi=150)
    plt.close()
    print("✅ Chart saved: branch_wise_salary.png")


def top_companies(df):
    """Horizontal bar chart: Top recruiting companies."""
    placed = df[df['Placement_Status'] == 'Placed']
    company_counts = placed['Company'].value_counts().head(15)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = sns.color_palette("viridis", len(company_counts))
    bars = ax.barh(company_counts.index[::-1], company_counts.values[::-1],
                   color=colors[::-1], edgecolor='white', linewidth=1)

    for bar, val in zip(bars, company_counts.values[::-1]):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f'{val}', va='center', fontweight='bold', fontsize=11)

    ax.set_title('Top 15 Recruiting Companies', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Number of Students Placed', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'top_companies.png'), dpi=150)
    plt.close()
    print("✅ Chart saved: top_companies.png")


def main():
    print("=" * 60)
    print("📊 AKGEC Placement Dataset — Exploratory Data Analysis")
    print("=" * 60)

    df = load_data()

    print("\n🔍 Generating charts...\n")
    placement_rate_by_branch(df)
    skill_vs_placement(df)
    cgpa_vs_salary(df)
    most_demanded_skills(df)
    salary_distribution(df)
    branch_wise_salary(df)
    top_companies(df)

    print(f"\n✅ All charts saved to '{CHARTS_DIR}/' directory!")


if __name__ == "__main__":
    main()
