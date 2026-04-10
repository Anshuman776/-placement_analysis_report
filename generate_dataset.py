"""
Generate a realistic synthetic dataset of 2000 students for
Ajay Kumar Garg Engineering College (AKGEC) Placement Analysis.
"""

import pandas as pd
import numpy as np
import os
import random

np.random.seed(42)
random.seed(42)

NUM_STUDENTS = 2000

# ── Indian first names & last names ─────────────────────────────────────
first_names_male = [
    "Aarav", "Arjun", "Vivaan", "Aditya", "Vihaan", "Sai", "Reyansh",
    "Ayaan", "Krishna", "Ishaan", "Shaurya", "Atharv", "Advik", "Pranav",
    "Advaith", "Aryan", "Dhruv", "Kabir", "Ritvik", "Harsh", "Rohan",
    "Kunal", "Amit", "Rahul", "Mohit", "Nikhil", "Ankur", "Gaurav",
    "Varun", "Deepak", "Vishal", "Prateek", "Ravi", "Suraj", "Akash",
    "Manish", "Ankit", "Saurabh", "Abhishek", "Rajat", "Tushar", "Vikas",
    "Yash", "Kartik", "Dev", "Neeraj", "Sachin", "Mayank", "Himanshu", "Aman"
]

first_names_female = [
    "Aanya", "Diya", "Saanvi", "Ananya", "Aadhya", "Isha", "Pari",
    "Anika", "Myra", "Sara", "Navya", "Riya", "Priya", "Sneha",
    "Pooja", "Neha", "Shruti", "Kavya", "Tanvi", "Simran", "Nidhi",
    "Sakshi", "Meghna", "Divya", "Anjali", "Swati", "Pallavi", "Komal",
    "Jyoti", "Sonal", "Aparna", "Ritika", "Bhavna", "Mansi", "Tanya",
    "Aishwarya", "Kriti", "Nikita", "Radhika", "Shivani", "Aditi",
    "Megha", "Trisha", "Vidhi", "Khushi", "Nupur", "Charu", "Garima",
    "Preeti", "Sonam"
]

last_names = [
    "Sharma", "Verma", "Gupta", "Singh", "Kumar", "Mishra", "Patel",
    "Agarwal", "Jain", "Yadav", "Chauhan", "Pandey", "Tiwari", "Dubey",
    "Srivastava", "Saxena", "Mehta", "Rastogi", "Bansal", "Goel",
    "Malhotra", "Kapoor", "Bhatia", "Arora", "Nair", "Reddy", "Das",
    "Mukherjee", "Chatterjee", "Iyer", "Joshi", "Rawat", "Thakur",
    "Rajput", "Bhatt", "Khanna", "Chopra", "Sethi", "Dhawan", "Bajaj"
]

BRANCHES = ["CSE", "IT", "ECE", "ME", "CE", "EE"]
BRANCH_WEIGHTS = [0.30, 0.20, 0.18, 0.14, 0.10, 0.08]

# Companies grouped by tier
TIER1_COMPANIES = ["Google", "Microsoft", "Amazon", "Adobe", "Goldman Sachs",
                   "Flipkart", "Uber", "Samsung R&D", "Oracle", "Qualcomm"]
TIER2_COMPANIES = ["TCS", "Infosys", "Wipro", "HCL", "Cognizant", "Accenture",
                   "Capgemini", "Tech Mahindra", "Mindtree", "L&T Infotech",
                   "Mphasis", "Hexaware", "Zoho", "Paytm", "Byju's"]
TIER3_COMPANIES = ["Mu Sigma", "NIIT", "Concentrix", "Genpact", "iGate",
                   "Sapient", "Amdocs", "Syntel", "Cyient", "Sasken"]


def generate_name():
    """Generate a random Indian name."""
    gender = random.choice(["M", "F"])
    if gender == "M":
        first = random.choice(first_names_male)
    else:
        first = random.choice(first_names_female)
    last = random.choice(last_names)
    return f"{first} {last}"


def generate_roll_number(branch, index):
    """Generate AKGEC-style roll number."""
    branch_codes = {"CSE": "CS", "IT": "IT", "ECE": "EC", "ME": "ME", "CE": "CE", "EE": "EE"}
    year = random.choice([21, 22, 23, 24])
    return f"{year}00{branch_codes[branch]}{str(index).zfill(3)}"


def generate_student(index):
    """Generate a single student record with realistic correlations."""
    branch = np.random.choice(BRANCHES, p=BRANCH_WEIGHTS)
    name = generate_name()
    roll_no = generate_roll_number(branch, index)

    # CGPA: normally distributed, branch-biased
    branch_cgpa_boost = {"CSE": 0.3, "IT": 0.2, "ECE": 0.1, "ME": 0.0, "CE": -0.1, "EE": -0.05}
    cgpa = np.clip(np.random.normal(7.2 + branch_cgpa_boost[branch], 1.2), 4.0, 10.0)
    cgpa = round(cgpa, 2)

    # Skills are correlated with CGPA (higher CGPA → generally better skills)
    cgpa_factor = (cgpa - 4.0) / 6.0  # normalized 0-1

    # DSA Score (0-100) — strongly correlated with CGPA for CSE/IT
    dsa_base = cgpa_factor * 60 + np.random.normal(20, 15)
    if branch in ["CSE", "IT"]:
        dsa_base += 10
    dsa_score = int(np.clip(dsa_base, 0, 100))

    # Web Dev Score (0-100)
    webdev_base = cgpa_factor * 40 + np.random.normal(25, 18)
    if branch in ["CSE", "IT"]:
        webdev_base += 8
    webdev_score = int(np.clip(webdev_base, 0, 100))

    # ML Score (0-100)
    ml_base = cgpa_factor * 45 + np.random.normal(15, 16)
    if branch in ["CSE", "ECE"]:
        ml_base += 5
    ml_score = int(np.clip(ml_base, 0, 100))

    # Aptitude Score (0-100) — moderately correlated with CGPA
    aptitude_base = cgpa_factor * 50 + np.random.normal(25, 14)
    aptitude_score = int(np.clip(aptitude_base, 0, 100))

    # Communication Score (0-100) — weakly correlated with CGPA
    comm_base = cgpa_factor * 20 + np.random.normal(40, 18)
    comm_score = int(np.clip(comm_base, 0, 100))

    # Internships (0-4) — correlated with skills
    avg_skill = (dsa_score + webdev_score + ml_score) / 300
    internship_prob = np.clip(avg_skill + cgpa_factor * 0.3, 0, 1)
    internships = np.random.choice([0, 1, 2, 3, 4],
                                   p=_internship_probs(internship_prob))

    # Projects (0-8) — correlated with skills
    projects = int(np.clip(np.random.poisson(2 + avg_skill * 4 + internships * 0.5), 0, 8))

    # Hackathons (0-5)
    hackathons = int(np.clip(np.random.poisson(0.5 + avg_skill * 2), 0, 5))

    # ── Placement Decision ──────────────────────────────────────────────
    placement_score = (
        cgpa_factor * 0.25 +
        (dsa_score / 100) * 0.20 +
        (webdev_score / 100) * 0.10 +
        (ml_score / 100) * 0.10 +
        (aptitude_score / 100) * 0.15 +
        (comm_score / 100) * 0.08 +
        (internships / 4) * 0.07 +
        (projects / 8) * 0.05
    )
    # Add noise
    placement_score += np.random.normal(0, 0.08)
    placement_threshold = 0.45
    placed = placement_score >= placement_threshold

    # Salary & Company
    salary = 0
    company = "Not Placed"
    if placed:
        if placement_score > 0.75:
            # Tier 1
            base_salary = np.random.uniform(12, 45)
            company = random.choice(TIER1_COMPANIES)
        elif placement_score > 0.55:
            # Tier 2
            base_salary = np.random.uniform(4.5, 12)
            company = random.choice(TIER2_COMPANIES)
        else:
            # Tier 3
            base_salary = np.random.uniform(3.0, 5.0)
            company = random.choice(TIER3_COMPANIES)
        salary = round(base_salary, 2)

    return {
        "Name": name,
        "Roll_No": roll_no,
        "Branch": branch,
        "CGPA": cgpa,
        "DSA_Score": dsa_score,
        "WebDev_Score": webdev_score,
        "ML_Score": ml_score,
        "Aptitude_Score": aptitude_score,
        "Communication_Score": comm_score,
        "Internships": internships,
        "Projects": projects,
        "Hackathons": hackathons,
        "Placement_Status": "Placed" if placed else "Not Placed",
        "Salary_LPA": salary,
        "Company": company,
    }


def _internship_probs(factor):
    """Return probability distribution for internship count based on factor."""
    if factor > 0.7:
        return [0.05, 0.15, 0.35, 0.30, 0.15]
    elif factor > 0.4:
        return [0.15, 0.35, 0.30, 0.15, 0.05]
    else:
        return [0.45, 0.30, 0.15, 0.08, 0.02]


def main():
    print("🎓 Generating AKGEC Placement Dataset...")
    students = [generate_student(i) for i in range(NUM_STUDENTS)]
    df = pd.DataFrame(students)

    # Create output directory
    os.makedirs("dataset", exist_ok=True)
    output_path = os.path.join("dataset", "students.csv")
    df.to_csv(output_path, index=False)

    print(f"✅ Dataset generated: {output_path}")
    print(f"   Total students: {len(df)}")
    print(f"   Placed: {len(df[df['Placement_Status'] == 'Placed'])} "
          f"({len(df[df['Placement_Status'] == 'Placed']) / len(df) * 100:.1f}%)")
    print(f"   Not Placed: {len(df[df['Placement_Status'] == 'Not Placed'])}")
    print(f"\n📊 Branch Distribution:")
    print(df['Branch'].value_counts().to_string())
    print(f"\n💰 Salary Stats (Placed students):")
    placed = df[df['Placement_Status'] == 'Placed']
    print(f"   Mean: {placed['Salary_LPA'].mean():.2f} LPA")
    print(f"   Median: {placed['Salary_LPA'].median():.2f} LPA")
    print(f"   Max: {placed['Salary_LPA'].max():.2f} LPA")


if __name__ == "__main__":
    main()
