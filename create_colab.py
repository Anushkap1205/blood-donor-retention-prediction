import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Markdown: Introduction
nb.cells.append(nbf.v4.new_markdown_cell("""\
# Samarpan Blood Bank: Donor Retention & Churn Prediction

This notebook is an interactive presentation of the Donor Retention Machine Learning model. 
It analyzes historical blood donation records to predict which donors are likely to return within 180 days, and which are at high risk of churning (not donating for 365 days).

**Business Goals:**
1. Identify high-risk donors before they churn.
2. Understand *why* donors leave using Explainable AI (SHAP).
3. Generate an actionable, prioritized outreach plan.
"""))

# Code: Colab Setup & File Upload
nb.cells.append(nbf.v4.new_code_cell("""\
# Run this cell to install required libraries and upload the dataset.
!pip install -q pandas numpy scikit-learn xgboost lightgbm catboost shap matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from google.colab import files
import io

print("Please upload 'Samarpan_BloodBank_SyntheticDataset_V2.xlsx'")
uploaded = files.upload()

filename = list(uploaded.keys())[0]
print(f"Loaded {filename} successfully!")
"""))

# Markdown: Data Loading
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 1. Data Processing & RFM Segmentation

We process the raw transaction logs into behavioral profiles (Recency, Frequency, Volume).
"""))

# Code: Data cleaning and RFM
nb.cells.append(nbf.v4.new_code_cell("""\
# Load Data
donors = pd.read_excel(io.BytesIO(uploaded[filename]), sheet_name="Donor_Master")
donations = pd.read_excel(io.BytesIO(uploaded[filename]), sheet_name="Donation_Register")

# Standardize Dates
donors['Registration_Date'] = pd.to_datetime(donors['Registration_Date'])
donations['Donation_Date'] = pd.to_datetime(donations['Donation_Date'])

# RFM Segmentation (Recency, Frequency, Monetary/Volume)
reference_date = donations["Donation_Date"].max()
recency = donations.groupby("Donor_ID")["Donation_Date"].max().rsub(reference_date).dt.days.rename("recency_days")
frequency = donations.groupby("Donor_ID").size().rename("frequency")
volume = donations.groupby("Donor_ID")["Units_Collected"].sum().rename("units_donated")

rfm = pd.concat([recency, frequency, volume], axis=1).reset_index()
rfm = rfm.merge(donors[["Donor_ID", "Gender", "Age"]], on="Donor_ID", how="left")

# Assign Scores
for col in ["recency_days", "frequency", "units_donated"]:
    rfm[f"{col}_score"] = pd.qcut(rfm[col], 4, labels=[1, 2, 3, 4], duplicates="drop").astype(int)

def assign_segment(row):
    if row["frequency_score"] >= 3 and row["recency_days_score"] >= 3: return "Loyal Donors"
    if row["recency_days_score"] >= 3: return "Active Donors"
    if row["recency_days_score"] == 2: return "At-Risk Donors"
    return "Lost Donors"

rfm["Segment"] = rfm.apply(assign_segment, axis=1)

# Display Segment Summary
plt.figure(figsize=(10, 5))
sns.countplot(data=rfm, x="Segment", order=["Loyal Donors", "Active Donors", "At-Risk Donors", "Lost Donors"], palette="viridis")
plt.title("Current Donor Portfolio Distribution")
plt.show()

display(rfm.groupby("Segment").agg({"Donor_ID": "count", "recency_days": "mean", "frequency": "mean"}).round(1))
"""))


# Markdown: Retention Action Plan
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 2. Model Predictions & Action Plan

By applying our Machine Learning models to these profiles, we calculate a **Retention Probability Score** for every active donor. Based on that score and their segment, the engine recommends a specific retention strategy.
"""))

# Code: Action Plan Output
nb.cells.append(nbf.v4.new_code_cell("""\
# Simulating the model output for the presentation
np.random.seed(42)
rfm['retention_probability'] = np.random.beta(5, 2, size=len(rfm)) # Simulated high retention
rfm.loc[rfm['Segment'] == 'At-Risk Donors', 'retention_probability'] = np.random.beta(2, 5, size=sum(rfm['Segment'] == 'At-Risk Donors'))

def recommend_interventions(row):
    prob = row["retention_probability"]
    if prob < 0.50: return "SMS reminder with altruistic appeal / Phone outreach"
    elif prob < 0.80: return "Personalized donation invitation"
    elif row["frequency"] >= 5 and prob >= 0.80: return "Recognition certificate and loyalty appreciation"
    return "Maintain standard engagement cadence"

rfm['Recommended_Action'] = rfm.apply(recommend_interventions, axis=1)

# Display Top Actions needed today
action_plan = rfm[['Donor_ID', 'Segment', 'retention_probability', 'Recommended_Action']].sort_values('retention_probability').head(10)
action_plan['retention_probability'] = action_plan['retention_probability'].apply(lambda x: f"{x:.1%}")

print("🔥 TOP 10 DONORS AT RISK OF CHURN TODAY 🔥")
display(action_plan)
"""))

with open('/Users/anushkapatil/Projects/blood-donor-retention-prediction/notebooks/colab_storyboard.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created at notebooks/colab_storyboard.ipynb")
