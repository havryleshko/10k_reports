### Predicting Corporate Cash Burn (2021–2024)

**This project uses financial statement data from public energy companies to predict whether a company will burn cash in the next fiscal year — a critical signal for startups and many other firms.**

**Requirements:** Python, Pandas, PyCaret, H2O.ai, Matplotlib

## Problem Statement

Can we predict if a public company will burn cash next year, using its financial statements from this year?

**Target:** burn_cash (1 if next-year Free Cash Flow < 0, else 0)

**Use case:** Corporate finance teams, startup analysts, or risk management for early-stage or capital-intensive sectors

## Data

**Source:** Manually extracted from Perplexity Finance (perplexity.ai/finance)

**Companies:** 32 (Renewable & Nuclear Energy + Industrial sectors)

**Years:** 2021–2024

**Rows:** 123 after cleaning

## Workflow

**1. Data Engineering**

1. Merged Balance Sheet + Cash Flow statements

2. Added future targets: shifted next-year Free Cash Flow (FCF), Operating Cash Flow (OCF)

3. Created binary burn_cash label

**2. Feature Engineering**

1. ~10+ derived features: investment ratios, working capital changes, debt repayment ratio, and YoY deltas

**3. Modeling**

1. AutoML with PyCaret: binary classification (used in final version due to small data size)

2. H2O.ai AutoML: experimental run

3. Metrics: Accuracy, AUC, F1, MCC, Recall

## Results

**Top PyCaret models:**

Model

Accuracy

AUC

F1 Score

MCC

Decision Tree

0.93

0.95

0.93

0.90

AdaBoost

0.93

0.95

0.93

0.90

Gradient Boosting

0.93

0.95

0.93

0.90

***Note: Results show signs of overfitting due to the small dataset size.***

## Lessons Learned

1. AutoML tools require strong feature engineering and domain knowledge

2. Even with clean financial data, data size limits generalization

3. Feature engineering, shifted targets, explored 2 AutoML tools

## How to Run

**clone the repository**
https://github.com/havryleshko/10k_reports

**install requirements**
pip install -r requirements.txt

**run the main script**
python main.py

## Case Study

Pushed into repository as a separate .pdf file

## Author

**Alex Havryleshko:**

**Youtube:** https://www.youtube.com/@havryleshko

**X:** https://x.com/alexhavryleshko

**GitHub:** https://github.com/havryleshko

