# Mortgage-Backed Securities Risk Modeling

This project analyzes key risks embedded in Mortgage-Backed Securities (MBS) using loan-level mortgage data and macroeconomic variables. The objective is to model and quantify credit risk, prepayment risk, and interest rate risk, which are the primary drivers of valuation and performance in mortgage-backed instruments.

The analysis combines statistical modeling, survival analysis, and interest rate simulations to provide a structured risk framework.

---

## Overview

Mortgage-Backed Securities are pools of mortgage loans whose cash flows depend on borrower behavior and interest rate movements. The performance of these securities is primarily affected by:

- Credit Risk – probability of borrower default  
- Prepayment Risk – early repayment of mortgages by borrowers  
- Interest Rate Risk – changes in rates affecting discounting and borrower behavior  

This project builds models to study each of these risks using publicly available datasets and Python-based analytical tools.

---

## Data Sources

- **Freddie Mac Single-Family Loan-Level Dataset**  
  Historical performance data on mortgage loans including borrower characteristics, loan terms, and delinquency status.

- **FRED (Federal Reserve Economic Data)**  
  Macroeconomic indicators such as interest rates used as explanatory variables.

---

## Methodology

### 1. Credit Risk Modeling

Borrower default risk is modeled using two approaches:

**Logistic Regression**
- Predicts probability of default
- Evaluated using ROC-AUC

**Cox Proportional Hazards Model**
- Survival analysis approach
- Models time until default event

Key features include borrower attributes, loan characteristics, and macroeconomic variables.

**Performance**
- Logistic Regression ROC-AUC: **0.83**
- Cox Model Concordance Index: **0.7623**

---

### 2. Prepayment Risk Modeling

Prepayment behavior is analyzed using the **PSA prepayment model**, a standard benchmark used in mortgage analytics.

The model estimates expected prepayment rates over time and helps assess how borrower refinancing behavior affects MBS cash flows.

---

### 3. Interest Rate Risk Modeling

Interest rate dynamics are simulated using the **Vasicek interest rate model**.

Monte Carlo simulations are performed to generate multiple interest rate paths and analyze the impact of rate movements on mortgage cash flows and valuation.

---

## Tools and Technologies

- Python  
- Pandas / NumPy – data manipulation  
- Scikit-learn – machine learning models  
- Lifelines – survival analysis (Cox model)  
- Matplotlib / Seaborn – visualization  

---
