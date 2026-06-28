# Mortgage-Backed Securities Risk Modeling

This project analyzes key risks embedded in Mortgage-Backed Securities (MBS) using loan-level mortgage data and macroeconomic variables. The objective is to model and quantify credit risk, prepayment risk, and interest rate risk, which are the primary drivers of valuation and performance in mortgage-backed instruments.

The analysis combines statistical modeling, survival analysis, and interest rate simulations to provide a structured risk framework.

---

## Overview

Mortgage-Backed Securities are pools of mortgage loans whose cash flows depend on borrower behavior and interest rate movements. The performance of these securities is primarily affected by:

- **Credit Risk** – probability of borrower default
- **Prepayment Risk** – early repayment of mortgages by borrowers
- **Interest Rate Risk** – changes in rates affecting discounting and borrower behavior

This project builds models to study each of these risks using publicly available datasets and Python-based analytical tools.

---

## Repository Structure

| File | Risk Type | Description |
|---|---|---|
| `logistic_regression_model.py` / `.ipynb` | Credit | Cleans loan-level data, engineers features, fits a logistic regression to predict default probability, evaluates via ROC-AUC / precision-recall |
| `cph.py` / `.ipynb` | Credit | Builds loan-level survival dataset (duration + event), fits a Cox Proportional Hazards model, checks multicollinearity (VIF) and the proportional hazards assumption |
| `PSA_Final.py` / `.ipynb` | Prepayment | Computes scheduled principal, prepayment amounts, SMM and CPR from loan-level cashflows; fits the result to the 100% PSA benchmark curve |
| `vasicek_model.py` / `.ipynb` | Interest Rate | Calibrates the Vasicek short-rate model via Maximum Likelihood Estimation on historical 10-year Treasury yields |
| `monte_carlo.ipynb` | Interest Rate | Simulates multiple interest rate paths under the calibrated Vasicek model |
| `data_merge.ipynb` | — | Combines loan-level (Freddie Mac) data with macroeconomic (FRED) data into the merged datasets the other scripts consume |
| `execution.ipynb` | — | End-to-end notebook tying the pipeline stages together |
| `DGS10.csv` | Data | 10-year Treasury daily yield series from FRED, used by the Vasicek model |

> **Note:** the `.py` files are notebook exports (they still contain `# In[ ]:` cell markers) — open the matching `.ipynb` if you want the original cell-by-cell output and plots.

---

## Setup

```bash
git clone https://github.com/ShubhamBalodi619/mbs-risk-modeling.git
cd mbs-risk-modeling
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Data

The loan-level inputs (`final_master_loan_data.csv`, `final_cph_data.csv`, `regression_data.csv`) are derived from the **Freddie Mac Single-Family Loan-Level Dataset** and are not included in this repo due to size and Freddie Mac's data licensing terms. To reproduce:

1. Register and download the loan-level dataset from [Freddie Mac's Single Family Loan-Level Dataset portal](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset).
2. Run `data_merge.ipynb` to merge it with `DGS10.csv` (already included) and produce the cleaned files each script expects.
3. Update the file paths at the top of each script (see below) to point at your local copies.

The Treasury yield data (`DGS10.csv`) is included and comes from [FRED](https://fred.stlouisfed.org/series/DGS10).

---

## Running the models

Each script/notebook is currently standalone and expects a hardcoded input path near the top, e.g.:

```python
df = pd.read_csv(r"C:\Users\sbalodi\Desktop\final_master_loan_data.csv")
```

Update this to your own path before running, for example:

```python
df = pd.read_csv("data/final_master_loan_data.csv")
```

Suggested run order if reproducing from raw data:

1. `data_merge.ipynb` — build merged datasets
2. `logistic_regression_model.py` and/or `cph.py` — credit risk
3. `PSA_Final.py` — prepayment risk
4. `vasicek_model.py` then `monte_carlo.ipynb` — interest rate risk
5. `execution.ipynb` — end-to-end view

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

### 2. Prepayment Risk Modeling

Prepayment behavior is analyzed using the **PSA prepayment model**, a standard benchmark used in mortgage analytics.

The model estimates expected prepayment rates over time and helps assess how borrower refinancing behavior affects MBS cash flows. Scheduled principal and prepayment amounts are derived directly from loan-level amortization mechanics (level-payment formula, recast on rate change, payoff via Zero Balance Code) rather than assumed, then aggregated into monthly SMM/CPR and fit to the 100% PSA benchmark curve.

### 3. Interest Rate Risk Modeling

Interest rate dynamics are simulated using the **Vasicek interest rate model**:

$$dr(t) = \kappa(\theta - r(t))dt + \sigma \, dW(t)$$

Parameters (κ, θ, σ) are calibrated via Maximum Likelihood Estimation on historical 10-year Treasury yields. Monte Carlo simulations are then performed to generate multiple interest rate paths and analyze the impact of rate movements on mortgage cash flows and valuation.

---

## Tools and Technologies

- Python
- Pandas / NumPy – data manipulation
- Scikit-learn – machine learning models
- Lifelines – survival analysis (Cox model)
- SciPy – Vasicek MLE calibration
- Statsmodels – multicollinearity diagnostics (VIF)
- Matplotlib / Seaborn – visualization

---

## Data Sources

- **[Freddie Mac Single-Family Loan-Level Dataset](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset)** — historical performance data on mortgage loans including borrower characteristics, loan terms, and delinquency status.
- **[FRED (Federal Reserve Economic Data)](https://fred.stlouisfed.org/)** — macroeconomic indicators such as interest rates used as explanatory variables.

---

## Known Limitations

- Input data paths are currently hardcoded per-script rather than passed as arguments or read from a config file.
- No automated tests; results were validated manually (VIF checks, PH assumption checks, ROC-AUC).
- The Vasicek model is single-factor and can produce negative rates; CIR or Hull-White would be natural extensions.
- `requirements.txt` versions are inferred from usage, not pinned from an original environment — pin exact versions with `pip freeze` if exact reproducibility matters.
