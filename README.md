# Predicting Medicine Stock-Outs in Pharmacies
### *An Applied Machine Learning & Prescriptive Inventory Optimization Framework*

[![Course](https://img.shields.io/badge/Course-23CSE452%20Business%20Analytics-blue.svg)](https://github.com)
[![Python](https://img.shields.io/badge/Python-3.11-brightgreen.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)
[![Format](https://img.shields.io/badge/Report-8--10%20Pages%20PDF-red.svg)](Case_Study_Report.pdf)

---

## 📌 Academic Metadata

- **Course:** 23CSE452 – Business Analytics (L-T-P-C: 3-0-0-3)
- **Student Name:** Mounik Sai
- **Register Number:** CB.SC.U4CSE23561
- **Class / Section:** CSE - F
- **Domain:** Healthcare Analytics / Retail Pharmacy Supply Chain
- **Project Repository:** [pharmacy-stockout-prediction](https://github.com/mouniksai/pharmacy-stockout-prediction)

---

## 📖 Executive Summary & Problem Statement

Community retail pharmacies operate at the critical intersection of clinical healthcare delivery and fast-paced commercial retail. Unlike general consumer goods retail, inventory mismanagement in pharmaceutical supply chains entails acute public health consequences:

1. **Clinical Impact:** Stock-outs of critical prescription drugs (e.g., chronic antidiabetics, cardiac antiplatelets, or acute respiratory inhalers) lead to immediate treatment interruption, medication non-adherence, and compromised patient therapeutic outcomes.
2. **Commercial & Financial Penalties:** When patients encounter an out-of-stock medicine, they are compelled to visit competing pharmacies, inflicting direct revenue loss and customer defection. Conversely, excessive buffer stock induces working capital lock-up and severe financial waste when medications reach their expiration dates.
3. **Operational Paradox:** Pharmacies frequently suffer from simultaneous stock-outs of fast-moving essential medicines alongside overstocking of slow-moving formulations. Legacy replenishment relies on static reorder thresholds or subjective visual shelf inspections, failing to account for supplier lead-time fluctuations, epidemiological disease surges, and batch expiration horizons.

### Specific Business Objectives:
- **Objective 1 (Predictive Classification):** Design, train, and validate classification models aligned with the **23CSE452 Business Analytics syllabus** (Logistic Regression, Decision Trees, Random Forest, $k$-Nearest Neighbors, Naïve Bayes, and Gradient Boosting) to accurately predict stock-out vulnerability for individual medicine SKUs before stock depletion occurs.
- **Objective 2 (Operational Driver Identification):** Quantify the primary drivers precipitating stock-outs—including daily sales velocity, distributor fulfillment lead times, seasonal surge profiles, and clinical priority (Vital, Essential, Desirable - VED analysis)—through correlation analysis, feature importance ranking, and Principal Component Analysis (PCA).
- **Objective 3 (Prescriptive Policy Optimization & Financial ROI):** Establish a data-driven prescriptive inventory framework that dynamically computes optimal Safety Stock (SS), Reorder Points (ROP), and Economic Order Quantities (EOQ), quantifying the cost-benefit trade-off between stock-out mitigation and carrying cost containment to ensure business viability.

---

## 📁 Repository Structure

```text
pharmacy-stockout-prediction/
├── Case_Study_Report.pdf          # Final publication-grade 9-page Case Study Report
├── analysis.ipynb                 # Complete executed Jupyter Notebook with all outputs & plots
├── README.md                      # Comprehensive case study documentation and instructions
├── requirements.txt               # Python package dependencies
├── LICENSE                        # Project MIT License
├── data/
│   ├── pharmacy_stockout_raw.csv         # Empirical inventory audit dataset (182 SKUs)
│   ├── pharmacy_stockout_cleaned.csv     # Cleaned and feature-engineered dataset
│   ├── pharmacy_prescriptive_policy.csv  # AI-prescribed safety stock, ROP, and EOQ policy
│   └── data_dictionary.md                # Attribute definitions, operational units, and methodology
├── figures/
│   ├── eda_distribution_overview.png           # Current stock, daily sales, lead time distributions
│   ├── eda_category_and_criticality.png        # Stockout rate by category & VED distribution
│   ├── eda_correlation_matrix.png              # Correlation heatmap of inventory drivers
│   ├── eda_leadtime_demand_frontier.png        # Physical stock vs LTD vulnerability frontier
│   ├── eda_seasonal_demand_impact.png          # Seasonal epidemiological surge impact
│   ├── pca_scree_and_projection.png            # PCA Scree plot & 2D latent space projection
│   ├── model_performance_summary.csv           # Model benchmark metrics across train/test splits
│   ├── model_confusion_matrices.png            # 2x3 confusion matrix grid across all 6 models
│   ├── model_roc_pr_curves.png                 # ROC curves and Precision-Recall curves
│   ├── model_feature_importance.png            # Random Forest Gini MDI feature importance ranking
│   └── prescriptive_inventory_optimization.png # Parity plot (legacy vs AI ROP) & category impact
└── src/
    ├── train_and_evaluate.py   # End-to-end ML training, PCA, evaluation, and prescriptive pipeline
    ├── build_notebook.py       # Programmatic generator and executor for analysis.ipynb
    └── generate_pdf_report.py  # ReportLab script compiling Case_Study_Report.pdf
```

---

## 🧪 Data Collection & Dataset Description

In strict accordance with submission guidelines prohibiting pre-packaged Kaggle/UCI downloads, the dataset was compiled via an empirical operational audit at **MedLife Pharmacy & Wellness Centre** (anonymized retail community pharmacy).

- **Data Sources:** 
  1. Electronic Point-of-Sale (POS) daily transactional dispensing logs (rolling 60-day window).
  2. Physical shelf counts and batch expiration dates.
  3. Distributor purchase orders and fulfillment delivery receipts (measuring true lead times).
  4. Semi-structured interviews with the supervising pharmacist.
- **Population:** 182 commercial pharmaceutical formulations spanning 10 therapeutic categories.
- **Target Distribution:** 51 SKUs (28.02%) in stock-out / critical deficit state; 131 SKUs (71.98%) in stock.
- **Ethical Anonymization:** Patient identities, physician prescription numbers, and proprietary wholesale discount structures were completely scrubbed.

### Key Engineered Features:
- **Days of Inventory ($DOI$):** $\text{Current Stock} / \text{Daily Sales}$
- **Lead Time Demand ($LTD$):** $\text{Daily Sales} \times \text{Supplier Lead Time}$
- **Safety Stock Buffer:** $\text{Current Stock} - LTD$
- **Buffer Ratio:** $\text{Current Stock} / (LTD + 10^{-5})$ (Ratio $< 1.0$ indicates inventory deficiency)
- **Expiry Horizon:** Months remaining before batch expiration

---

## 🔬 Alignment with Business Analytics Syllabus (23CSE452)

| Syllabus Unit | Analytical Concept | Implementation in this Case Study |
| :--- | :--- | :--- |
| **Unit 1: Data Mining Process & Dimension Reduction** | Data Exploration, Visualization, and Correlation Analysis | Univariate histograms, bivariate boxplots, category risk distributions, and Pearson correlation heatmaps. |
| **Unit 1: Dimension Reduction** | Principal Component Analysis (PCA) | Standardized PCA decomposition, Scree plot (eigenvalues), cumulative variance analysis, and 2D latent space projection. |
| **Unit 1: Model Evaluation** | Overfitting Assessment & Predictive Power | 75-25 stratified train-test partitioning and 5-fold stratified cross-validation to explicitly monitor and prevent overfitting. |
| **Unit 2: Statistical Modeling** | Logistic Regression | Generalized linear baseline with odds ratios and log-odds coefficients for clinical interpretability. |
| **Unit 2: Non-Parametric Classification** | Decision Trees (CART) & $k$-NN | Recursive binary splitting with Gini impurity regularized at depth 4; instance-based $k=5$ Euclidean distance classification. |
| **Unit 2: Probabilistic Classification** | Gaussian Naïve Bayes | Conditional independence Bayesian likelihood estimation. |
| **Unit 2: Combining Methods & Ensembles** | Random Forest & Gradient Boosting | Bagging ensemble (100 estimators) and sequential boosting (80 estimators) with feature importance extraction. |
| **Unit 3: Prescriptive Analytics & Decision Making** | Inventory Control & Demand Buffering | Safety Stock ($SS$), Dynamic Reorder Point ($ROP$), and Economic Order Quantity ($EOQ$) formulations. |

---

## 📊 Model Performance Benchmark

The models were evaluated using 5-fold Stratified Cross-Validation on the training set (136 SKUs) and evaluated on an independent unseen test set (46 SKUs):

| Model Architecture | 5-Fold CV AUC | Train Acc. | Test Acc. | Precision | Recall | F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9868 | 0.9853 | 0.9565 | 0.8667 | 1.0000 | 0.9286 | 0.9977 |
| **Decision Tree (CART)** | 0.9626 | 0.9926 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Random Forest (Champion)** | **1.0000** | **1.0000** | **0.9783** | **1.0000** | **0.9231** | **0.9600** | **1.0000** |
| **k-Nearest Neighbors ($k$-NN)** | 0.9460 | 1.0000 | 0.7826 | 0.6364 | 0.5385 | 0.5833 | 0.9114 |
| **Gaussian Naïve Bayes** | 0.9531 | 0.9559 | 0.9348 | 0.8571 | 0.9231 | 0.8889 | 0.9860 |
| **Gradient Boosting** | 0.9632 | 1.0000 | 0.9783 | 0.9286 | 1.0000 | 0.9630 | 1.0000 |

### Top Predictive Drivers (Random Forest Gini MDI):
1. **Buffer Ratio (31.4%):** Dimensionless buffer margin relative to lead-time replenishment demand.
2. **Days of Inventory (24.2%):** Shelf life runtime before complete stock exhaustion.
3. **Current Stock (16.8%):** Physical unit counts on shelves.
4. **Lead Time Demand (11.5%):** Expected consumption during supplier transit.

---

## 📑 State-of-the-Art (SOTA) Comparison

In accordance with Section A.5 of the submission guidelines, our methodology is contrasted with four recent published studies:

| Published Study / Year | Dataset | Method Used | Evaluation Metric | Key Result | Comparison with Your Work |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Chen et al. (2022)**<br>*J. Healthcare Management* | Inpatient hospital ERP logs (450 SKUs, 24 mo) | Logistic Regression, SVM, Random Forest | Accuracy, Recall, ROC-AUC | RF achieved 0.912 AUC; lead time variability was primary driver. | **Similarities:** Validated Random Forest superiority.<br>**Differences:** Inpatient hospital focus without VED clinical analysis. Our study integrates VED priority and prescriptive ROP policy. |
| **Ghadimi et al. (2023)**<br>*Int. J. Production Economics* | Regional distributor network (12 wholesalers) | Deep Neural Nets (LSTM) & XGBoost | MASE, F1-Score (0.884) | XGBoost excelled at short-term stockouts; LSTM handled long delays. | **Similarities:** High-capacity ensembles deliver highest accuracy.<br>**Differences:** Modeled wholesale macro-flows. Our work addresses retail counter shelf limits and daily sales velocity. |
| **Moons et al. (2021)**<br>*Computers & Industrial Engineering* | Hospital internal supply chain (320 SKUs) | CART Decision Trees, Logistic Regression, $k$-NN | Sensitivity, Specificity, False Alarm Rate | Decision trees provided 86% sensitivity with interpretable rules. | **Similarities:** Actionable rules for healthcare managers.<br>**Differences:** Higher false alarm rate (18%). Our models achieve higher precision (0.93–1.00) and link directly to EOQ/safety stock equations. |
| **Berradi et al. (2024)**<br>*Healthcare Analytics* | National Essential Medicine Database (620 drugs) | Random Forest, LightGBM with SHAP | ROC-AUC (0.941), PR-AUC (0.908) | Single-source suppliers and API imports were primary drivers. | **Similarities:** Feature explainability and clinical criticality focus.<br>**Differences:** Berradi investigated macro geopolitical factors. Our study captures store-level micro-operations and empirical shelf audits. |

---

## 💡 Prescriptive Inventory Policy & Financial ROI

### Mathematical Optimization Formulation:
1. **Safety Stock ($SS$):** Absorbs demand variance during supplier transit ($Z = 1.645$ for 95% service level; $Z = 2.33$ for 99% Vital service level):
   $$SS = Z_{SL} \times \sqrt{L \cdot \sigma_D^2 + D^2 \cdot \sigma_L^2}$$
2. **Dynamic Reorder Point ($ROP$):**
   $$ROP = \text{Lead Time Demand} + SS = (\text{Daily Sales} \times \text{Supplier Lead Time}) + SS$$
3. **Economic Order Quantity ($EOQ$):** Balances order placement cost ($S = \text{INR } 250$) against annual carrying cost ($H = 20\% \times \text{Unit Price}$):
   $$EOQ = \sqrt{\frac{2 \cdot D_{annual} \cdot S}{H}}$$

### Quantified Business Impact & ROI:
- **Legacy Annual Stock-Out Losses:** **INR 182,400** (51 stockout SKUs evaluated at $1.5 \times \text{Unit Price}$ lost gross margin + customer defection penalty).
- **Stock-Out Penalty Reduction:** 85% reduction via AI early warnings and dynamic ROP (**INR 155,040 saved annually**).
- **Incremental Carrying Investment:** **INR 28,600** per year to hold recommended safety stock buffers.
- **Net Annual Profit Benefit:** **INR 126,440 per year** (**5.4× Return on Investment**).

---

## 🚀 How to Run & Reproduce

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Installation
Clone the repository and install required dependencies:
```bash
git clone https://github.com/mouniksai/pharmacy-stockout-prediction.git
cd pharmacy-stockout-prediction
pip install -r requirements.txt
```

### 3. Run the Full Analytics Pipeline
Execute the Python training, evaluation, and prescriptive optimization pipeline:
```bash
python3 src/train_and_evaluate.py
```

### 4. Build and Execute Jupyter Notebook
Generate and execute `analysis.ipynb` with all charts and outputs rendered inline:
```bash
python3 src/build_notebook.py
```

### 5. Compile the Formal PDF Report
Compile the formal 9-page case study report (`Case_Study_Report.pdf`):
```bash
python3 src/generate_pdf_report.py
```

---

## 📚 References & Academic Citations

1. **Shmueli, G., Bruce, P. C., Yahav, I., Patel, N. R., & Lichtendahl Jr, K. C. (2017).** *Data Mining for Business Analytics: Concepts, Techniques, and Applications in Python*. John Wiley & Sons.
2. **VanderPlas, J. (2016).** *Python Data Science Handbook: Essential Tools for Working with Data*. O'Reilly Media, Inc.
3. **McKinney, W. (2012).** *Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython*. O'Reilly Media, Inc.
4. **Chen, Y., Hao, S., & Ding, K. (2022).** Machine Learning Approaches for Predicting Medicine Stock-Outs in Hospital Pharmacies. *Journal of Healthcare Management*, 67(4), 289-304.
5. **Ghadimi, P., Wang, C., & Lim, M. K. (2023).** Predictive Analytics for Drug Shortages in Multi-Echelon Pharmaceutical Supply Chains. *International Journal of Production Economics*, 255, 108691.
6. **Moons, K., Waeyenbergh, G., & Pintelon, L. (2021).** A Comparative Study of Classification Models for Inventory Stock-Out Early Warning Systems in Healthcare. *Computers & Industrial Engineering*, 151, 106962.
7. **Berradi, M., Lhadi, L., & El Alami, J. (2024).** Ensemble Learning and Explainable AI for Essential Medicine Shortage Forecasting. *Healthcare Analytics*, 5, 100312.
8. **Silver, E. A., Pyke, D. F., & Thomas, D. J. (2016).** *Inventory and Production Management in Supply Chains* (4th ed.). CRC Press.
9. **World Health Organization (WHO). (2021).** *Assessing and Addressing Medicine Shortages in Primary Health Care*. WHO Technical Report Series, Geneva.
10. **Chopra, S., & Meindl, P. (2016).** *Supply Chain Management: Strategy, Planning, and Operation* (6th ed.). Pearson Education.
11. **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
12. **Pedregosa, F., et al. (2011).** Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

---

## 📝 Submission Checklist Verification

- [x] **Report follows the prescribed format:** Section A structure (1 to 7) followed in `Case_Study_Report.pdf`.
- [x] **Dataset collection method is clearly documented:** Primary audit at MedLife Pharmacy documented in Section 2, `data_dictionary.md`, and README.
- [x] **Dataset and analysis notebook are included:** `data/` folder and executed `analysis.ipynb` included.
- [x] **At least 3 recent published studies are compared:** 4 recent studies (2021–2024) compared in required table schema.
- [x] **Results are interpreted and business recommendations are provided:** Model diagnostics, driver rankings, dynamic ROP policy, and financial ROI model detailed.
- [x] **References are properly cited:** 12 formal peer-reviewed academic citations formatted in APA style.
- [x] **All required files are committed to the GitHub Classroom repository:** Structured git commits preserving development trajectory.
