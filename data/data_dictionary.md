# Data Dictionary: Pharmacy Stock-Out Prediction Dataset

## 1. Overview & Data Collection Methodology

This dataset was compiled through a structured, multi-week operational inventory audit conducted at **MedLife Pharmacy & Wellness Centre** (anonymized retail community pharmacy). The audit recorded physical shelf counts, point-of-sale (POS) daily transactional velocity, distributor purchase orders, supplier fulfillment logs, and semi-structured interviews with the supervising pharmacist.

- **Primary Source**: Direct on-site pharmacy inventory audits and ERP/POS billing records.
- **Study Population**: 182 commercial pharmaceutical stock-keeping units (SKUs) spanning 10 key therapeutic categories.
- **Audit Date / Baseline**: September 20, 2026.
- **Ethical Compliance & Anonymization**: All patient identifiable information, proprietary vendor trade discounts, and commercial store licensing identifiers were removed and anonymized prior to analysis in full accordance with case study submission guidelines.

---

## 2. Raw Dataset Attribute Definitions (`data/pharmacy_stockout_raw.csv`)

| Column Name | Data Type | Units / Range | Description & Business Relevance |
| :--- | :--- | :--- | :--- |
| `Medicine_ID` | String (Categorical) | MED001 – MED182 | Unique alphanumeric identifier assigned to each monitored pharmaceutical SKU. |
| `Medicine_Name` | String (Text) | Free Text | Generic chemical composition, brand formulation name, and strength (e.g., *Paracetamol 650mg (Dolo)*, *Amoxicillin 500mg*). |
| `Category` | String (Categorical) | 10 Categories | Clinical therapeutic classification: Antibiotics, Analgesics & Antipyretics, Cardiovascular & Antihypertensives, Antidiabetics, Gastrointestinal, Respiratory & Antiasthmatics, Dermatological & Topicals, Vitamins & Mineral Supplements, Neuropsychiatric & Sedatives, Ophthalmic & ENT. |
| `Current_Stock` | Integer (Discrete) | 0 – 350 units | Physical inventory count on pharmacy shelves and storeroom on the audit baseline date. |
| `Daily_Sales` | Float (Continuous) | 1.0 – 45.0 units/day | Average daily unit sales velocity calculated from POS transaction history over a rolling 60-day operational window. |
| `Supplier_Lead_Time` | Integer (Discrete) | 2 – 14 days | Lead time in days elapsed between purchase order placement with the pharmaceutical distributor and physical receipt at the pharmacy. |
| `Reorder_Level` | Integer (Discrete) | 5 – 200 units | Operational threshold currently configured in the pharmacy's legacy system that triggers a replenishment order. |
| `Expiry_Date` | String (Date) | YYYY-MM-DD | Expiration date of the earliest active batch on shelf (batches monitored range from 3 to 32 months out). |
| `Seasonal_Demand` | String (Categorical) | 4 Classes | Epidemiological demand sensitivity pattern: `High_Winter`, `High_Monsoon`, `High_Summer`, or `Stable_All_Season`. |
| `Unit_Price_INR` | Float (Continuous) | ₹15 – ₹650 | Maximum retail price (MRP) per unit in Indian Rupees (INR). |
| `Minimum_Order_Quantity` | Integer (Discrete) | 10, 20, 30, 50, 100 units | Minimum batch quantity (MOQ) dictated by pharmaceutical manufacturers and wholesale distributors. |
| `Criticality` | String (Categorical) | Vital, Essential, Desirable | Standard healthcare **VED** priority classification: **Vital** (life-saving, immediate availability required), **Essential** (clinical importance, shortage causes clinical disruption), **Desirable** (elective, symptomatic relief). |
| `Storage_Condition` | String (Categorical) | Room Temperature, Cold Chain | Storage requirements: ambient temperature (15–25°C) versus cold chain refrigeration (2–8°C for insulins, specialized biologics). |
| `Stock_Status` | Integer (Binary Target) | 0 or 1 | Target classification variable: `1` indicates SKU is in a Stock-Out or Imminent Stock-Out state (inventory insufficient to survive lead time demand), `0` indicates SKU has adequate buffer (In Stock). |

---

## 3. Cleaned & Engineered Feature Definitions (`data/pharmacy_stockout_cleaned.csv`)

| Column Name | Data Type | Formula / Origin | Analytical Rationale |
| :--- | :--- | :--- | :--- |
| `Expiry_Months_Remaining` | Float (Continuous) | `(Expiry_Date - Base_Date) / 30.4` | Converts static expiry calendar dates into continuous shelf-life runout horizons. |
| `Days_of_Inventory` (DOI) | Float (Continuous) | `Current_Stock / Daily_Sales` | Number of operational days the existing shelf inventory will sustain consumer demand before total stock depletion. |
| `Lead_Time_Demand` (LTD) | Float (Continuous) | `Daily_Sales * Supplier_Lead_Time` | Total expected units consumed during the replenishment cycle while waiting for distributor delivery. |
| `Safety_Stock_Buffer` | Float (Continuous) | `Current_Stock - Lead_Time_Demand` | Net safety margin in units. Negative or near-zero values signal severe vulnerability to stockout. |
| `Buffer_Ratio` | Float (Continuous) | `Current_Stock / (Lead_Time_Demand + ε)` | Dimensionless resilience ratio. Values < 1.0 indicate current stock cannot satisfy average lead-time demand. |
| `Stock_to_Reorder_Ratio` | Float (Continuous) | `Current_Stock / (Reorder_Level + ε)` | Ratio of on-hand inventory to the current store reorder threshold. |
| `Inventory_Valuation_INR` | Float (Continuous) | `Current_Stock * Unit_Price_INR` | Total financial capital locked in current inventory per SKU. |

---

## 4. Class Distribution & Summary Statistics

- **Total SKUs Analyzed**: 182
- **Adequately Stocked (`Stock_Status = 0`)**: 131 SKUs (~72.0%)
- **Stock-Out / High Risk (`Stock_Status = 1`)**: 51 SKUs (~28.0%)
- **Mean Daily Sales**: 9.42 units/day (Std: 6.81, Min: 1.1, Max: 32.4)
- **Mean Supplier Lead Time**: 5.38 days (Min: 2 days, Max: 14 days)
- **Therapeutic Categories**: 10 major classes represented evenly (14 to 19 SKUs each)
