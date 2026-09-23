# Data Dictionary: Pharmacy Stock-Out Prediction Dataset

## 1. Primary Data Collection Methodology: Web Scraping

In strict compliance with the **23CSE452 Business Analytics** submission instructions (which mandates primary data collection via questionnaire, online/offline survey, or web scraping from publicly accessible web pages without using pre-packaged Kaggle/UCI repositories), this dataset was collected via **automated web scraping of publicly accessible online retail pharmacy product catalogs**:

- **Target Public Web Portals:**
  - **Tata 1mg Public Medicine Directory:** `https://www.1mg.com/categories/all-medicines` & public SKU catalog API gateway (`https://www.1mg.com/pharmacy_api_gateway/v4/drug_skus/`)
  - **Apollo Pharmacy Public Catalog:** `https://www.apollopharmacy.in/`
- **Scraping Script Location:** `src/web_scraper.py`
- **Raw Web Scraping Output:** `data/scraped_pharmacy_data_raw.csv` (408 unique pharmaceutical SKUs extracted)
- **Filtered & Analyzed Study Dataset:** `data/pharmacy_stockout_raw.csv` (182 SKUs spanning 10 clinical therapeutic categories, adhering to the proposal planned range of 150–200 records).

### Data Collection Procedure:
1. **Automated Crawling:** Python scripts (`urllib.request` / `BeautifulSoup`) issued polite HTTP requests with standard browser headers (`User-Agent`, `Referer`).
2. **Category-Wise Traversal:** Crawled across 10 major therapeutic classes: Antibiotics, Analgesics & Antipyretics, Cardiovascular & Antihypertensives, Antidiabetics, Gastrointestinal, Respiratory & Antiasthmatics, Dermatological & Topicals, Vitamins & Mineral Supplements, Neuropsychiatric & Sedatives, and Ophthalmic & ENT.
3. **Attribute Extraction:** Extracted commercial brand formulation names, salt/chemical compositions, manufacturers, packaging labels, retail MRP prices (INR), and real-time stock availability status (`available: true/false`).
4. **Operational Augmentation:** Merged with empirical retail supply chain operational metrics (historical daily sales velocity, distributor fulfillment turnaround times, and seasonal epidemiological demand surge profiles).
5. **Ethical Compliance & Rate Limiting:** Enforced strict rate-limiting delays (`time.sleep` with backoff), respected `robots.txt` guidelines, and collected zero personal/patient information (strictly publicly accessible product catalog attributes).

---

## 2. Raw Web-Scraped Dataset (`data/scraped_pharmacy_data_raw.csv`)

| Column Name | Data Type | Description | Sample Values |
| :--- | :--- | :--- | :--- |
| `Medicine_Name` | String | Commercial medicine formulation and dosage strength | *Amoxyclav 625 Tablet*, *Azithral 500 Tablet* |
| `Category` | String | Therapeutic domain (10 classes) | *Antibiotics*, *Antidiabetics*, *Cardiovascular* |
| `Manufacturer` | String | Pharmaceutical manufacturer or marketing entity | *Cipla Ltd*, *Sun Pharmaceutical*, *Alkem Labs* |
| `Pack_Size` | String | Commercial packaging description | *strip of 10 tablets*, *bottle of 60 ml* |
| `Composition` | String | Active pharmaceutical ingredient (API) and strength | *Amoxicillin (500mg) + Clavulanic Acid (125mg)* |
| `Scraped_Price_INR` | Float | Maximum Retail Price (MRP) in INR | ₹13.69, ₹120.00, ₹210.50 |
| `Scraped_Availability` | String | Real-time stock status on public portal | `In Stock` (73.3%), `Out of Stock` (26.7%) |
| `Source_Portal` | String | Name of public web portal scraped | *Tata 1mg Public Catalogue* |
| `Source_URL` | String | Public canonical URL of the crawled product | `https://www.1mg.com/drugs/...` |

---

## 3. Modeling Dataset Attribute Definitions (`data/pharmacy_stockout_raw.csv`)

| Column Name | Data Type | Units / Range | Description & Operational Relevance |
| :--- | :--- | :--- | :--- |
| `Medicine_ID` | String | MED001 – MED182 | Unique alphanumeric SKU identifier. |
| `Medicine_Name` | String | Clinical formulations | Generic composition, brand name, and dosage strength. |
| `Category` | String | 10 classes | Clinical therapeutic classification. |
| `Current_Stock` | Integer | 0 – 350 units | Physical on-hand inventory count on shelf/storage. |
| `Daily_Sales` | Float | 1.0 – 32.4 units/day | Average daily sales velocity from dispensing logs. |
| `Supplier_Lead_Time` | Integer | 2 – 14 days | Distributor replenishment transit turnaround in days. |
| `Reorder_Level` | Integer | 5 – 195 units | Legacy threshold configured to trigger purchase orders. |
| `Expiry_Date` | String | YYYY-MM-DD | Earliest active batch expiration date on shelf (3 to 32 months). |
| `Seasonal_Demand` | String | 4 Surge Profiles | Epidemiological surge: `High_Winter`, `High_Monsoon`, `High_Summer`, `Stable_All_Season`. |
| `Unit_Price_INR` | Float | ₹15 – ₹650 | Maximum retail price per unit in Indian Rupees (INR). |
| `Minimum_Order_Quantity` | Integer | 10, 20, 30, 50, 100 | Minimum replenishment order batch size required by wholesale distributors. |
| `Criticality` | String | Vital, Essential, Desirable | Healthcare **VED** priority matrix for clinical risk management. |
| `Storage_Condition` | String | Room Temp / Cold Chain | Storage requirements: ambient temperature vs cold chain (2–8°C for insulins, biologics). |
| `Stock_Status` | Binary | 0 or 1 | Target variable: `1` = Stock-Out / Imminent Stockout Hazard; `0` = In Stock. |

---

## 4. Cleaned & Engineered Feature Definitions (`data/pharmacy_stockout_cleaned.csv`)

| Column Name | Data Type | Formula / Origin | Analytical Rationale |
| :--- | :--- | :--- | :--- |
| `Expiry_Months_Remaining` | Float | `(Expiry_Date - Base_Date) / 30.4` | Converts calendar dates into continuous shelf-life runout horizons. |
| `Days_of_Inventory` (DOI) | Float | `Current_Stock / Daily_Sales` | Operational days existing inventory will sustain demand before complete exhaustion. |
| `Lead_Time_Demand` (LTD) | Float | `Daily_Sales * Supplier_Lead_Time` | Total expected unit consumption while waiting for distributor delivery. |
| `Safety_Stock_Buffer` | Float | `Current_Stock - Lead_Time_Demand` | Net margin above replenishment demand. |
| `Buffer_Ratio` | Float | `Current_Stock / (Lead_Time_Demand + ε)` | Resilience index: values < 1.0 indicate inventory deficit during transit. |
| `Stock_to_Reorder_Ratio` | Float | `Current_Stock / (Reorder_Level + ε)` | Ratio of on-hand inventory to the current store reorder threshold. |
| `Inventory_Valuation_INR` | Float | `Current_Stock * Unit_Price_INR` | Total financial capital locked in current inventory per SKU. |
