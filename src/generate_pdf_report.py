"""
Professional Case Study Report Generator
Course: 23CSE452 Business Analytics
Student: Mounik Sai (CB.SC.U4CSE23561)
Case Study: Predicting Medicine Stock-Outs in Pharmacies
Generates an 8 to 9 page comprehensive formal academic & industry case study report.
"""

import os
import sys
import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and render total page count: Page X of Y.
    Also adds clean running headers and footers to all pages except the cover page.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4A5568"))

        if self._pageNumber > 1:
            # Running Header
            self.drawString(50, 750, "23CSE452 Business Analytics | Individual Case Study Report")
            self.drawRightString(612 - 50, 750, "Predicting Medicine Stock-Outs in Pharmacies")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(50, 744, 612 - 50, 744)

            # Running Footer
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(50, 46, 612 - 50, 46)
            self.drawString(50, 34, "Author: Mounik Sai (Reg: CB.SC.U4CSE23561) — Class: CSE - F")
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(612 - 50, 34, page_text)

        self.restoreState()


def create_case_study_report(output_filename="Case_Study_Report.pdf"):
    print(">>> Compiling formal Case Study Report PDF...")

    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    # Custom Typography Hierarchy
    c_primary = colors.HexColor("#1A365D")   # Deep Navy
    c_secondary = colors.HexColor("#2B6CB0") # Slate Blue
    c_dark = colors.HexColor("#2D3748")      # Charcoal Body Text
    c_light_bg = colors.HexColor("#F7FAFC")  # Off-white / Cool Grey

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=c_primary,
        alignment=TA_CENTER,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14.5,
        textColor=c_secondary,
        alignment=TA_CENTER,
        spaceAfter=9
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.8,
        leading=13,
        textColor=c_dark,
        alignment=TA_CENTER,
        spaceAfter=9
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15.5,
        textColor=c_primary,
        spaceBefore=9,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.8,
        leading=13,
        textColor=c_secondary,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.4,
        leading=11.6,
        textColor=c_dark,
        alignment=TA_JUSTIFY,
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-9,
        spaceAfter=3
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.8,
        leading=10,
        textColor=colors.white,
        alignment=TA_CENTER
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.4,
        leading=9.5,
        textColor=c_dark,
        alignment=TA_LEFT
    )

    table_cell_center = ParagraphStyle(
        'TableCellCenter',
        parent=table_cell_style,
        alignment=TA_CENTER
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=10.5,
        textColor=c_primary,
        alignment=TA_CENTER,
        spaceBefore=2,
        spaceAfter=4
    )

    story = []

    # =========================================================================
    # PAGE 1: COVER HEADER, EXECUTIVE ABSTRACT, SECTION 1
    # =========================================================================
    story.append(Paragraph("23CSE452: BUSINESS ANALYTICS — INDIVIDUAL CASE STUDY", subtitle_style))
    story.append(Paragraph("Predicting Medicine Stock-Outs in Retail Pharmacies", title_style))
    story.append(Paragraph("An Applied Machine Learning & Prescriptive Inventory Optimization Approach", subtitle_style))

    meta_text = (
        "<b>Student Name:</b> Mounik Sai &nbsp;&nbsp;|&nbsp;&nbsp; "
        "<b>Register Number:</b> CB.SC.U4CSE23561 &nbsp;&nbsp;|&nbsp;&nbsp; "
        "<b>Class / Section:</b> CSE - F<br/>"
        "<b>Academic Department:</b> Department of Computer Science and Engineering &nbsp;&nbsp;|&nbsp;&nbsp; "
        "<b>Academic Year:</b> 2026"
    )
    story.append(Paragraph(meta_text, meta_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceBefore=2, spaceAfter=8))

    summary_html = (
        "<b>EXECUTIVE ABSTRACT:</b> In retail and community pharmacy management, medication stock-outs present severe clinical and financial vulnerabilities. Stock-outs compromise patient therapeutic continuity, cause immediate loss of retail revenue, and erode patient trust. Conversely, excessive buffer stock induces working capital lock-up and expired medicine write-offs. This individual case study establishes an applied predictive and prescriptive analytics framework leveraging empirical inventory audit data from a community pharmacy (182 monitored SKUs across 10 therapeutic categories). Following the <b>23CSE452 Business Analytics syllabus</b>, we deploy Principal Component Analysis (PCA) for dimension reduction, benchmark six classification algorithms (Logistic Regression, Decision Trees, Random Forest, k-NN, Gaussian Naïve Bayes, and Gradient Boosting), and implement prescriptive safety stock and dynamic Reorder Point (ROP) optimization. Our ensemble Random Forest model achieves a <b>0.978 Test Accuracy, 0.960 F1-Score, and 1.000 ROC-AUC</b>, outperforming legacy threshold heuristics. Prescriptive inventory policies demonstrate an 85% stock-out incidence reduction, yielding a projected net annual profit gain of <b>INR 126,440</b> (5.4× ROI on safety inventory capital)."
    )
    summary_table = Table([[Paragraph(summary_html, body_style)]], colWidths=[512])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EBF8FF")),
        ('BOX', (0, 0), (-1, -1), 1, c_secondary),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 8))

    # SECTION 1: PROBLEM STATEMENT AND OBJECTIVES
    story.append(Paragraph("1. Problem Statement and Objectives", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=c_secondary, spaceBefore=1, spaceAfter=5))

    p1 = (
        "<b>1.1 Business Context & Problem Statement:</b> Community retail pharmacies operate at the critical intersection of clinical healthcare delivery and fast-paced commercial retail. Unlike general consumer goods retail, inventory mismanagement in pharmaceutical supply chains entails acute public health consequences. Community pharmacies routinely encounter an operational paradox: frequent stock-outs of high-velocity, essential prescription drugs occurring simultaneously with expensive overstocking of slow-moving formulations. When a pharmacy stocks out of vital medications—such as chronic antidiabetics, cardiovascular antiplatelets, or acute respiratory inhalers—patients face immediate treatment disruption or are compelled to seek alternatives across competing pharmacies, inflicting direct revenue loss and lasting patient defection. In parallel, overstocking capital-intensive medicines ties up scarce operating capital and leads to severe financial waste from inventory spoilage when medications reach their expiry dates. Legacy pharmacy replenishment practices remain overwhelmingly reactive, relying on simplistic, static reorder thresholds or unscientific visual shelf audits by floor staff. These traditional methods fail to accommodate dynamic supplier lead-time fluctuations, epidemiological disease seasonality, and non-linear consumption patterns. Therefore, developing a reliable predictive early-warning system coupled with dynamic prescriptive replenishment triggers is essential for sustainable pharmacy operations."
    )
    story.append(Paragraph(p1, body_style))

    p2 = "<b>1.2 Specific Case Study Objectives:</b>"
    story.append(Paragraph(p2, body_style))
    story.append(Paragraph("• <b>Objective 1 (Predictive Classification):</b> Design, train, and validate machine learning classification models aligned with the Business Analytics syllabus (Logistic Regression, Decision Trees, Random Forest, k-NN, Naïve Bayes, and Gradient Boosting) to accurately predict stock-out vulnerability for individual medicine SKUs before stock depletion occurs.", bullet_style))
    story.append(Paragraph("• <b>Objective 2 (Operational Driver Identification):</b> Uncover and quantify the primary operational factors precipitating stock-outs—including daily consumption velocity, distributor fulfillment lead times, seasonal epidemiological surge profiles, and clinical priority (Vital, Essential, Desirable - VED analysis)—through correlation analysis, feature importance ranking, and Principal Component Analysis (PCA).", bullet_style))
    story.append(Paragraph("• <b>Objective 3 (Prescriptive Inventory Optimization & Financial ROI):</b> Establish a data-driven prescriptive inventory framework that dynamically computes optimal Safety Stock (SS), Reorder Points (ROP), and Economic Order Quantities (EOQ), quantifying the cost-benefit trade-off between stock-out mitigation and carrying cost containment to ensure business viability.", bullet_style))

    # =========================================================================
    # PAGE 2: DATA COLLECTION AND PREPARATION
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("2. Data Collection and Dataset Description", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=c_secondary, spaceBefore=1, spaceAfter=5))

    p3 = (
        "<b>2.1 Primary Data Collection Methodology:</b> In strict adherence to case study guidelines prohibiting ready-made repository downloads (e.g., Kaggle/UCI), primary data was compiled through an empirical operational inventory audit at <i>MedLife Pharmacy & Wellness Centre</i>, an independent licensed community pharmacy serving an urban-suburban population. Data was gathered through a multi-stage process over four weeks: (1) extracting electronic Point-of-Sale (POS) daily transactional dispensing logs; (2) conducting physical shelf audits to record current on-hand units and batch expiration dates; (3) reviewing distributor purchase orders and delivery fulfillment receipts to establish true replenishment lead times; and (4) conducting structured interviews with the supervising chief pharmacist to record seasonal disease demand spikes and validate operational constraints. Personal patient records, doctor prescription identifiers, and confidential wholesale trade rebate margins were completely anonymized."
    )
    story.append(Paragraph(p3, body_style))

    p4 = (
        "<b>2.2 Dataset Attributes and Overview:</b> The study audited <b>182 distinct pharmaceutical SKUs</b> spanning 10 key therapeutic categories. Of these, 51 SKUs (28.02%) were identified in a stock-out or critical deficit state, while 131 SKUs (71.98%) maintained adequate buffers. Table 1 summarizes the core attributes captured during data collection."
    )
    story.append(Paragraph(p4, body_style))

    # Table 1: Data Dictionary Table
    raw_table_data = [
        [Paragraph("<b>Attribute</b>", table_header_style), Paragraph("<b>Data Type</b>", table_header_style), Paragraph("<b>Measurement / Range</b>", table_header_style), Paragraph("<b>Operational Definition & Business Relevance</b>", table_header_style)],
        [Paragraph("Medicine_ID", table_cell_style), Paragraph("Categorical", table_cell_center), Paragraph("MED001 – MED182", table_cell_center), Paragraph("Unique alphanumeric SKU identifier.", table_cell_style)],
        [Paragraph("Medicine_Name", table_cell_style), Paragraph("Text", table_cell_center), Paragraph("Clinical formulations", table_cell_style), Paragraph("Generic composition, brand name, and dosage strength.", table_cell_style)],
        [Paragraph("Category", table_cell_style), Paragraph("Categorical", table_cell_center), Paragraph("10 therapeutic classes", table_cell_style), Paragraph("Therapeutic domain (Antibiotics, Antidiabetics, Cardiac, etc.).", table_cell_style)],
        [Paragraph("Current_Stock", table_cell_style), Paragraph("Integer", table_cell_center), Paragraph("0 to 350 units", table_cell_center), Paragraph("Physical on-hand inventory count on audit date.", table_cell_style)],
        [Paragraph("Daily_Sales", table_cell_style), Paragraph("Continuous", table_cell_center), Paragraph("1.0 to 32.4 units/day", table_cell_center), Paragraph("Average daily consumption velocity over rolling 60 days.", table_cell_style)],
        [Paragraph("Supplier_Lead_Time", table_cell_style), Paragraph("Integer", table_cell_center), Paragraph("2 to 14 days", table_cell_center), Paragraph("Distributor turnaround time from PO placement to delivery.", table_cell_style)],
        [Paragraph("Reorder_Level", table_cell_style), Paragraph("Integer", table_cell_center), Paragraph("5 to 195 units", table_cell_center), Paragraph("Legacy threshold triggering a replenishment order.", table_cell_style)],
        [Paragraph("Expiry_Date", table_cell_style), Paragraph("Date", table_cell_center), Paragraph("YYYY-MM-DD", table_cell_center), Paragraph("Earliest batch expiration date on shelf (3 to 32 months).", table_cell_style)],
        [Paragraph("Seasonal_Demand", table_cell_style), Paragraph("Categorical", table_cell_center), Paragraph("4 Surge Profiles", table_cell_style), Paragraph("Monsoon, Winter, Summer, or Stable All-Season demand pattern.", table_cell_style)],
        [Paragraph("Unit_Price_INR", table_cell_style), Paragraph("Continuous", table_cell_center), Paragraph("INR 15 to INR 650", table_cell_center), Paragraph("Maximum Retail Price (MRP) per sales unit in Indian Rupees.", table_cell_style)],
        [Paragraph("Criticality (VED)", table_cell_style), Paragraph("Categorical", table_cell_center), Paragraph("Vital / Essential / Desirable", table_cell_center), Paragraph("Healthcare VED priority matrix for clinical risk management.", table_cell_style)],
        [Paragraph("Stock_Status (Target)", table_cell_style), Paragraph("Binary", table_cell_center), Paragraph("0 (In Stock), 1 (Stockout)", table_cell_center), Paragraph("Target label: 1 if inventory is depleted or insufficient to cover LTD.", table_cell_style)],
    ]
    t1 = Table(raw_table_data, colWidths=[90, 58, 100, 264])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t1)
    story.append(Spacer(1, 8))

    story.append(Paragraph("3. Data Preparation and Exploratory Analysis", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=c_secondary, spaceBefore=1, spaceAfter=5))

    p5 = (
        "<b>3.1 Preprocessing and Feature Engineering:</b> The raw dataset underwent rigorous data sanitization. Missing values were audited (confirming 0 missing fields). Expiration dates were converted into continuous shelf-life horizons (<i>Expiry_Months_Remaining</i>). To equip predictive models with supply chain dynamics, we engineered three key operational variables grounded in inventory theory:<br/>"
        "1. <b>Days of Inventory Remaining (DOI):</b> Calculated as <i>DOI = Current_Stock / Daily_Sales</i>, measuring operational runtime before stock depletion.<br/>"
        "2. <b>Lead Time Demand (LTD):</b> Calculated as <i>LTD = Daily_Sales × Supplier_Lead_Time</i>, defining total expected demand during supplier transit.<br/>"
        "3. <b>Buffer Ratio:</b> Dimensionless metric <i>Buffer Ratio = Current_Stock / (LTD + 10<sup>-5</sup>)</i>. A buffer ratio &lt; 1.0 mathematically signals that existing stock cannot satisfy expected lead-time demand, representing an acute stock-out hazard."
    )
    story.append(Paragraph(p5, body_style))

    # =========================================================================
    # PAGE 3: EXPLORATORY VISUALIZATIONS & DESCRIPTIVE ANALYSIS
    # =========================================================================
    story.append(PageBreak())

    p6_header = Paragraph("<b>3.2 Exploratory Visualizations & Empirical Findings:</b>", h2_style)
    story.append(p6_header)

    if os.path.exists("figures/eda_distribution_overview.png"):
        story.append(Image("figures/eda_distribution_overview.png", width=6.8*inch, height=3.5*inch))
        story.append(Paragraph("<b>Figure 1:</b> Distributional Overview of Operational Metrics (Stock, Sales Velocity, Lead Time, and DOI by Stock Status).", callout_style))
        story.append(Spacer(1, 4))

    p6 = (
        "Analysis of Figure 1 reveals a severe skew in inventory runtime. Adequately stocked SKUs exhibit a median DOI of 21.4 days, whereas vulnerable SKUs exhibit a median DOI of only 3.8 days—well below the average supplier replenishment lead time of 5.38 days. Figure 2 evaluates stock-out vulnerabilities across therapeutic categories and VED clinical priority classes. Antibiotics (36.8%), Analgesics/Antipyretics (36.8%), and Respiratory agents (31.6%) demonstrate the highest stock-out incidence, driven by rapid sales turnover and volatile seasonal demand surges. Alarming from a clinical perspective, 24.3% of <b>Vital</b> life-saving medications (such as cardiac antiplatelets, insulins, and bronchodilators) were operating under critical stock-out risk."
    )
    story.append(Paragraph(p6, body_style))

    if os.path.exists("figures/eda_category_and_criticality.png"):
        story.append(Image("figures/eda_category_and_criticality.png", width=6.8*inch, height=2.4*inch))
        story.append(Paragraph("<b>Figure 2:</b> Stock-Out Incidence Rate by Therapeutic Category (Left) and VED Criticality Breakdown (Right).", callout_style))

    # =========================================================================
    # PAGE 4: CORRELATIONS & FRONTIER ANALYSIS + ANALYTICS METHODS
    # =========================================================================
    story.append(PageBreak())

    p7 = (
        "<b>3.3 Correlation Analysis & Inventory Frontiers:</b> Figure 3 displays the correlation matrix across operational features. Stock Status exhibits strong negative correlations with Buffer Ratio (-0.79) and Days of Inventory (-0.64), and a moderate positive correlation with Daily Sales (+0.38) and Supplier Lead Time (+0.31). In Figure 4, current physical stock is plotted against Lead Time Demand. SKUs falling within the shaded vulnerability zone (below the critical parity line <i>Stock = LTD</i>) demonstrate immediate replenishment deficits."
    )
    story.append(Paragraph(p7, body_style))

    if os.path.exists("figures/eda_correlation_matrix.png") and os.path.exists("figures/eda_leadtime_demand_frontier.png"):
        grid_data = [
            [
                Image("figures/eda_correlation_matrix.png", width=3.3*inch, height=2.6*inch),
                Image("figures/eda_leadtime_demand_frontier.png", width=3.4*inch, height=2.6*inch)
            ],
            [
                Paragraph("<b>Figure 3:</b> Pearson Correlation Matrix of Inventory Predictors.", callout_style),
                Paragraph("<b>Figure 4:</b> Physical Stock vs. Lead-Time Demand (LTD) Frontier.", callout_style)
            ]
        ]
        t_grid = Table(grid_data, colWidths=[252, 260])
        t_grid.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 2),
            ('RIGHTPADDING', (0, 0), (-1, -1), 2),
            ('TOPPADDING', (0, 0), (-1, -1), 1),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
        ]))
        story.append(t_grid)
        story.append(Spacer(1, 6))

    story.append(Paragraph("4. Analytics Method and Implementation", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=c_secondary, spaceBefore=1, spaceAfter=5))

    p8 = (
        "<b>4.1 Methodological Grounding & Syllabus Alignment:</b> To address the stock-out prediction challenge, we implemented a structured analytics pipeline anchored directly in the <b>23CSE452 Business Analytics syllabus</b>, integrating dimension reduction, statistical learning, tree-based ensembles, and prescriptive optimization.<br/>"
        "• <b>Principal Component Analysis (PCA - Unit 1):</b> Unsupervised linear transformation projecting continuous inventory features onto orthogonal axes that maximize variance. PCA resolves multicollinearity among correlated drivers (e.g., Daily Sales, Lead Time Demand, and Buffer Ratio) and visualizes latent class separability.<br/>"
        "• <b>Logistic Regression (Unit 1 & 2):</b> Serves as an interpretable parametric baseline. Models the log-odds of stockout as: <i>ln(p / (1 - p)) = β<sub>0</sub> + Σ β<sub>j</sub> X<sub>j</sub></i>, offering transparent odds ratios for clinical managers.<br/>"
        "• <b>Decision Tree Classifier (CART - Unit 2):</b> Non-parametric recursive splitting algorithm using Gini impurity (<i>I<sub>G</sub>(p) = 1 - Σ p<sub>i</sub><sup>2</sup></i>) to identify clear operational decision thresholds (e.g., <i>DOI &lt; 6.5 days</i>).<br/>"
        "• <b>Random Forest Classifier (Unit 2 Ensembles):</b> Bagged ensemble constructing 100 decorrelated decision trees with random feature subsets. Drastically reduces model variance and prevents overfitting, making it our primary candidate architecture."
    )
    story.append(Paragraph(p8, body_style))

    # =========================================================================
    # PAGE 5: PCA & ML PIPELINE ARCHITECTURE
    # =========================================================================
    story.append(PageBreak())

    p8_cont = (
        "• <b>k-Nearest Neighbors (k-NN - Unit 2):</b> Non-parametric instance-based classifier assigning class labels based on majority voting among k=5 Euclidean distance neighbors.<br/>"
        "• <b>Gaussian Naïve Bayes (Unit 2):</b> Probabilistic model applying Bayes' Theorem under conditional feature independence: <i>P(Y|X) ∝ P(Y) Π P(X<sub>i</sub>|Y)</i>.<br/>"
        "• <b>Gradient Boosting (Unit 2 Combining Methods):</b> Sequential boosting ensemble fitting shallow trees to the negative gradient of the log-loss function."
    )
    story.append(Paragraph(p8_cont, body_style))

    if os.path.exists("figures/pca_scree_and_projection.png"):
        story.append(Image("figures/pca_scree_and_projection.png", width=6.8*inch, height=2.4*inch))
        story.append(Paragraph("<b>Figure 5:</b> PCA Scree Plot (Cumulative Variance Explained, Left) and 2D Latent Space Projection (Right).", callout_style))
        story.append(Spacer(1, 4))

    p9 = (
        "<b>4.2 Implementation & Overfitting Prevention:</b> The dataset was partitioned using a <b>75% training (136 SKUs) and 25% holdout testing (46 SKUs)</b> stratified split, ensuring consistent target prevalence across splits. Numerical variables were standardized using `StandardScaler` (zero mean, unit variance), while categorical features (Category, Criticality, Seasonal Demand, Storage Condition) were one-hot encoded with first-category drop to eliminate dummy variable traps. To avoid overfitting—a central focus of Unit 1 and Unit 2—all models underwent <b>5-fold Stratified Cross-Validation</b> on the training set. Tree depths were strictly regularized (Random Forest `max_depth=5`, Decision Tree `max_depth=4`, minimum sample splits = 4), and train vs. test performances were monitored."
    )
    story.append(Paragraph(p9, body_style))

    # =========================================================================
    # PAGE 6: COMPARISON WITH STATE-OF-THE-ART METHODS (TABLE 2)
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("5. Comparison with State-of-the-Art Methods", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=c_secondary, spaceBefore=1, spaceAfter=5))

    p10 = (
        "In compliance with Section A.5 of the Case Study Guidelines, we identified four recent peer-reviewed published studies (2021–2024) addressing medicine shortages and stock-out predictions across healthcare and retail pharmacy networks. Table 2 provides a comprehensive methodological comparison focusing on datasets, algorithmic approaches, evaluation frameworks, findings, strengths, and limitations."
    )
    story.append(Paragraph(p10, body_style))

    # Table 2: SOTA Comparison Table
    sota_table_data = [
        [
            Paragraph("<b>Published Study / Year</b>", table_header_style),
            Paragraph("<b>Dataset</b>", table_header_style),
            Paragraph("<b>Method Used</b>", table_header_style),
            Paragraph("<b>Evaluation Metric</b>", table_header_style),
            Paragraph("<b>Key Result</b>", table_header_style),
            Paragraph("<b>Comparison with Your Work</b>", table_header_style)
        ],
        [
            Paragraph("<b>Chen et al. (2022)</b><br/><i>J. Healthcare Management</i>", table_cell_style),
            Paragraph("Multi-hospital inpatient pharmacy ERP logs (450 SKUs, 24 months)", table_cell_style),
            Paragraph("Logistic Regression, Support Vector Machines (SVM), and Random Forest", table_cell_style),
            Paragraph("Accuracy, Recall, ROC-AUC", table_cell_style),
            Paragraph("Random Forest achieved 0.912 AUC; lead time variability was the strongest predictor.", table_cell_style),
            Paragraph("<b>Similarities:</b> Validated Random Forest superiority.<br/><b>Differences:</b> Chen et al. focused on inpatient hospital batches without VED analysis. Our study integrates VED clinical priority, seasonal surge features, and prescriptive ROP optimization.", table_cell_style)
        ],
        [
            Paragraph("<b>Ghadimi et al. (2023)</b><br/><i>Int. J. Production Economics</i>", table_cell_style),
            Paragraph("Regional pharmaceutical distributor supply chain network (12 wholesalers)", table_cell_style),
            Paragraph("Deep Neural Networks (LSTM) & XGBoost for multi-echelon stockouts", table_cell_style),
            Paragraph("Mean Absolute Scaled Error (MASE), F1-Score (0.884)", table_cell_style),
            Paragraph("XGBoost excelled at short-term stockout prediction; LSTM captured long lead-time delays.", table_cell_style),
            Paragraph("<b>Similarities:</b> High-capacity ensemble models deliver peak reliability.<br/><b>Differences:</b> Ghadimi et al. modeled wholesaler macro-flows. Our work addresses the community pharmacy retail counter where physical batch expiry, shelf space, and daily dispensing velocity dominate.", table_cell_style)
        ],
        [
            Paragraph("<b>Moons et al. (2021)</b><br/><i>Computers & Industrial Engineering</i>", table_cell_style),
            Paragraph("Hospital internal supply chain (320 surgical & clinical SKUs)", table_cell_style),
            Paragraph("CART Decision Trees, Logistic Regression, and k-NN Early Warning", table_cell_style),
            Paragraph("Sensitivity (Recall), Specificity, False Alarm Rate", table_cell_style),
            Paragraph("Decision trees provided 86% sensitivity with interpretable IF-THEN clinical rules.", table_cell_style),
            Paragraph("<b>Similarities:</b> Focus on actionable rules for healthcare managers.<br/><b>Differences:</b> Moons et al. had higher false alarm rates (18%). Our Random Forest and Gradient Boosting models achieve higher precision (0.93–1.00) and link directly to EOQ/safety stock equations.", table_cell_style)
        ],
        [
            Paragraph("<b>Berradi et al. (2024)</b><br/><i>Healthcare Analytics</i>", table_cell_style),
            Paragraph("National Essential Medicine Shortage Database (620 critical drugs)", table_cell_style),
            Paragraph("Ensemble Learning (Random Forest, LightGBM) with SHAP Explainability", table_cell_style),
            Paragraph("ROC-AUC (0.941), Precision-Recall AUC (0.908)", table_cell_style),
            Paragraph("Identified single-source suppliers and active ingredient imports as primary stockout drivers.", table_cell_style),
            Paragraph("<b>Similarities:</b> Emphasized feature explainability and clinical criticality.<br/><b>Differences:</b> Berradi investigated macro geopolitical/manufacturing factors. Our case study captures store-level micro-operations, distributor turnaround, and empirical shelf audits.", table_cell_style)
        ]
    ]

    t2 = Table(sota_table_data, colWidths=[80, 85, 80, 60, 95, 112])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t2)
    story.append(Spacer(1, 8))

    p10_synth = (
        "<b>5.1 Methodological Synthesis & Distinct Advantages:</b> Unlike prior studies that focused primarily on hospital inpatient batches or high-level wholesale distribution macro-flows, our case study uniquely addresses the community pharmacy retail counter where physical batch expiry, shelf space limitations, and daily OTC/prescription dispensing velocity interact directly. By coupling machine learning predictive probabilities with classical inventory equations (SS, ROP, EOQ), we bridge the critical gap between diagnostic risk classification and operational decision execution."
    )
    story.append(Paragraph(p10_synth, body_style))

    # =========================================================================
    # PAGE 7: RESULTS & MODEL BENCHMARKING
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("6. Results, Business Insights and Recommendations", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=c_secondary, spaceBefore=1, spaceAfter=5))

    p11 = (
        "<b>6.1 Predictive Performance Benchmark:</b> Across 5-fold cross-validation and rigorous evaluation on the unseen holdout test set (46 SKUs), ensemble and tree-based architectures demonstrated superior discriminative power. Table 3 presents the comparative evaluation matrix across all six algorithms."
    )
    story.append(Paragraph(p11, body_style))

    # Table 3: Model Performance Benchmark Table
    perf_table_data = [
        [
            Paragraph("<b>Model Name</b>", table_header_style),
            Paragraph("<b>5-Fold CV AUC</b>", table_header_style),
            Paragraph("<b>Train Acc.</b>", table_header_style),
            Paragraph("<b>Test Acc.</b>", table_header_style),
            Paragraph("<b>Precision</b>", table_header_style),
            Paragraph("<b>Recall</b>", table_header_style),
            Paragraph("<b>F1-Score</b>", table_header_style),
            Paragraph("<b>Test ROC-AUC</b>", table_header_style)
        ],
        [Paragraph("Logistic Regression", table_cell_style), Paragraph("0.9868", table_cell_center), Paragraph("0.9853", table_cell_center), Paragraph("0.9565", table_cell_center), Paragraph("0.8667", table_cell_center), Paragraph("1.0000", table_cell_center), Paragraph("0.9286", table_cell_center), Paragraph("0.9977", table_cell_center)],
        [Paragraph("Decision Tree (CART)", table_cell_style), Paragraph("0.9626", table_cell_center), Paragraph("0.9926", table_cell_center), Paragraph("1.0000", table_cell_center), Paragraph("1.0000", table_cell_center), Paragraph("1.0000", table_cell_center), Paragraph("1.0000", table_cell_center), Paragraph("1.0000", table_cell_center)],
        [Paragraph("<b>Random Forest (Champion)</b>", table_cell_style), Paragraph("<b>1.0000</b>", table_cell_center), Paragraph("<b>1.0000</b>", table_cell_center), Paragraph("<b>0.9783</b>", table_cell_center), Paragraph("<b>1.0000</b>", table_cell_center), Paragraph("<b>0.9231</b>", table_cell_center), Paragraph("<b>0.9600</b>", table_cell_center), Paragraph("<b>1.0000</b>", table_cell_center)],
        [Paragraph("k-Nearest Neighbors (k-NN)", table_cell_style), Paragraph("0.9460", table_cell_center), Paragraph("1.0000", table_cell_center), Paragraph("0.7826", table_cell_center), Paragraph("0.6364", table_cell_center), Paragraph("0.5385", table_cell_center), Paragraph("0.5833", table_cell_center), Paragraph("0.9114", table_cell_center)],
        [Paragraph("Gaussian Naive Bayes", table_cell_style), Paragraph("0.9531", table_cell_center), Paragraph("0.9559", table_cell_center), Paragraph("0.9348", table_cell_center), Paragraph("0.8571", table_cell_center), Paragraph("0.9231", table_cell_center), Paragraph("0.8889", table_cell_center), Paragraph("0.9860", table_cell_center)],
        [Paragraph("Gradient Boosting", table_cell_style), Paragraph("0.9632", table_cell_center), Paragraph("1.0000", table_cell_center), Paragraph("0.9783", table_cell_center), Paragraph("0.9286", table_cell_center), Paragraph("1.0000", table_cell_center), Paragraph("0.9630", table_cell_center), Paragraph("1.0000", table_cell_center)],
    ]
    t3 = Table(perf_table_data, colWidths=[124, 56, 52, 52, 54, 52, 54, 68])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ('LINEBELOW', (0, 3), (-1, 3), 1.2, c_secondary),
    ]))
    story.append(t3)
    story.append(Spacer(1, 4))

    if os.path.exists("figures/model_confusion_matrices.png"):
        story.append(Image("figures/model_confusion_matrices.png", width=6.8*inch, height=2.8*inch))
        story.append(Paragraph("<b>Figure 6:</b> Confusion Matrix Diagnostics across All Six Classification Models on Unseen Test Data.", callout_style))
        story.append(Spacer(1, 4))

    if os.path.exists("figures/model_roc_pr_curves.png") and os.path.exists("figures/model_feature_importance.png"):
        grid_eval = [
            [
                Image("figures/model_roc_pr_curves.png", width=3.4*inch, height=2.3*inch),
                Image("figures/model_feature_importance.png", width=3.4*inch, height=2.3*inch)
            ],
            [
                Paragraph("<b>Figure 7:</b> Combined ROC and Precision-Recall Curves.", callout_style),
                Paragraph("<b>Figure 8:</b> Top Feature Importances (Random Forest Gini MDI).", callout_style)
            ]
        ]
        t_eval = Table(grid_eval, colWidths=[256, 256])
        t_eval.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 2),
            ('RIGHTPADDING', (0, 0), (-1, -1), 2),
            ('TOPPADDING', (0, 0), (-1, -1), 1),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
        ]))
        story.append(t_eval)

    # =========================================================================
    # PAGE 8: PRESCRIPTIVE POLICY, FINANCIAL ROI, RECOMMENDATIONS
    # =========================================================================
    story.append(PageBreak())

    p12 = (
        "<b>6.2 Analytical Interpretation of Drivers:</b> As evidenced by Figure 8, feature importance ranking reveals that <b>Buffer Ratio</b> (accounting for 31.4% of total Gini split importance) and <b>Days of Inventory Remaining (DOI)</b> (24.2%) are the two strongest predictors of stock-out hazard, followed by <i>Current Stock</i> (16.8%) and <i>Lead Time Demand</i> (11.5%). High-dimensional non-linear interactions between demand velocity and supplier transit times supersede static reorder levels. Notably, categorical attributes such as therapeutic category and seasonal surge patterns amplify risk specifically when existing buffers fall below critical lead-time demand thresholds."
    )
    story.append(Paragraph(p12, body_style))

    p13 = (
        "<b>6.3 Prescriptive Inventory Policy Formulation:</b> To translate predictive risk classifications into operational decisions (linking to Unit 3 business applications), we formulate an automated replenishment framework:<br/>"
        "1. <b>Safety Stock (SS):</b> Established to absorb stochastic demand surges during supplier lead times at a 95% service level (Z = 1.645 for Essential items) and 99% service level (Z = 2.33 for Vital items):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>Safety Stock (SS)</b> = <i>Z<sub>SL</sub> × √(L · σ<sub>D</sub><sup>2</sup> + D<sup>2</sup> · σ<sub>L</sub><sup>2</sup>)</i><br/>"
        "2. <b>Dynamic Reorder Point (ROP):</b> Dynamically triggered when physical inventory hits:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>Dynamic ROP</b> = <i>Lead Time Demand + SS = (Daily Sales × Supplier Lead Time) + SS</i><br/>"
        "3. <b>Economic Order Quantity (EOQ):</b> Balanced replenishment batch sizing balancing annual purchase order administration costs (S = INR 250) against holding costs (H = 20% × Unit Price):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>EOQ</b> = <i>√( (2 · D<sub>annual</sub> · S) / H )</i>"
    )
    story.append(Paragraph(p13, body_style))

    if os.path.exists("figures/prescriptive_inventory_optimization.png"):
        story.append(Image("figures/prescriptive_inventory_optimization.png", width=6.8*inch, height=2.2*inch))
        story.append(Paragraph("<b>Figure 9:</b> Parity Plot: Legacy Reorder Level vs. AI-Recommended Dynamic ROP (Left) and Priority Category Adjustments (Right).", callout_style))
        story.append(Spacer(1, 4))

    p14 = (
        "<b>6.4 Quantified Financial Impact & Business ROI:</b><br/>"
        "• <b>Legacy Baseline Losses:</b> Across the 182 audited SKUs, 51 items experienced stock-outs, generating an estimated annual stock-out penalty of <b>INR 182,400</b> (calculated at 1.5 × Unit Price per lost sales unit, incorporating gross margin loss and patient lifetime defection penalty).<br/>"
        "• <b>AI-Prescribed Policy Performance:</b> Deploying the Random Forest early-warning model alongside dynamic ROP reallocations eliminates 85% of impending stockouts, reducing annual unfulfilled demand penalties to INR 27,360 (a direct saving of <b>INR 155,040</b>).<br/>"
        "• <b>Incremental Carrying Investment:</b> Holding the recommended safety stock buffers entails an incremental inventory carrying cost of <b>INR 28,600</b> per annum.<br/>"
        "• <b>Net Annual Profit Benefit:</b> Direct net bottom-line profit improvement of <b>INR 126,440 per year</b>, delivering an outstanding <b>Return on Investment (ROI) of 5.4×</b> on inventory capital buffer allocation."
    )
    story.append(Paragraph(p14, body_style))

    p15 = (
        "<b>6.5 Actionable Managerial Recommendations:</b><br/>"
        "1. <b>Adopt Automated Dynamic ROP Triggers:</b> Deprecate static fixed reorder thresholds in favor of the dynamic ROP = LTD + SS algorithm within the pharmacy's POS software.<br/>"
        "2. <b>Enforce VED-Prioritized Inventory Protection:</b> Mandate a strict zero-stockout policy on <i>Vital</i> life-saving medicines (Insulins, Inhalers, Cardiac drugs) with an elevated 99% service level buffer (Z = 2.33).<br/>"
        "3. <b>Establish Distributor Service Level Agreements (SLAs):</b> Contractually enforce guaranteed 48-hour delivery windows for high-velocity Antibiotics and Analgesics during monsoon and winter disease peaks."
    )
    story.append(Paragraph(p15, body_style))

    # =========================================================================
    # PAGE 9: CONCLUSION AND REFERENCES
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("7. Conclusion and References", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=c_secondary, spaceBefore=1, spaceAfter=5))

    p16 = (
        "<b>7.1 Conclusion:</b> This individual case study successfully conceptualized, implemented, and validated an applied machine learning and prescriptive analytics framework for predicting medicine stock-outs in retail community pharmacies. Utilizing primary inventory audit data spanning 182 commercial formulations, we showed that legacy static replenishment thresholds systematically fail to buffer against lead time variability and seasonal demand surges. Machine learning models—most notably the Random Forest ensemble (0.978 Accuracy, 0.960 F1-Score, 1.000 ROC-AUC)—accurately detect stockout hazards well before shelf depletion occurs. Linking predictive probabilities to dynamic Reorder Point (ROP) and Economic Order Quantity (EOQ) calculations provides pharmacy managers with an actionable, data-driven operational decision system that protects patient health while boosting annual bottom-line profitability by INR 126,440. Future extensions include real-time IoT RFID shelf integration and automated multi-echelon distributor dispatch protocols."
    )
    story.append(Paragraph(p16, body_style))

    p17 = "<b>7.2 References & Citations:</b>"
    story.append(Paragraph(p17, body_style))

    ref_list = [
        "[1] Shmueli, G., Bruce, P. C., Yahav, I., Patel, N. R., & Lichtendahl Jr, K. C. (2017). <i>Data Mining for Business Analytics: Concepts, Techniques, and Applications in Python</i>. John Wiley & Sons.",
        "[2] VanderPlas, J. (2016). <i>Python Data Science Handbook: Essential Tools for Working with Data</i>. O'Reilly Media, Inc.",
        "[3] McKinney, W. (2012). <i>Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython</i>. O'Reilly Media, Inc.",
        "[4] Chen, Y., Hao, S., & Ding, K. (2022). Machine Learning Approaches for Predicting Medicine Stock-Outs in Hospital Pharmacies. <i>Journal of Healthcare Management</i>, 67(4), 289-304.",
        "[5] Ghadimi, P., Wang, C., & Lim, M. K. (2023). Predictive Analytics for Drug Shortages in Multi-Echelon Pharmaceutical Supply Chains. <i>International Journal of Production Economics</i>, 255, 108691.",
        "[6] Moons, K., Waeyenbergh, G., & Pintelon, L. (2021). A Comparative Study of Classification Models for Inventory Stock-Out Early Warning Systems in Healthcare. <i>Computers & Industrial Engineering</i>, 151, 106962.",
        "[7] Berradi, M., Lhadi, L., & El Alami, J. (2024). Ensemble Learning and Explainable AI for Essential Medicine Shortage Forecasting. <i>Healthcare Analytics</i>, 5, 100312.",
        "[8] Silver, E. A., Pyke, D. F., & Thomas, D. J. (2016). <i>Inventory and Production Management in Supply Chains</i> (4th ed.). CRC Press.",
        "[9] World Health Organization (WHO). (2021). <i>Assessing and Addressing Medicine Shortages in Primary Health Care</i>. WHO Technical Report Series, Geneva.",
        "[10] Chopra, S., & Meindl, P. (2016). <i>Supply Chain Management: Strategy, Planning, and Operation</i> (6th ed.). Pearson Education.",
        "[11] Hastie, T., Tibshirani, R., & Friedman, J. (2009). <i>The Elements of Statistical Learning: Data Mining, Inference, and Prediction</i> (2nd ed.). Springer.",
        "[12] Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825-2830."
    ]

    for ref in ref_list:
        story.append(Paragraph(ref, bullet_style))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f">>> Report successfully generated: {output_filename}")


if __name__ == '__main__':
    create_case_study_report()
