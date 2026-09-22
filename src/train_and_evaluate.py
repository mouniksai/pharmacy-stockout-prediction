"""
Pharmacy Medicine Stock-Out Prediction & Inventory Optimization Pipeline
Course: 23CSE452 Business Analytics
Student: Mounik Sai (CB.SC.U4CSE23561)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc
)

# Configuration & Styling
np.random.seed(42)
os.makedirs('figures', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def generate_eda_visualizations(df):
    print(">>> Generating Exploratory Data Analysis (EDA) figures...")
    
    # 1. Distribution of Key Operational Metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    sns.histplot(df['Current_Stock'], kde=True, ax=axes[0, 0], color='#1f77b4', bins=20)
    axes[0, 0].set_title('Distribution of Current Stock (Units)', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Current Stock (Units)')
    axes[0, 0].set_ylabel('Frequency (SKUs)')
    axes[0, 0].axvline(df['Current_Stock'].median(), color='red', linestyle='--', label=f'Median: {df["Current_Stock"].median():.0f}')
    axes[0, 0].legend()
    
    sns.histplot(df['Daily_Sales'], kde=True, ax=axes[0, 1], color='#2ca02c', bins=20)
    axes[0, 1].set_title('Distribution of Daily Sales Velocity', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Daily Sales (Units/Day)')
    axes[0, 1].set_ylabel('Frequency (SKUs)')
    axes[0, 1].axvline(df['Daily_Sales'].mean(), color='darkgreen', linestyle='--', label=f'Mean: {df["Daily_Sales"].mean():.1f}')
    axes[0, 1].legend()
    
    sns.countplot(data=df, x='Supplier_Lead_Time', ax=axes[1, 0], palette='Blues_r')
    axes[1, 0].set_title('Distributor Replenishment Lead Time', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Supplier Lead Time (Days)')
    axes[1, 0].set_ylabel('Number of SKUs')
    
    sns.boxplot(data=df, x='Stock_Status', y='Days_of_Inventory', ax=axes[1, 1], palette=['#2b83ba', '#d7191c'])
    axes[1, 1].set_title('Days of Inventory (DOI) by Stock Status', fontsize=13, fontweight='bold')
    axes[1, 1].set_xticklabels(['In Stock (0)', 'Stock-Out / At Risk (1)'])
    axes[1, 1].set_xlabel('Target Status')
    axes[1, 1].set_ylabel('Days of Inventory Remaining')
    axes[1, 1].set_ylim(0, 45)
    
    plt.tight_layout()
    plt.savefig('figures/eda_distribution_overview.png', dpi=300)
    plt.close()
    
    # 2. Stockout Risk by Category & VED Criticality
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    cat_risk = df.groupby('Category')['Stock_Status'].agg(
        Total='count',
        Stockouts='sum'
    ).reset_index()
    cat_risk['Stockout_Rate'] = (cat_risk['Stockouts'] / cat_risk['Total']) * 100
    cat_risk = cat_risk.sort_values('Stockout_Rate', ascending=False)
    
    sns.barplot(data=cat_risk, x='Stockout_Rate', y='Category', ax=axes[0], palette='Reds_r')
    axes[0].set_title('Stock-Out Incidence Rate by Therapeutic Category (%)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Stock-Out Rate (%)')
    axes[0].set_ylabel('')
    for i, v in enumerate(cat_risk['Stockout_Rate']):
        axes[0].text(v + 0.8, i, f"{v:.1f}% ({cat_risk['Stockouts'].iloc[i]}/{cat_risk['Total'].iloc[i]})", va='center', fontsize=9)
    axes[0].set_xlim(0, 50)
    
    ved_order = ['Vital', 'Essential', 'Desirable']
    sns.countplot(data=df, x='Criticality', hue='Stock_Status', order=ved_order, ax=axes[1], palette=['#4575b4', '#d73027'])
    axes[1].set_title('Stock Status Distribution across VED Criticality Classes', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('VED Clinical Priority')
    axes[1].set_ylabel('Number of SKUs')
    axes[1].legend(['Adequately Stocked', 'Stock-Out / Critical Risk'], title='Status')
    
    plt.tight_layout()
    plt.savefig('figures/eda_category_and_criticality.png', dpi=300)
    plt.close()

    # 3. Correlation Matrix Heatmap of Inventory Drivers
    plt.figure(figsize=(9, 7))
    num_cols = [
        'Current_Stock', 'Daily_Sales', 'Supplier_Lead_Time', 'Reorder_Level',
        'Unit_Price_INR', 'Expiry_Months_Remaining', 'Days_of_Inventory',
        'Lead_Time_Demand', 'Buffer_Ratio', 'Stock_Status'
    ]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix of Operational Inventory Variables', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('figures/eda_correlation_matrix.png', dpi=300)
    plt.close()

    # 4. Lead Time Demand vs Current Stock with Decision Boundary
    plt.figure(figsize=(9, 6))
    scatter = sns.scatterplot(
        data=df, x='Lead_Time_Demand', y='Current_Stock', hue='Stock_Status',
        style='Criticality', size='Daily_Sales', sizes=(40, 220),
        palette=['#2b83ba', '#d7191c'], alpha=0.85
    )
    max_val = max(df['Lead_Time_Demand'].max(), 160)
    plt.plot([0, max_val], [0, max_val], 'k--', label='Critical Threshold: Stock = LTD (Buffer=0)')
    plt.plot([0, max_val], [0, max_val * 1.5], 'g:', label='Safety Threshold: Stock = 1.5 x LTD')
    plt.fill_between([0, max_val], [0, max_val], color='#fee08b', alpha=0.25, label='Stock-Out Vulnerability Zone')
    
    plt.title('Current Physical Stock vs. Lead-Time Demand (LTD)', fontsize=13, fontweight='bold')
    plt.xlabel('Replenishment Lead-Time Demand (Units = Daily Sales x Lead Time)')
    plt.ylabel('Current Shelf Stock (Units)')
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig('figures/eda_leadtime_demand_frontier.png', dpi=300)
    plt.close()

    # 5. Seasonal Demand Impact
    plt.figure(figsize=(9, 5))
    season_summary = df.groupby('Seasonal_Demand')['Stock_Status'].mean().reset_index()
    season_summary['Stockout_Rate'] = season_summary['Stock_Status'] * 100
    sns.barplot(data=season_summary, x='Seasonal_Demand', y='Stockout_Rate', palette='Spectral')
    plt.title('Stock-Out Incidence Rate by Seasonal Disease Surge Profile', fontsize=12, fontweight='bold')
    plt.xlabel('Epidemiological Surge Pattern')
    plt.ylabel('Observed Stock-Out Rate (%)')
    for i, v in enumerate(season_summary['Stockout_Rate']):
        plt.text(i, v + 0.8, f"{v:.1f}%", ha='center', fontweight='bold')
    plt.ylim(0, 45)
    plt.tight_layout()
    plt.savefig('figures/eda_seasonal_demand_impact.png', dpi=300)
    plt.close()
    
    print(">>> EDA figures saved successfully.")


def run_pca_dimension_reduction(X_train_df, y_train):
    print(">>> Running Principal Component Analysis (PCA)...")
    numeric_cols = [
        'Current_Stock', 'Daily_Sales', 'Supplier_Lead_Time', 'Reorder_Level',
        'Unit_Price_INR', 'Minimum_Order_Quantity', 'Expiry_Months_Remaining',
        'Days_of_Inventory', 'Lead_Time_Demand', 'Buffer_Ratio'
    ]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_df[numeric_cols])
    
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    explained_var = pca.explained_variance_ratio_ * 100
    cum_explained_var = np.cumsum(explained_var)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].bar(range(1, len(explained_var) + 1), explained_var, color='#3182bd', alpha=0.8, label='Individual Variance')
    axes[0].plot(range(1, len(explained_var) + 1), cum_explained_var, 'ro-', linewidth=2, label='Cumulative Variance')
    axes[0].axhline(80, color='grey', linestyle='--', label='80% Explained Threshold')
    axes[0].set_title('PCA Scree Plot & Cumulative Explained Variance', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Principal Component Index')
    axes[0].set_ylabel('Percentage of Variance Explained (%)')
    axes[0].set_xticks(range(1, len(explained_var) + 1))
    axes[0].legend()
    
    scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='coolwarm', alpha=0.85, edgecolor='k', s=50)
    axes[1].set_title('2D PCA Projection: Inventory Latent Separation', fontsize=12, fontweight='bold')
    axes[1].set_xlabel(f'PC1 ({explained_var[0]:.1f}% Variance) - Inventory Buffer Depth')
    axes[1].set_ylabel(f'PC2 ({explained_var[1]:.1f}% Variance) - Demand & Replenishment Scale')
    cbar = plt.colorbar(scatter, ax=axes[1], ticks=[0, 1])
    cbar.ax.set_yticklabels(['In Stock (0)', 'Stock-Out (1)'])
    
    plt.tight_layout()
    plt.savefig('figures/pca_scree_and_projection.png', dpi=300)
    plt.close()
    
    loadings = pd.DataFrame(
        pca.components_[:3, :].T,
        columns=['PC1', 'PC2', 'PC3'],
        index=numeric_cols
    )
    loadings.to_csv('figures/pca_loadings.csv')
    print(">>> PCA completed. Top 3 components explain {:.1f}% total variance.".format(cum_explained_var[2]))
    return pca, scaler, loadings


def train_and_evaluate_models(df):
    print(">>> Splitting dataset into 75% train and 25% test (stratified)...")
    
    feature_cols = [
        'Category', 'Current_Stock', 'Daily_Sales', 'Supplier_Lead_Time',
        'Reorder_Level', 'Seasonal_Demand', 'Unit_Price_INR',
        'Minimum_Order_Quantity', 'Criticality', 'Storage_Condition',
        'Expiry_Months_Remaining', 'Days_of_Inventory', 'Lead_Time_Demand',
        'Buffer_Ratio', 'Stock_to_Reorder_Ratio'
    ]
    target_col = 'Stock_Status'
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    cat_features = ['Category', 'Seasonal_Demand', 'Criticality', 'Storage_Condition']
    num_features = [c for c in feature_cols if c not in cat_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
        ]
    )
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        'Decision Tree (CART)': DecisionTreeClassifier(max_depth=4, min_samples_split=6, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=42),
        'k-Nearest Neighbors (k-NN)': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'Gaussian Naive Bayes': GaussianNB(),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, max_depth=3, random_state=42)
    }
    
    results = []
    trained_pipelines = {}
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n" + "="*95)
    print(f"{'Model Name':<28} | {'CV ROC-AUC':<10} | {'Train Acc':<10} | {'Test Acc':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<8} | {'Test AUC':<8}")
    print("="*95)
    
    for name, clf in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='roc_auc')
        mean_cv_auc = cv_scores.mean()
        
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe
        
        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)
        
        y_test_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else pipe.decision_function(X_test)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        prec = precision_score(y_test, y_test_pred)
        rec = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test, y_test_proba)
        
        results.append({
            'Model': name,
            'CV_ROC_AUC': round(mean_cv_auc, 4),
            'Train_Accuracy': round(train_acc, 4),
            'Test_Accuracy': round(test_acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1_Score': round(f1, 4),
            'Test_ROC_AUC': round(roc_auc, 4)
        })
        
        print(f"{name:<28} | {mean_cv_auc:.4f}     | {train_acc:.4f}     | {test_acc:.4f}    | {prec:.4f}     | {rec:.4f}     | {f1:.4f}   | {roc_auc:.4f}")
    
    print("="*95 + "\n")
    
    df_results = pd.DataFrame(results)
    df_results.to_csv('figures/model_performance_summary.csv', index=False)
    
    # 6. Confusion Matrices Grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    
    for idx, (name, pipe) in enumerate(trained_pipelines.items()):
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=['In Stock', 'Stock-Out'],
                    yticklabels=['In Stock', 'Stock-Out'])
        axes[idx].set_title(f"{name}\nAcc: {accuracy_score(y_test, y_pred)*100:.1f}% | F1: {f1_score(y_test, y_pred):.3f}",
                            fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_ylabel('Actual Label')
        
    plt.tight_layout()
    plt.savefig('figures/model_confusion_matrices.png', dpi=300)
    plt.close()
    
    # 7. ROC Curves and Precision-Recall Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for idx, (name, pipe) in enumerate(trained_pipelines.items()):
        clf = pipe.named_steps['classifier']
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else pipe.decision_function(X_test)
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_val = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {roc_val:.3f})", color=colors[idx], linewidth=2)
        
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_val = auc(rec, prec)
        axes[1].plot(rec, prec, label=f"{name} (PR-AUC = {pr_val:.3f})", color=colors[idx], linewidth=2)
        
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('False Positive Rate (1 - Specificity)')
    axes[0].set_ylabel('True Positive Rate (Sensitivity / Recall)')
    axes[0].legend(loc='lower right')
    
    axes[1].set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Recall (Sensitivity)')
    axes[1].set_ylabel('Precision (Positive Predictive Value)')
    axes[1].legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig('figures/model_roc_pr_curves.png', dpi=300)
    plt.close()
    
    # 8. Feature Importance Analysis
    rf_model = trained_pipelines['Random Forest'].named_steps['classifier']
    preproc = trained_pipelines['Random Forest'].named_steps['preprocessor']
    
    encoded_cat_names = preproc.named_transformers_['cat'].get_feature_names_out(cat_features)
    all_feature_names = num_features + list(encoded_cat_names)
    
    rf_importances = rf_model.feature_importances_
    df_feat_imp = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': rf_importances
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_feat_imp.head(10), x='Importance', y='Feature', palette='mako')
    plt.title('Top 10 Feature Importances (Random Forest Ensemble Gini Impurity)', fontsize=12, fontweight='bold')
    plt.xlabel('Relative Feature Importance (MDI)')
    plt.ylabel('')
    for i, v in enumerate(df_feat_imp.head(10)['Importance']):
        plt.text(v + 0.005, i, f"{v*100:.1f}%", va='center', fontsize=9, fontweight='bold')
    plt.xlim(0, max(df_feat_imp['Importance']) * 1.15)
    plt.tight_layout()
    plt.savefig('figures/model_feature_importance.png', dpi=300)
    plt.close()
    
    # 9. Prescriptive Inventory Policy Optimization
    print(">>> Computing Prescriptive Inventory Optimization Policy (ROP, SS, EOQ)...")
    Z_95 = 1.645
    ordering_cost_S = 250.0  # INR per order
    holding_cost_rate_H = 0.20 # 20% annual carrying cost
    
    df_prescriptive = df.copy()
    sigma_d = 0.30 * df_prescriptive['Daily_Sales']
    df_prescriptive['Safety_Stock_Recommended'] = np.ceil(Z_95 * np.sqrt(df_prescriptive['Supplier_Lead_Time']) * sigma_d).astype(int)
    df_prescriptive['Reorder_Point_Recommended'] = np.ceil(df_prescriptive['Lead_Time_Demand'] + df_prescriptive['Safety_Stock_Recommended']).astype(int)
    
    annual_demand = df_prescriptive['Daily_Sales'] * 365
    annual_holding_cost_per_unit = np.maximum(df_prescriptive['Unit_Price_INR'] * holding_cost_rate_H, 2.0)
    df_prescriptive['EOQ_Recommended'] = np.ceil(np.sqrt((2 * annual_demand * ordering_cost_S) / annual_holding_cost_per_unit)).astype(int)
    df_prescriptive['EOQ_Recommended'] = np.maximum(df_prescriptive['EOQ_Recommended'], df_prescriptive['Minimum_Order_Quantity'])
    
    df_prescriptive['Current_ROP_Gap'] = df_prescriptive['Reorder_Level'] - df_prescriptive['Reorder_Point_Recommended']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(df_prescriptive['Reorder_Level'], df_prescriptive['Reorder_Point_Recommended'],
                    c=df_prescriptive['Stock_Status'], cmap='coolwarm', alpha=0.8, edgecolor='k', s=60)
    axes[0].plot([0, 160], [0, 160], 'k--', label='Parity Line (Legacy = Recommended)')
    axes[0].set_title('Legacy Reorder Level vs. AI-Recommended Dynamic ROP', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Legacy Pharmacy Reorder Level (Units)')
    axes[0].set_ylabel('AI-Prescribed Reorder Point (Units)')
    axes[0].legend()
    
    cat_fin = df_prescriptive.groupby('Category').apply(lambda g: pd.Series({
        'Stockout_Risk_SKUs': (g['Stock_Status'] == 1).sum(),
        'Avg_Lead_Time_Days': g['Supplier_Lead_Time'].mean(),
        'Recommended_Buffer_Units': g['Safety_Stock_Recommended'].mean()
    }), include_groups=False).reset_index()
    
    sns.barplot(data=cat_fin, x='Stockout_Risk_SKUs', y='Category', ax=axes[1], palette='flare')
    axes[1].set_title('High-Risk SKUs per Category Requiring Immediate ROP Adjustment', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Number of Vulnerable SKUs')
    axes[1].set_ylabel('')
    for i, v in enumerate(cat_fin['Stockout_Risk_SKUs']):
        axes[1].text(v + 0.1, i, f"{int(v)} SKUs", va='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('figures/prescriptive_inventory_optimization.png', dpi=300)
    plt.close()
    
    df_prescriptive.to_csv('data/pharmacy_prescriptive_policy.csv', index=False)
    print(">>> Prescriptive policy computed and exported to data/pharmacy_prescriptive_policy.csv.")
    
    return trained_pipelines, df_results, df_prescriptive, X_train, X_test, y_train, y_test

if __name__ == '__main__':
    df = pd.read_csv('data/pharmacy_stockout_cleaned.csv')
    generate_eda_visualizations(df)
    
    X_train_df, _, y_train_df, _ = train_test_split(df, df['Stock_Status'], test_size=0.25, random_state=42, stratify=df['Stock_Status'])
    run_pca_dimension_reduction(X_train_df, y_train_df)
    
    train_and_evaluate_models(df)
    print(">>> Entire analytics workflow finished successfully!")
