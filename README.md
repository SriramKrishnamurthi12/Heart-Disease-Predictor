# Multi-Dataset Stacking Ensemble for Heart Disease Prediction with Explainable AI

This repository contains the implementation and experimental workflow for the research paper:

"A Multi-Dataset Stacking Ensemble Framework for Heart Disease Prediction with Three-Tier Explainable AI and Probability Calibration"

## Authors
Sriram Krishnamurthi  
Niharika Prasanna Kumar  
Department of Information Science and Engineering  
RV Institute of Technology and Management, Bengaluru, India

---

## Overview

This project proposes a stacking ensemble framework for heart disease prediction, designed to address three key challenges in clinical machine learning:

- Generalization across multiple datasets  
- Model interpretability  
- Reliability of predicted probabilities  

The framework combines multiple machine learning models with a meta-learner and integrates explainability and calibration techniques to improve clinical relevance.

---

## Key Features

- Multi-dataset evaluation (Cleveland, Statlog, Heart Failure)
- Stacking ensemble using:
  - XGBoost  
  - LightGBM  
  - Random Forest  
  - Support Vector Machine  
- Logistic Regression as meta-learner  
- Three-tier explainability:
  - SHAP (global feature importance)  
  - LIME (local explanations)  
  - DiCE (counterfactual explanations)  
- Probability calibration:
  - Platt Scaling  
  - Isotonic Regression  
- Performance evaluation using ROC-AUC, Recall, F1-score, and Brier Score  

---

## Datasets

The following publicly available datasets are used:

- UCI Cleveland Heart Disease Dataset  
- Statlog Heart Dataset  
- UCI Heart Failure Clinical Records Dataset  

All datasets are preprocessed using a consistent pipeline including imputation, scaling, feature selection, and class balancing.

---

## Installation

Install required dependencies using:

```bash
pip install xgboost lightgbm imbalanced-learn shap lime dice-ml scikit-learn pandas numpy matplotlib seaborn scipy
