# 🏗️ Fiber-Reinforced Concrete Compressive Strength Estimator

A machine learning project that estimates the **compressive strength (MPa)** of fiber-reinforced concrete from its mix proportions, with a web app for instant predictions.

**🔗 Live app:** [https://reinforced-concrete-strength-estimator-2vcgnaw3em6dnrvnjqszdl.streamlit.app/](https://reinforced-concrete-strength-estimator-2vcgnaw3em6dnrvnjqszdl.streamlit.app/)

---

## Overview

Compressive strength is normally confirmed by casting and crushing specimens, which takes time and material. This project explores how well strength can be estimated directly from the mix design.

The notebook covers the full workflow: data exploration, preprocessing, feature selection, model comparison, honest validation, and deployment as a Streamlit app.

## Dataset

- **521 mixes** of fiber-reinforced concrete (`comp.csv`)
- **12 input features:** fiber content, fiber aspect ratio, fiber length, fine aggregate, coarse aggregate, water, additive (admixture), silica fume, slag, fly ash, metakaolin, cement
- **Target:** compressive strength (MPa)
- No missing values; 3 duplicate rows

## Approach

1. **Exploration:** correlation analysis, scatter plots, histograms, IQR outlier check
2. **Preprocessing:** Yeo-Johnson power transform (fit on training data only) for the linear models
3. **Feature selection:** `SelectKBest` with `f_regression` on the training set, keeping the top 6 features
4. **Models compared:**
   - Linear Regression (scikit-learn)
   - Linear Regression implemented from scratch (normal equation)
   - Linear Regression trained with gradient descent (NumPy)
   - Random Forest (baseline, tuned, and regularised versions)
5. **Validation:** random train/test split, 5-fold cross-validation, and **group-based validation** where mixes in the test set are never seen in training
6. **Deployment:** final Random Forest saved with `joblib` and served through Streamlit

### Selected features for the final model

`silicafumecontent`, `additivecontent`, `water`, `cementecontent`, `flyashcontent`, `fibercontent`

## Results

| Model | Test R² | MAE (MPa) |
|---|---|---|
| Linear Regression | 0.60 | 8.06 |
| Gradient Descent (from scratch) | 0.60 | 8.06 |
| Random Forest (random split) | 0.85 | 4.18 |

### Honest validation

Many rows in this dataset share the same base mix and differ only in fiber content. A random split therefore places near-identical mixes in both the training and test sets, which makes results look better than they would be on a genuinely new mix.

| Validation method | R² | MAE (MPa) |
|---|---|---|
| Random split | 0.85 | 4.2 |
| 5-fold group cross-validation (RF, 6 features) | ≈ 0.70 | – |
| Single unseen-mix split | 0.45 | 7.7 |

**Takeaway:** the model interpolates well between mixes similar to those it has seen, but is considerably less accurate on entirely new mixes. For new mixes, expect a typical error of roughly **±8 MPa**. The app displays this range with every prediction.

Other findings:
- Random Forest clearly outperformed linear models, indicating a non-linear relationship between mix proportions and strength.
- Adding a water-binder ratio feature and trying Gradient Boosting did not give a meaningful improvement under group cross-validation.
- Water and silica fume were the most influential features in the Random Forest.

## The app

The Streamlit app takes the six mix inputs (kg/m³), and returns:
- the estimated compressive strength
- the disagreement between the forest's trees
- a realistic ±8 MPa range
- a warning if any input falls **outside the training range** (Random Forests cannot extrapolate reliably)

## Run it locally

```bash
git clone https://github.com/naradtrupti/concrete-strength-estimator.git
cd concrete-strength-estimator
pip install -r requirements.txt
streamlit run app.py
```

## Repository structure

```
├── app.py                          # Streamlit app
├── requirements.txt                # Python dependencies
├── concrete_final_model.pkl        # Trained Random Forest
├── concrete_final_meta.pkl         # Feature names, input ranges, default values
├── concrete_strength_model.ipynb   # Full analysis notebook
└── comp.csv                        # Dataset (if license permits)
```

## Limitations

- Based on only 521 mixes, many of which share base mixes
- The data does not include curing age or detailed fiber properties beyond what is listed above
- Accuracy drops on mixes that differ substantially from the training data
- **This is an educational estimator. Always verify with laboratory testing. Not for structural design use.**

## Possible improvements

- Collect more independent mixes to improve generalisation
- Include curing age and test conditions
- Add uncertainty estimates based on quantile regression or conformal prediction
- Compare against gradient boosting libraries such as XGBoost and LightGBM on a larger dataset

## Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Streamlit

## Author

**Trupti Narad**
B.Tech Civil Engineering, Walchand College of Engineering, Sangli
GitHub: [naradtrupti](https://github.com/naradtrupti)
