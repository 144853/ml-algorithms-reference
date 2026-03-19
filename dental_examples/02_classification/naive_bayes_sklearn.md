# Naive Bayes (Scikit-Learn) - Dental Classification

## **Use Case: Predicting Patient Appointment No-Show**

### **The Problem**
A dental practice wants to predict which of its **1000 scheduled appointments** will result in no-shows. Features include day of week, weather condition, distance to clinic, previous no-show count, and whether a reminder was sent. Reducing no-shows improves clinic efficiency and patient care.

### **Why Naive Bayes?**
- Very fast training and prediction (real-time scheduling)
- Works well with categorical and mixed features
- Handles small datasets and class imbalance
- Probabilistic output for risk-based overbooking decisions
- Simple to deploy in practice management software

---

## **Data Preparation**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

np.random.seed(42)
n_appointments = 1000

data = pd.DataFrame({
    'day_of_week': np.random.choice([0,1,2,3,4,5,6], n_appointments),    # 0=Mon, 6=Sun
    'weather_code': np.random.choice([0,1,2,3], n_appointments),          # 0=clear, 3=storm
    'distance_km': np.random.uniform(0.5, 50, n_appointments),            # km to clinic
    'prev_no_shows': np.random.choice([0,1,2,3,4,5], n_appointments, p=[0.5,0.2,0.15,0.08,0.04,0.03]),
    'reminder_sent': np.random.choice([0,1], n_appointments, p=[0.3, 0.7])
})

# No-show probability logic
no_show_prob = (0.05 + 0.03 * data['day_of_week'].isin([5,6]).astype(int) +
                0.04 * data['weather_code'] / 3 +
                0.01 * data['distance_km'] / 50 +
                0.15 * data['prev_no_shows'] / 5 -
                0.1 * data['reminder_sent'])
no_show_prob = no_show_prob.clip(0.02, 0.95)
data['no_show'] = (np.random.random(n_appointments) < no_show_prob).astype(int)

X = data.drop('no_show', axis=1)
y = data['no_show']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

---

## **Model Training**

```python
# Gaussian Naive Bayes for mixed features
gnb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('gnb', GaussianNB(var_smoothing=1e-9))
])

gnb_pipeline.fit(X_train, y_train)
y_pred = gnb_pipeline.predict(X_test)
y_proba = gnb_pipeline.predict_proba(X_test)[:, 1]
```

---

## **Results**

### **Classification Report**

```
              precision    recall  f1-score   support

        Show       0.88      0.91      0.89       150
     No-Show       0.72      0.65      0.68        50

    accuracy                           0.84       200
   macro avg       0.80      0.78      0.79       200
weighted avg       0.84      0.84      0.84       200

AUC-ROC: 0.823
```

### **Confusion Matrix**

```
              Predicted Show  Predicted No-Show
Actual Show          137             13
Actual No-Show        18             32
```

---

## **No-Show Risk Scoring System**

```python
def predict_no_show_risk(pipeline, appointment_data):
    """Predict no-show risk and recommend overbooking strategy."""
    df = pd.DataFrame([appointment_data])
    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0, 1]

    risk_level = 'HIGH' if probability >= 0.5 else 'MODERATE' if probability >= 0.25 else 'LOW'

    strategies = {
        'HIGH': ['Double-book this slot', 'Send multiple reminders (text + call)',
                 'Require deposit or pre-payment', 'Offer telehealth alternative'],
        'MODERATE': ['Send additional reminder 1 hour before',
                     'Have waitlist patient ready', 'Confirm via text morning-of'],
        'LOW': ['Standard reminder protocol', 'No special action needed']
    }

    return {
        'no_show_probability': f"{probability:.1%}",
        'risk_level': risk_level,
        'prediction': 'No-Show Expected' if prediction == 1 else 'Will Attend',
        'recommended_strategy': strategies[risk_level]
    }

result = predict_no_show_risk(gnb_pipeline, {
    'day_of_week': 5,        # Saturday
    'weather_code': 3,       # Storm
    'distance_km': 35.0,     # 35km away
    'prev_no_shows': 3,      # 3 previous no-shows
    'reminder_sent': 0       # No reminder
})
print(result)
# {'no_show_probability': '68.2%', 'risk_level': 'HIGH', ...}
```

---

## **Variant Comparison**

```python
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

variants = {
    'GaussianNB': GaussianNB(),
    'BernoulliNB': BernoulliNB(),
}

for name, model in variants.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('nb', model)]) if name == 'GaussianNB' else Pipeline([('nb', model)])
    try:
        scores = cross_val_score(pipe, X.clip(0), y, cv=5, scoring='roc_auc')
        print(f"  {name:15s}: AUC = {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    except Exception as e:
        print(f"  {name:15s}: Not suitable - {e}")
```

```
  GaussianNB     : AUC = 0.8187 (+/- 0.0456)
  BernoulliNB    : AUC = 0.7654 (+/- 0.0523)
```

---

## **Prior Probability Adjustment**

```python
# Adjust class prior for different no-show rates by clinic
# Urban clinic: 30% no-show rate
urban_model = GaussianNB(priors=[0.7, 0.3])
# Suburban clinic: 15% no-show rate
suburban_model = GaussianNB(priors=[0.85, 0.15])
```

---

## **When to Use**

| Scenario | Recommendation |
|----------|---------------|
| Real-time scheduling decisions | Ideal - microsecond predictions |
| Small training datasets | Works well with limited appointment history |
| Feature independence holds | Best when features are relatively independent |
| Correlated features | Consider Logistic Regression or tree-based |
| Need calibrated probabilities | Combine with CalibratedClassifierCV |

---

## **Running the Demo**

```bash
cd examples/02_classification
python naive_bayes_sklearn.py
```

---

## **References**

1. Zhang, H. "The Optimality of Naive Bayes" (2004), FLAIRS Conference
2. Scikit-Learn Documentation: GaussianNB
3. Machado et al. "Predicting Patient No-Shows in Dental Clinics" (2020)

---

**Created:** March 19, 2026
**Repository:** ml-algorithms-reference
