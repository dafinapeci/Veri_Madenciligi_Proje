#main.py code file
# this code file is about training and saving the models

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
)
from xgboost import XGBClassifier


# 1. load raw data


df_2014 = pd.read_csv("Datasets/Mental Health in Tech Survey (Responses) - Form Responses 1.csv")
df_2016 = pd.read_csv("Datasets/mental-heath-in-tech-2016_20161114.csv")

# 2. Rename 2014 Columns — Long Names to Short Names

df_2014 = df_2014.rename(columns={
    "If you live in the United States, which state or territory do you live in?":   "state",
    "Are you self-employed?":                                                        "self_employed",
    "Do you have a family history of mental illness?":                               "family_history",
    "Have you sought treatment for a mental health condition?":                      "treatment",
    "If you have a mental health condition, do you feel that it interferes with your work?": "work_interfere",
    "How many employees does your company or organization have?":                    "no_employees",
    "Do you work remotely (outside of an office) at least 50% of the time?":        "remote_work",
    "Is your employer primarily a tech company/organization?":                       "tech_company",
    "Does your employer provide mental health benefits?":                            "benefits",
    "Do you know the options for mental health care your employer provides?":        "care_options",
    "Has your employer ever discussed mental health as part of an employee wellness program?": "wellness_program",
    "Does your employer provide resources to learn more about mental health issues and how to seek help?": "seek_help",
    "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?": "anonymity",
    "How easy is it for you to take medical leave for a mental health condition?":   "leave",
    "Do you think that discussing a mental health issue with your employer would have negative consequences?": "mental_health_consequence",
    "Do you think that discussing a physical health issue with your employer would have negative consequences?": "phys_health_consequence",
    "Would you be willing to discuss a mental health issue with your coworkers?":    "coworkers",
    "Would you be willing to discuss a mental health issue with your direct supervisor(s)?": "supervisor",
    "Would you bring up a mental health issue with a potential employer in an interview?":    "mental_health_interview",
    "Would you bring up a physical health issue with a potential employer in an interview?":  "phys_health_interview",
    "Do you feel that your employer takes mental health as seriously as physical health?":    "mental_vs_physical",
    "Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?": "obs_consequence",
    "Any additional notes or comments":                                              "comments",
})


# 3. Rename 2016 Columns — Long Names to Short Names

df_2016 = df_2016.rename(columns={
    "What is your age?":                                                             "Age",
    "What is your gender?":                                                          "Gender",
    "What country do you live in?":                                                  "Country",
    "What US state or territory do you live in?":                                    "state",
    "Are you self-employed?":                                                        "self_employed",
    "Do you have a family history of mental illness?":                               "family_history",
    "Have you ever sought treatment for a mental health issue from a mental health professional?": "treatment",
    "If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?": "work_interfere",
    "How many employees does your company or organization have?":                    "no_employees",
    "Do you work remotely?":                                                         "remote_work",
    "Is your employer primarily a tech company/organization?":                       "tech_company",
    "Does your employer provide mental health benefits as part of healthcare coverage?": "benefits",
    "Do you know the options for mental health care available under your employer-provided coverage?": "care_options",
    "Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?": "wellness_program",
    "Does your employer offer resources to learn more about mental health concerns and options for seeking help?": "seek_help",
    "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?": "anonymity",
    "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:": "leave",
    "Do you think that discussing a mental health disorder with your employer would have negative consequences?": "mental_health_consequence",
    "Do you think that discussing a physical health issue with your employer would have negative consequences?": "phys_health_consequence",
    "Would you feel comfortable discussing a mental health disorder with your coworkers?":          "coworkers",
    "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?": "supervisor",
    "Would you bring up a mental health issue with a potential employer in an interview?":           "mental_health_interview",
    "Would you be willing to bring up a physical health issue with a potential employer in an interview?": "phys_health_interview",
    "Do you feel that your employer takes mental health as seriously as physical health?":           "mental_vs_physical",
    "Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?": "obs_consequence",
})


# 4. Keep Shared Columns And Merge

shared_columns = [col for col in df_2014.columns if col in df_2016.columns]

df_2014 = df_2014[shared_columns]
df_2016 = df_2016[shared_columns]

df_2014["survey_year"] = 2014
df_2016["survey_year"] = 2016

df = pd.concat([df_2014, df_2016], ignore_index=True)

# verify treatment exists before going further
assert "treatment" in df.columns, "treatment column missing — check your renames above"
print("tedavi sütununda mevcut olduğu doğrulandı")

# 5. clean target variable — treatment
# clean it once here. never touch it again.

df["treatment"] = df["treatment"].astype(str).str.strip().str.capitalize()
df["treatment"] = df["treatment"].map({"Yes": 1, "No": 0})
df = df.dropna(subset=["treatment"])
df["treatment"] = df["treatment"].astype(int)

print("\nTedavi değeri sayısı:")
print(df["treatment"].value_counts())


# 6. clean age

df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df = df[df["Age"].between(15, 75)]


# 7. clean gender

def clean_gender(value):
    if pd.isna(value):
        return "Other"
    v = str(value).strip().lower()
    male_terms   = ["male","m","man","men","cis male","cis man","mail","mal",
                    "make","msle","maile","malr","guy","dude","boy","gentleman"]
    female_terms = ["female","f","woman","women","cis female","cis woman",
                    "femail","femake","girl","lady","femaile"]
    if v in male_terms:
        return "Male"
    elif v in female_terms:
        return "Female"
    else:
        return "Other"

df["Gender"] = df["Gender"].apply(clean_gender)
print("\nCinsiyet sayısı:", df["Gender"].value_counts().to_dict())


# 8. drop useless columns and handle missing values

df = df.drop(columns=[c for c in ["Timestamp","comments","state"] if c in df.columns])

if "work_interfere" in df.columns:
    df["work_interfere"] = df["work_interfere"].fillna("Unknown")
if "self_employed" in df.columns:
    df["self_employed"]  = df["self_employed"].fillna(0)

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nKalan eksik değerler:", df.isnull().sum().sum())
print("Son veri kümesinin yapısı:", df.shape)


# 9. Encode and Split


gender_series = df["Gender"].copy()   # Save before encoding for fairness audit

X = df.drop(columns=["treatment"])
y = df["treatment"]
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

gender_test   = gender_series.loc[y_test.index].reset_index(drop=True)
y_test_reset  = y_test.reset_index(drop=True)
X_test_reset  = X_test.reset_index(drop=True)

print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")


# 10. Train All Three Models
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 10. Define the parameter grids for each model

# create a dictionary of grids to loop through them
tuning_configs = [
    {
        "name": "Logistic Regression",
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "C": [0.1, 1, 10],
            "solver": ["liblinear", "lbfgs"]
        }
    },
    {
        "name": "Random Forest",
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }
    },
    {
        "name": "XGBoost",
        "model": XGBClassifier(eval_metric="logloss", random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        }
    }
]

# 11. tune all models and find the best one
best_overall_score = 0
best_overall_model = None
best_model_name = ""
tuned_models = {}

print("Tüm modeller için hiperparametre ayarlaması başlatılıyor...\n")

for config in tuning_configs:
    print(f"Tuning {config['name']}...")
    grid_search = GridSearchCV(
        estimator=config["model"],
        param_grid=config["params"],
        cv=5,
        scoring="f1",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # get the best version of this specific model
    best_model = grid_search.best_estimator_
    preds = best_model.predict(X_test)
    f1 = f1_score(y_test, preds)

    # store the tuned model
    tuned_models[config['name']] = best_model

    print(f"  Best Params: {grid_search.best_params_}")
    print(f"  F1 Score: {f1 * 100:.2f}% | Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")
    print("-" * 30)

    # track which model is the overall winner
    if f1 > best_overall_score:
        best_overall_score = f1
        best_overall_model = best_model
        best_model_name = config['name']

print(f"\nOverall Winner: {best_model_name} with F1 Score: {best_overall_score * 100:.2f}%")

# 12. Save Everything

# save the best model with its name as well
joblib.dump(best_overall_model, f"Models/best_model_{best_model_name.replace(' ', '_').lower()}.joblib")
# This is the universal filename the app can always point to, without needing to specify the model name
joblib.dump(best_overall_model, "Models/best_model.joblib")

# save the other tuned versions just in case
for name, model in tuned_models.items():
    filename = f"Models/tuned_{name.replace(' ', '_').lower()}.joblib"
    joblib.dump(model, filename)

# save metadata
joblib.dump(list(X_train.columns), "Models/feature_columns.joblib")
joblib.dump(X_test_reset, "Models/X_test.joblib")
joblib.dump(y_test_reset, "Models/y_test.joblib")
joblib.dump(gender_test, "Models/gender_test.joblib")

print("\nTüm ayarlanmış modeller kaydedildi.")
print(f"'{best_model_name}' “best_model.joblib” adıyla ana dosya olarak kaydedildi")