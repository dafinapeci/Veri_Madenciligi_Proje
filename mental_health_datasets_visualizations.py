import os

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.ticker as mtick


# this file is created only for visualization purposes
# before we start doing any training and preprocessing this data
# we need to check what data are we working with, its details, distribution etc.
# and visuals are big helpers for us humans, since they make understanding much easier


# 1. Load Data
df_2014 = pd.read_csv("Datasets/Mental Health in Tech Survey (Responses) - Form Responses 1.csv")
df_2016 = pd.read_csv("Datasets/mental-heath-in-tech-2016_20161114.csv")

print("2014 Ham Veri Kümesinin Şekli: ", df_2014.shape)
print("2016 Ham Veri Kümesinin Şekli: ", df_2016.shape)

# we will save the missing values snapshot before cleaning
missing_before = pd.concat([df_2014.isnull().sum(), df_2016.isnull().sum()], axis=1)
missing_before.columns = ["2014 Missing", "2016 Missing"]

print("2014 Sütünları: ", df_2014.columns)
print("2016 Sütünları: ", df_2016.columns)

# 2. Rename the columns to match

df_2014 = df_2014.rename(columns={
    "Age": "Age",
    "Gender": "Gender",
    "Country": "Country",
    "If you live in the United States, which state or territory do you live in?": "state",
    "Are you self-employed?": "self_employed",
    "Do you have a family history of mental illness?": "family_history",
    "Have you sought treatment for a mental health condition?": "treatment",
    "If you have a mental health condition, do you feel that it interferes with your work?": "work_interfere",
    "How many employees does your company or organization have?": "no_employees",
    "Do you work remotely (outside of an office) at least 50% of the time?": "remote_work",
    "Is your employer primarily a tech company/organization?": "tech_company",
    "Does your employer provide mental health benefits?": "benefits",
    "Do you know the options for mental health care your employer provides?": "care_options",
    "Has your employer ever discussed mental health as part of an employee wellness program?": "wellness_program",
    "Does your employer provide resources to learn more about mental health issues and how to seek help?": "seek_help",
    "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?": "anonymity",
    "How easy is it for you to take medical leave for a mental health condition?": "leave",
    "Do you think that discussing a mental health issue with your employer would have negative consequences?": "mental_health_consequence",
    "Would you be willing to discuss a mental health issue with your coworkers?": "coworkers",
    "Would you be willing to discuss a mental health issue with your direct supervisor(s)?": "supervisor",
    "Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?": "obs_consequence",
    "Timestamp": "Timestamp"
})

df_2016 = df_2016.rename(columns={
    "What is your age?": "Age",
    "What is your gender?": "Gender",
    "What country do you live in?": "Country",
    "What US state or territory do you live in?": "state",
    "Are you self-employed?": "self_employed",
    "Do you have a family history of mental illness?": "family_history",
    "Have you ever sought treatment for a mental health issue from a mental health professional?": "treatment",
    "If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?": "work_interfere",
    "How many employees does your company or organization have?": "no_employees",
    "Do you work remotely?": "remote_work",
    "Is your employer primarily a tech company/organization?": "tech_company",
    "Does your employer provide mental health benefits as part of healthcare coverage?": "benefits",
    "Do you know the options for mental health care available under your employer-provided coverage?": "care_options",
    "Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?": "wellness_program",
    "Does your employer offer resources to learn more about mental health concerns and options for seeking help?": "seek_help",
    "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?": "anonymity",
    "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:": "leave",
    "Do you think that discussing a mental health disorder with your employer would have negative consequences?": "mental_health_consequence",
    "Would you feel comfortable discussing a mental health disorder with your coworkers?": "coworkers",
    "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?": "supervisor",
    "Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?": "obs_consequence"
})


# now let's check the columns of both datasets after renaming and how many distinct columns they have from each other
cols_2014 = set(df_2014.columns)
cols_2016 = set(df_2016.columns)

shared_columns = sorted(cols_2014.intersection(cols_2016))
only_2014 = sorted(cols_2014 - cols_2016)
only_2016 = sorted(cols_2016 - cols_2014)

# compare columns before keeping only shared ones
print("\nOrtak sütunları tutmadan önce sütun karşılaştırması")
print("=" * 60)
print(f"2014 toplam sütunlar: {len(cols_2014)}")
print(f"2016 toplam sütunlar: {len(cols_2016)}")
print(f"Ortak sütunlar: {len(shared_columns)}")
print(f"Sadece 2014'ta: {len(only_2014)}")
print(f"Sadece 2016'da: {len(only_2016)}")


print("\nSadece 2014'ta sütunlar:")
for col in only_2014:
    print(" -", col)


print("\nSadece 2016'da sütunlar:")
for col in only_2016:
    print(" -", col)

print("\nOrtak sütunlar:")
for col in shared_columns:
    print(" -", col)

print("\nOrtak sütunlar: ", len(shared_columns))

# 3. We will keep shared columns only

shared_columns = [col for col in df_2014.columns if col in df_2016.columns]
df_2014 = df_2014[shared_columns]
df_2016 = df_2016[shared_columns]

df_2014["survey_year"] = 2014
df_2016["survey_year"] = 2016

df = pd.concat([df_2014, df_2016], ignore_index=True)
print("Birleştirilmiş şekil:", df.shape)

# 4. Next clean target variable

df["treatment"] = df["treatment"].astype(str).str.strip().str.capitalize()
df["treatment"] = df["treatment"].map({"Yes": 1, "No": 0})
df = df.dropna(subset=["treatment"])
df["treatment"] = df["treatment"].astype(int)

# 5. clean age and also for this project I want to keep only ages 15-75

df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df = df[df["Age"].between(15, 75)]

# 6. next clean gender
# there are lots of inconsistencies in gender column values so I had to do some preprocessing
distinct_values = df['Gender'].unique()
print("\n“Cinsiyet” sütunundaki benzersiz değerler: ", distinct_values.tolist())


def clean_gender(value):
    if pd.isna(value):
        return "Other"
    v = str(value).strip().lower()
    male_terms = ["male","m","man","men","cis male","cis man","mail","mal",
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

# 7. Handle missing values with Unkown

for col in ["work_interfere", "benefits", "care_options", "no_employees",
            "remote_work", "family_history", "mental_health_consequence"]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")


# 8. Create charts

# 1. Treatment distribution
plt.figure(figsize=(7,5))
treatment_sns = sns.countplot(data=df, x="treatment")
for container in treatment_sns.containers:
    treatment_sns.bar_label(container)
plt.title("Tedavi Dağılımı")
plt.xlabel("Tedavi (0 = Hayır, 1 = Evet)")
plt.ylabel("Sayı")
treatment_sns.set_xticks([0, 1])
treatment_sns.set_xticklabels(['Hayır', 'Evet'])
plt.tight_layout()
plt.savefig("Dataset Visuals/01_treatment_distribution.png", dpi=150)
plt.show()

# 2. Gender distribution
plt.figure(figsize=(7,5))
gender_sns = sns.countplot(data=df, x="Gender", order=df["Gender"].value_counts().index)
for container in gender_sns.containers:
    gender_sns.bar_label(container)
plt.title("Cinsiyet Dağılımı")
plt.xlabel("Cinsiyet")
plt.ylabel("Sayı")
gender_sns.set_xticks(range(len(df["Gender"].unique())))
gender_sns.set_xticklabels(['Erkek', 'Kadın', 'Diğer'])
plt.tight_layout()
plt.savefig("Dataset Visuals/02_gender_distribution.png", dpi=150)
plt.show()

# 3. Missing values before cleaning
top_missing = missing_before.sum(axis=1).sort_values(ascending=False).head(15)
plt.figure(figsize=(30,8)) # Genişlik artırıldı
missing_values_sns = sns.barplot(x=top_missing.values, y=top_missing.index)
for container in missing_values_sns.containers:
    missing_values_sns.bar_label(container)
plt.title("Eksik Değerler İçeren İlk 15 Sütun\nTemizlemeden Önce")
plt.xlabel("Eksik Değer Sayısı")
plt.ylabel("Sütun")
# left=0.4 sayesinde çok uzun cümleler bile sığacaktır
plt.subplots_adjust(left=0.4, right=0.95, top=0.9, bottom=0.1)
plt.savefig("Dataset Visuals/03_missing_values_before_cleaning.png", dpi=150)
plt.show()

# 4. Age distribution by treatment
plt.figure(figsize=(8,5))
age_sns = sns.histplot(data=df, x="Age", hue="treatment", bins=20, kde=True, multiple="stack")
plt.title("Tedaviye Göre Yaş Dağılımı")
plt.xlabel("Yaş")
plt.ylabel("Sayı")
plt.legend(title='Tedavi', labels=['Evet', 'Hayır'])
plt.tight_layout()
plt.savefig("Dataset Visuals/04_age_distribution_by_treatment.png", dpi=150)
plt.show()

# 5. Treatment rate by gender
gender_rate = df.groupby("Gender")["treatment"].mean().reset_index()
gender_rate["treatment"] = gender_rate["treatment"] * 100
plt.figure(figsize=(7,5))
treatment_rate_gender_sns = sns.barplot(data=gender_rate, x="Gender", y="treatment")
for container in treatment_rate_gender_sns.containers:
    treatment_rate_gender_sns.bar_label(container, fmt='%.1f%%')
plt.title("Cinsiyete Göre Tedavi Oranı")
plt.xlabel("Cinsiyet")
plt.ylabel("Tedavi Oranı (%)")
treatment_rate_gender_sns.set_xticks(range(len(gender_rate)))
treatment_rate_gender_sns.set_xticklabels(['Erkek', 'Kadın', 'Diğer'])
plt.tight_layout()
plt.savefig("Dataset Visuals/05_treatment_rate_by_gender.png", dpi=150)
plt.show()

# 6. Treatment rate by family history
fh_rate = df.groupby("family_history")["treatment"].mean().reset_index()
fh_rate["treatment"] = fh_rate["treatment"] * 100
plt.figure(figsize=(7,5))
treatment_rate_family_sns = sns.barplot(data=fh_rate, x="family_history", y="treatment")
for container in treatment_rate_family_sns.containers:
    treatment_rate_family_sns.bar_label(container, fmt='%.1f%%')
plt.title("Aile Geçmişine Göre Tedavi Oranı")
plt.xlabel("Aile Geçmişi")
plt.ylabel("Tedavi Oranı (%)")
treatment_rate_family_sns.set_xticks([0, 1])
treatment_rate_family_sns.set_xticklabels(['Hayır', 'Evet'])
plt.tight_layout()
plt.savefig("Dataset Visuals/06_treatment_rate_by_family_history.png", dpi=150)
plt.show()

# 7. Treatment rate by benefits
benefit_rate = df.groupby("benefits")["treatment"].mean().reset_index()
benefit_rate["treatment"] = benefit_rate["treatment"] * 100
plt.figure(figsize=(8,5))
treatment_rate_benefit_sns = sns.barplot(data=benefit_rate, x="benefits", y="treatment")
for container in treatment_rate_benefit_sns.containers:
    treatment_rate_benefit_sns.bar_label(container, fmt='%.1f%%')
plt.title("Ruh Sağlığı Yardımlarına Göre Tedavi Oranı")
plt.xlabel("Avantajlar")
plt.ylabel("Tedavi Oranı (%)")
treatment_rate_benefit_sns.set_xticks(range(len(benefit_rate)))
treatment_rate_benefit_sns.set_xticklabels(['Bilmiyorum', 'Hayır', 'Evet'])
plt.tight_layout()
plt.savefig("Dataset Visuals/07_treatment_rate_by_benefits.png", dpi=150)
plt.show()

# 8. Treatment rate by work interference
wi_rate = df.groupby("work_interfere")["treatment"].mean().reset_index()
wi_rate["treatment"] = wi_rate["treatment"] * 100
order = ["Never", "Rarely", "Sometimes", "Often", "Unknown"]
# Deprecated uyarısı için önce string'e zorluyoruz
wi_rate["work_interfere"] = wi_rate["work_interfere"].astype(str)
wi_rate["work_interfere"] = pd.Categorical(wi_rate["work_interfere"], categories=order, ordered=True)
wi_rate = wi_rate.sort_values("work_interfere")

plt.figure(figsize=(8,5))
wi_sns = sns.barplot(data=wi_rate, x="work_interfere", y="treatment")
for container in wi_sns.containers:
    wi_sns.bar_label(container, fmt='%.1f%%')
plt.title("İş Hayatına Müdahaleye Göre Tedavi Oranı")
plt.xlabel("İş Hayatına Etkisi (Sıklık)")
plt.ylabel("Tedavi Oranı (%)")
wi_sns.set_xticks(range(len(order)))
wi_sns.set_xticklabels(['Hiçbir zaman', 'Nadiren', 'Bazen', 'Sıklıkla', 'Bilinmiyor'])
plt.tight_layout(pad=2.0)
plt.savefig("Dataset Visuals/08_treatment_rate_by_work_interfere.png", dpi=150)
plt.show()

# 9. Treatment rate by company size
size_rate = df.groupby("no_employees")["treatment"].mean().reset_index()
size_rate["treatment"] = size_rate["treatment"] * 100
size_order = ['1-5', '6-25', '26-100', '100-500', '501-1000', 'More than 1000']
# Deprecated uyarısı için önce string'e zorluyoruz
size_rate["no_employees"] = size_rate["no_employees"].astype(str)
size_rate["no_employees"] = pd.Categorical(size_rate["no_employees"], categories=size_order, ordered=True)
size_rate = size_rate.sort_values("no_employees")

plt.figure(figsize=(9,5))
size_sns = sns.barplot(data=size_rate, x="no_employees", y="treatment")
for container in size_sns.containers:
    size_sns.bar_label(container, fmt='%.1f%%')
plt.title("Şirket Büyüklüğüne Göre Tedavi Oranı")
plt.xlabel("Çalışan Sayısı")
plt.ylabel("Tedavi Oranı (%)")
size_sns.set_xticks(range(len(size_order)))
size_sns.set_xticklabels(['1-5', '6-25', '26-100', '100-500', '501-1000', '1000+'])
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("Dataset Visuals/09_treatment_rate_by_company_size.png", dpi=150)
plt.show()

# 10. Treatment rate by remote work
remote_rate = df.groupby("remote_work")["treatment"].mean().reset_index()
remote_rate["treatment"] = remote_rate["treatment"] * 100
plt.figure(figsize=(7,5))
remote_sns = sns.barplot(data=remote_rate, x="remote_work", y="treatment")
for container in remote_sns.containers:
    remote_sns.bar_label(container, fmt='%.1f%%')
plt.title("Uzaktan Çalışmaya Göre Tedavi Oranı")
plt.xlabel("Uzaktan Çalışma")
plt.ylabel("Tedavi Oranı (%)")
remote_sns.set_xticks([0, 1])
remote_sns.set_xticklabels(['Hayır', 'Evet'])
plt.tight_layout(pad=2.0)
plt.savefig("Dataset Visuals/10_treatment_rate_by_remote_work.png", dpi=150)
plt.show()

# Türkçe karakterlerin düzgün görünmesi için
plt.rcParams["font.family"] = "DejaVu Sans"

# Grafik stili
sns.set_theme(style="whitegrid")


# helpful functions

def add_value_labels_vertical(ax, fmt="{:.2f}", suffix="", fontsize=9):
    for p in ax.patches:
        height = p.get_height()
        if height <= 0:
            continue
        if pd.isna(height):
            continue
        ax.annotate(
            fmt.format(height) + suffix,
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            xytext=(0, 4),
            textcoords="offset points"
        )


def add_value_labels_horizontal(ax, fmt="{:.4f}", suffix="", fontsize=9):

    for p in ax.patches:
        width = p.get_width()
        if pd.isna(width):
            continue
        ax.annotate(
            fmt.format(width) + suffix,
            (width, p.get_y() + p.get_height() / 2.0),
            ha="left",
            va="center",
            fontsize=fontsize,
            xytext=(5, 0),
            textcoords="offset points"
        )


def safe_filename(text):
    replacements = {
        "ı": "i", "İ": "I",
        "ş": "s", "Ş": "S",
        "ğ": "g", "Ğ": "G",
        "ü": "u", "Ü": "U",
        "ö": "o", "Ö": "O",
        "ç": "c", "Ç": "C",
        " ": "_"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.lower()

# 1. upload tested values inside models folder

X_test = joblib.load("Models/X_test.joblib")
y_test = joblib.load("Models/y_test.joblib")

print("Test verileri başarıyla yüklendi.")
print(f"X_test boyutu: {X_test.shape}")
print(f"y_test boyutu: {y_test.shape}")


# 2. upload models

model_files = {
    "Lojistik Regresyon": "Models/tuned_logistic_regression.joblib",
    "Rastgele Orman": "Models/tuned_random_forest.joblib",
    "XGBoost": "Models/tuned_xgboost.joblib"
}

models = {}

print("\nModel dosyaları kontrol ediliyor...")
for model_name, file_name in model_files.items():
    if os.path.exists(file_name):
        models[model_name] = joblib.load(file_name)
        print(f"Yüklendi: {model_name} -> {file_name}")
    else:
        print(f"Bulunamadı: {file_name}  |  {model_name} atlandı.") # skips if it doesn't find

if len(models) == 0:
    raise FileNotFoundError(
        "Hiçbir kayıtlı model dosyası bulunamadı. "
        "Önce main.py ile modelleri kaydetmelisin."
    )


# 3.generate models predictions and performance results

results = []
predictions = {}

print("\nModel performansları hesaplanıyor...")

for model_name, model in models.items():
    preds = model.predict(X_test)
    predictions[model_name] = preds

    acc = accuracy_score(y_test, preds) * 100
    prec = precision_score(y_test, preds, zero_division=0) * 100
    rec = recall_score(y_test, preds, zero_division=0) * 100
    f1 = f1_score(y_test, preds, zero_division=0) * 100

    results.append({
        "Model": model_name,
        "Doğruluk": round(acc, 2),
        "Kesinlik": round(prec, 2),
        "Duyarlılık": round(rec, 2),
        "F1-Skoru": round(f1, 2)
    })

    print("-" * 50)
    print(f"Model: {model_name}")
    print(f"Accuracy : {acc:.2f}%")
    print(f"Precision : {prec:.2f}%")
    print(f"Recall: {rec:.2f}%")
    print(f"F1-Score : {f1:.2f}%")

results_df = pd.DataFrame(results)

print("\nModel Performans Tablosu:")
print(results_df.to_string(index=False))

results_df.to_csv("Dataset Visuals/model_karsilastirma_sonuclari.csv", index=False, encoding="utf-8-sig")
print("\nCSV kaydedildi: model_karsilastirma_sonuclari.csv")

# 4. model performance comparison chart

results_melted = results_df.melt(
    id_vars="Model",
    value_vars=["Doğruluk", "Kesinlik", "Duyarlılık", "F1-Skoru"],
    var_name="Metrik",
    value_name="Skor (%)"
)

plt.figure(figsize=(12, 7))
ax = sns.barplot(
    data=results_melted,
    x="Metrik",
    y="Skor (%)",
    hue="Model"
)

plt.title("Modellerin Performans Karşılaştırması", fontsize=14, fontweight="bold")
plt.xlabel("Değerlendirme Metriği")
plt.ylabel("Skor (%)")
plt.ylim(0, 110)

add_value_labels_vertical(ax, fmt="{:.2f}", suffix="%")
plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.savefig("Dataset Visuals/model_performans_karsilastirma.png", dpi=150, bbox_inches="tight")
plt.show()

print("Kaydedildi: model_performans_karsilastirma.png")


# 5. confusion matrix for each model

for model_name, preds in predictions.items():
    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Tedavi Aramadı", "Tedavi Aradı"]
    )

    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title(f"Karışıklık Matrisi - {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tahmin Edilen Sınıf")
    ax.set_ylabel("Gerçek Sınıf")

    file_name = f"karisiklik_matrisi_{safe_filename(model_name)}.png"
    plt.tight_layout()
    plt.savefig("Dataset Visuals/"+file_name, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Kaydedildi: {file_name}")


# 6. Feature Importance

for model_name, model in models.items():
    file_suffix = safe_filename(model_name)

    # Ağaç tabanlı modeller için feature importance
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Özellik": X_test.columns,
            "Önem": model.feature_importances_
        }).sort_values("Önem", ascending=False).head(10)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=importance_df,
            x="Önem",
            y="Özellik"
        )

        max_val = importance_df["Önem"].max()
        plt.xlim(0, max_val * 1.20)

        plt.title(f"İlk 10 Özellik Önemi - {model_name}", fontsize=13, fontweight="bold")
        plt.xlabel("Önem Düzeyi")
        plt.ylabel("Özellik")

        add_value_labels_horizontal(ax, fmt="{:.4f}")

        file_name = f"ozellik_onemi_ilk10_{file_suffix}.png"
        plt.tight_layout()
        plt.savefig("Dataset Visuals/"+file_name, dpi=150, bbox_inches="tight")
        plt.show()

        print(f"Kaydedildi: {file_name}")

    # Lojistik Regresyon için katsayılar
    elif hasattr(model, "coef_"):
        coef_df = pd.DataFrame({
            "Özellik": X_test.columns,
            "Katsayı": model.coef_[0]
        })

        coef_df["Mutlak Katsayı"] = coef_df["Katsayı"].abs()

        top_coef_df = coef_df.sort_values(
            "Mutlak Katsayı",
            ascending=False
        ).head(10)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=top_coef_df,
            x="Mutlak Katsayı",
            y="Özellik"
        )

        max_val = top_coef_df["Mutlak Katsayı"].max()
        plt.xlim(0, max_val * 1.20)

        plt.title(f"En Etkili 10 Özellik - {model_name}", fontsize=13, fontweight="bold")
        plt.xlabel("Mutlak Katsayı Değeri")
        plt.ylabel("Özellik")

        add_value_labels_horizontal(ax, fmt="{:.4f}")

        file_name = f"en_etkili_10_ozellik_{file_suffix}.png"
        plt.tight_layout()
        plt.savefig("Dataset Visuals/"+file_name, dpi=150, bbox_inches="tight")
        plt.show()

        print(f"Kaydedildi: {file_name}")

    else:
        print(f"{model_name} için özellik önemi veya katsayı grafiği oluşturulamadı.")


# 7. save results table as an image

fig, ax = plt.subplots(figsize=(10, 2 + len(results_df) * 0.5))
ax.axis("off")

table = ax.table(
    cellText=results_df.values,
    colLabels=results_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

plt.title("Model Performans Sonuçları Tablosu", fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("Dataset Visuals/model_performans_tablosu.png", dpi=150, bbox_inches="tight")
plt.show()

print("Kaydedildi: model_performans_tablosu.png")

# 11. Korelasyon Isı Haritası (Correlation Heatmap)
print("\nKorelasyon Isı Haritası oluşturuluyor...")

# Sadece en önemli sütunları alalım ki grafik çok karmaşık olmasın
cols_to_plot = ['treatment', 'Age', 'family_history', 'work_interfere',
                'benefits', 'care_options', 'wellness_program', 'anonymity', 'leave']

df_encoded = df[cols_to_plot].copy()

# Kategorik verileri ısı haritasında görebilmek için sayısal formata (label encoding) çeviriyoruz.
# Hata almamak için Age ve treatment (zaten sayısal) dışındaki tüm sütunları zorla dönüştürüyoruz.
for col in df_encoded.columns:
    if col not in ['treatment', 'Age']:
        df_encoded[col] = df_encoded[col].astype(str).astype('category').cat.codes

plt.figure(figsize=(14, 10))
corr_matrix = df_encoded.corr()

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.title("Özellikler Arası Korelasyon Isı Haritası", fontsize=15, fontweight="bold")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("Dataset Visuals/11_correlation_heatmap.png", dpi=150)
plt.show()


# 12. Keman Grafiği (Violin Plot) - Yaş, Cinsiyet ve Tedavi İlişkisi
# Veri dağılımını (yoğunluğunu) en iyi gösteren akademik grafiklerden biridir.
print("\nKeman Grafiği oluşturuluyor...")
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="Gender", y="Age", hue="treatment", split=True, inner="quart", palette="muted")
plt.title("Cinsiyet ve Tedavi Arayışına Göre Yaş Dağılımı Yoğunluğu", fontsize=14, fontweight="bold")
plt.xlabel("Cinsiyet", fontsize=12)
plt.ylabel("Yaş", fontsize=12)
plt.legend(title="Tedavi Arayışı", labels=["0 - Hayır", "1 - Evet"], loc="upper left")
plt.tight_layout()
plt.savefig("Dataset Visuals/12_age_gender_treatment_violin.png", dpi=150)
plt.show()


# 13. Yığılmış Çubuk Grafiği (Stacked Bar) - Şirket Büyüklüğü vs. Sağlık Yardımları
# İnsan kaynakları bakış açısı için çok değerlidir. Şirket büyüdükçe imkanlar artıyor mu?
print("\nŞirket Büyüklüğü ve Yardımlar Grafiği oluşturuluyor...")
size_benefits = df.groupby(['no_employees', 'benefits']).size().unstack(fill_value=0)

# Şirket boyutlarını mantıklı bir sıraya dizelim
size_order = ['1-5', '6-25', '26-100', '100-500', '501-1000', 'More than 1000']
# Verideki isimleri kontrol ederek eşleştiriyoruz
existing_sizes = [s for s in size_order if s in size_benefits.index]
size_benefits = size_benefits.reindex(existing_sizes)

# Yüzdelik oranlara çevirelim ki adil bir karşılaştırma olsun (Her şirketin toplamını %100 yapıyoruz)
size_benefits_pct = size_benefits.div(size_benefits.sum(axis=1), axis=0) * 100

ax = size_benefits_pct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis', edgecolor='black')
plt.title("Şirket Büyüklüğüne Göre Ruh Sağlığı Yardımı (Benefits) Sunma Oranları", fontsize=14, fontweight="bold")
plt.xlabel("Şirket Çalışan Sayısı", fontsize=12)
plt.ylabel("Yüzdelik Oran (%)", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title="Yardım Sunuluyor mu?", bbox_to_anchor=(1.05, 1), loc='upper left')

# Çubukların içine % değerlerini yazdırma
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    if height > 5: # Çok küçük dilimlere yazı yazma (görüntü bozulmasın)
        ax.text(x + width/2, y + height/2, f'{height:.1f}%', horizontalalignment='center', verticalalignment='center', color='white', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig("Dataset Visuals/13_company_size_vs_benefits_stacked.png", dpi=150)
plt.show()
print("Tüm yeni görseller 'Dataset Visuals' klasörüne başarıyla kaydedildi!")

# --- BU KODLARI mental_health_datasets_visualizations.py DOSYANIZIN EN ALTINA EKLEYİN ---

# 14. Veri Seti Kesişim Grafiği (2014 vs 2016)
# Bu grafik, metodolojide anlattığımız "sadece ortak sütunları aldık" teorisini kanıtlar.
print("\nVeri Seti Kesişim Grafiği oluşturuluyor...")
plt.figure(figsize=(9, 5))
dataset_names = ['2014 Anketi', '2016 Anketi', 'Birleştirilmiş (Ortak) Veri']
column_counts = [len(cols_2014), len(cols_2016), len(shared_columns)]

ax1 = sns.barplot(x=dataset_names, y=column_counts, palette=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black')
plt.title("2014 ve 2016 Veri Setlerindeki Özellik (Sütun) Sayılarının Karşılaştırması", fontsize=13, fontweight="bold")
plt.ylabel("Sütun (Özellik) Sayısı", fontsize=12)

# Çubukların üzerine sayıları yazdırıyoruz
for p in ax1.patches:
    ax1.annotate(f"{int(p.get_height())} Sütun",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='black', xytext=(0, 5), textcoords='offset points')

plt.ylim(0, max(column_counts) + 15)
plt.tight_layout()
plt.savefig("Dataset Visuals/14_dataset_intersection.png", dpi=150)
plt.show()


# 15. One-Hot Encoding Öncesi ve Sonrası Veri Boyutu (Dimensionality)
# Neden PCA kullanmadık, boyutu ne kadardı argumanını kanıtlar.
print("\nOne-Hot Encoding Boyut Etkisi Grafiği oluşturuluyor...")

# One-hot encoding işlemini simüle edelim (sadece sayısını bulmak için)
df_for_encoding = df.drop(columns=["treatment", "survey_year", "Timestamp"], errors="ignore")
cols_before = len(df_for_encoding.columns)
# get dummies uygulayarak yeni sütun sayısını bulalım
df_encoded_sim = pd.get_dummies(df_for_encoding, drop_first=True)
cols_after = len(df_encoded_sim.columns)

plt.figure(figsize=(8, 5))
encoding_labels = ['Kategorik Hal (Encoding Öncesi)', 'Genişletilmiş Hal (Encoding Sonrası)']
encoding_counts = [cols_before, cols_after]

ax2 = sns.barplot(x=encoding_labels, y=encoding_counts, palette=['#f39c12', '#9b59b6'], edgecolor='black')
plt.title("One-Hot Encoding İşleminin Veri Boyutuna (Dimensionality) Etkisi", fontsize=13, fontweight="bold")
plt.ylabel("Toplam Özellik (Feature) Sayısı", fontsize=12)

for p in ax2.patches:
    ax2.annotate(f"{int(p.get_height())} Özellik",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='black', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig("Dataset Visuals/15_onehot_encoding_impact.png", dpi=150)
plt.show()
print("Yeni metodoloji grafikleri 'Dataset Visuals' klasörüne kaydedildi!")


# --- ŞEKİL 1. HEDEF DEĞİŞKEN (TREATMENT) SINIF DAĞILIMI ---
# Amacı: Veri setinde "Sınıf Dengesizliği" (Class Imbalance) olmadığını kanıtlamak.
plt.figure(figsize=(7, 5))
ax = sns.countplot(x='treatment', data=df, palette=['#e74c3c', '#2ecc71'], edgecolor='black')
plt.title("Şekil 1: Hedef Değişken (Tedavi Arayışı) Dağılımı", fontsize=14, fontweight="bold")
plt.xlabel("Tedavi Aradı mı?", fontsize=12)
plt.ylabel("Kişi Sayısı", fontsize=12)
plt.xticks(ticks=[0, 1], labels=['0 - Hayır (Aramadı)', '1 - Evet (Aradı)'], fontsize=11)

# Çubukların üzerine sayıları ve YÜZDELERİ yazdırıyoruz (Akademik standart)
total = len(df)
for p in ax.patches:
    height = p.get_height()
    percentage = f'{(100 * height / total):.1f}%'
    ax.annotate(f"{int(height)} Kişi\n({percentage})",
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='black', xytext=(0, 5), textcoords='offset points')

plt.ylim(0, max(df['treatment'].value_counts()) * 1.2) # Üstten biraz boşluk bırak
plt.tight_layout()
plt.savefig("Dataset Visuals/Sekil_1_Treatment_Dagilimi.png", dpi=150)
plt.show()


# --- ŞEKİL 2. CİNSİYET DAĞILIMI VE TEMSİL ---
# Amacı: Erkek çoğunluğunu göstererek "Fairness Audit" modülünü neden yazdığımızı haklı çıkarmak.
plt.figure(figsize=(8, 5))
# Gender sütununu sayalım ve büyükten küçüğe sıralayalım
gender_counts = df['Gender'].value_counts()
ax2 = sns.barplot(x=gender_counts.index, y=gender_counts.values, palette='Blues_r', edgecolor='black')

plt.title("Şekil 2: Veri Seti Cinsiyet Dağılımı (Temsil Sınırlılığı)", fontsize=14, fontweight="bold")
plt.xlabel("Cinsiyet Grubu", fontsize=12)
plt.ylabel("Kişi Sayısı", fontsize=12)

for p in ax2.patches:
    height = p.get_height()
    percentage = f'{(100 * height / total):.1f}%'
    ax2.annotate(f"{int(height)}\n({percentage})",
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='black', xytext=(0, 5), textcoords='offset points')

plt.ylim(0, max(gender_counts.values) * 1.15)
plt.tight_layout()
plt.savefig("Dataset Visuals/Sekil_2_Cinsiyet_Dagilimi.png", dpi=150)
plt.show()


# --- ŞEKİL 3. TEMİZLEME ÖNCESİ EKSİK DEĞERLER (MISSING VALUES) ---
# Amacı: Veri ön işlemenin (Unknown ile doldurmanın) ne kadar gerekli olduğunu görselleştirmek.
# Bunun için ham (temizlenmemiş) verinin kopyasını simüle ediyoruz.
print("Eksik veri analizi grafiği hazırlanıyor...")

# Birleştirilmiş ham veri setindeki eksik değerleri hesapla (sadece eksik olanları al)
missing_data = df_2014.isnull().sum().add(df_2016.isnull().sum(), fill_value=0)
missing_data = missing_data[missing_data > 0].sort_values(ascending=False).head(8) # En çok eksik olan ilk 8 sütun

plt.figure(figsize=(10, 6))
ax3 = sns.barplot(x=missing_data.values, y=missing_data.index, palette='Reds_r', edgecolor='black')
plt.title("Şekil 3: Veri Temizleme Öncesi En Çok Eksik Değer İçeren Sütunlar", fontsize=14, fontweight="bold")
plt.xlabel("Eksik (NaN) Satır Sayısı", fontsize=12)
plt.ylabel("Anket Soruları (Sütunlar)", fontsize=12)

# Çubukların yanına sayıları yazdır
for p in ax3.patches:
    width = p.get_width()
    ax3.annotate(f"{int(width)}",
                (width, p.get_y() + p.get_height() / 2.),
                ha='left', va='center', fontsize=10, fontweight='bold', color='black', xytext=(5, 0), textcoords='offset points')

plt.xlim(0, max(missing_data.values) * 1.15)
plt.tight_layout()
plt.savefig("Dataset Visuals/Sekil_3_Eksik_Degerler.png", dpi=150)
plt.show()

print("Şekil 1, Şekil 2 ve Şekil 3 'Dataset Visuals' klasörüne başarıyla kaydedildi!")