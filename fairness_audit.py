# loads the saved model and test data then checks whether the model
# performs equally well for Male, Female and Other
# If accuracy is very different across groups then there is algorithmic bias
# can be run only if main.py has been executed

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 1. load saved files


model       = joblib.load("Models/best_model.joblib")
X_test      = joblib.load("Models/X_test.joblib")
y_test      = joblib.load("Models/y_test.joblib")
gender_test = joblib.load("Models/gender_test.joblib")

print("Dosyalar başarıyla yüklendi.")
print(f"Test kümesinin boyutu: {len(X_test)} satır\n")


# 2. get predictions on the full test set

all_preds = model.predict(X_test)

# 3. split by gender and measure performance per group

groups  = ["Male", "Female", "Other"]
results = []

for group in groups:
    mask = gender_test == group

    if mask.sum() < 10:
        print(f"{group} atlanıyor. Test kümesinde 10'dan az örnek var ({mask.sum()} bulundu).")
        continue

    y_group    = y_test[mask]
    pred_group = all_preds[mask]

    acc  = accuracy_score(y_group, pred_group)
    prec = precision_score(y_group, pred_group, zero_division=0)
    rec  = recall_score(y_group, pred_group, zero_division=0)
    f1   = f1_score(y_group, pred_group, zero_division=0)
    n    = mask.sum()

    results.append({
        "Gender":    group,
        "N":         n,
        "Accuracy":  round(acc  * 100, 1),
        "Precision": round(prec * 100, 1),
        "Recall":    round(rec  * 100, 1),
        "F1 Score":  round(f1   * 100, 1),
    })

    print(f"{'─' * 40}")
    print(f"  Group: {group}  (n={n})")
    print(f"  Accuracy:  {acc  * 100:.1f}%")
    print(f"  Precision: {prec * 100:.1f}%")
    print(f"  Recall:    {rec  * 100:.1f}%")
    print(f"  F1 Score:  {f1   * 100:.1f}%")

results_df = pd.DataFrame(results)
print("\n")
print(results_df.to_string(index=False))

# 4. flag bias if the gap is large

if len(results_df) >= 2:
    f1_max  = results_df["F1 Score"].max()
    f1_min  = results_df["F1 Score"].min()
    f1_gap  = f1_max - f1_min

    acc_max = results_df["Accuracy"].max()
    acc_min = results_df["Accuracy"].min()
    acc_gap = acc_max - acc_min

    print("\n" + "=" * 40)
    print("BIAS KONTROL ÖZETİ")
    print("=" * 40)
    print(f"  F1 Score gap across groups:  {f1_gap:.1f} percentage points")
    print(f"  Accuracy gap across groups:  {acc_gap:.1f} percentage points")

    if f1_gap > 10:
        print(
            f"\n  Notable bias detected: F1 differs by {f1_gap:.1f}pp across groups."
        )
        best_group  = results_df.loc[results_df["F1 Score"].idxmax(), "Gender"]
        worst_group = results_df.loc[results_df["F1 Score"].idxmin(), "Gender"]
        print(f"     Model performs best for {best_group} and worst for {worst_group}.")
    else:
        print(
            f"\n  No major bias detected (gap < 10pp). "
            "The model performs similarly across gender groups."
        )

# 5. visualise, grouped bar chart

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

gender_tr = {
    "Male": "Erkek",
    "Female": "Kadın",
    "Other": "Diğer"
}

plot_df = results_df.copy()
plot_df["Gender"] = plot_df["Gender"].map(gender_tr).fillna(plot_df["Gender"])

melted = plot_df.melt(
    id_vars="Gender", value_vars=metrics,
    var_name="Metric", value_name="Score (%)"
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.set_theme(style="whitegrid")

# left: grouped bar chart — all metrics by gender
sns.barplot(
    x="Metric", y="Score (%)", hue="Gender",
    data=melted, palette="viridis", ax=axes[0]
)
axes[0].set_title("Cinsiyet Gruplarına Göre Model Performansı",
                  fontsize=13, fontweight="bold")
axes[0].set_ylabel("Skor (%)")
axes[0].set_ylim(0, 110)
axes[0].set_xlabel("Ölçütler")
axes[0].legend(title="Cinsiyet")

# right: F1 score only — easier to read the gap
sns.barplot(
    x="Gender", y="F1 Score", hue="Gender", data=plot_df,
    palette="viridis", ax=axes[1], legend=False
)
axes[1].set_title("Cinsiyete Göre F1 Puan Farkı",
                  fontsize=13, fontweight="bold")
axes[1].set_ylabel("F1 Skor (%)")
axes[1].set_xlabel("Cinsiyet")
axes[1].set_ylim(0, 110)

for bar in axes[1].patches:
    height = bar.get_height()
    if height > 0:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

plt.tight_layout()
plt.savefig("Dataset Visuals/fairness_audit.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFairness chart saved as fairness_audit.png")