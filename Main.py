import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Heart_disease_cleveland_new.csv")

features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
healthy_levels = {
    "age": 50,
    "trestbps": 120,
    "chol": 200,
    "thalach": 100,
    "oldpeak": 0
}
healthy_ranges = ["<50", "<=120", "<200", ">100", "~0"]

feature_means = [round(df[feature].mean(), 2) for feature in features]

fig, ax = plt.subplots(figsize=(10, 4))

ax.bar(features, feature_means, color="skyblue", label="Dataset Mean")

ax.plot(features, list(healthy_levels.values()), color="green", linestyle="--", marker="o", label="Healthy Threshold")

ax.set_title("Heart Health Risk: Dataset Mean vs Healthy Levels", fontsize=11)
ax.set_ylabel("Value")
ax.grid(True, axis="y")
ax.legend()

table_data = [
    ["Feature", "Healthy Range", "Dataset Mean"]
] + [
    [features[i], healthy_ranges[i], feature_means[i]] for i in range(len(features))
]

table = plt.table(cellText=table_data, colWidths=[0.2]*3, cellLoc='center', loc='right')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.4)

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
for i, feature in enumerate(features):
    plt.subplot(1, len(features), i + 1)
    sns.boxplot(y=df[feature], color='lightcoral')
    plt.title(feature)
    plt.tight_layout()

plt.suptitle("Distribution and Outliers in Numeric Features", fontsize=14, y=1.05)
plt.show()

summary_df = pd.DataFrame({
    "Feature": features,
    "Healthy Range": healthy_ranges,
    "Dataset Mean": feature_means
})
summary_df.to_csv("summary_table.csv", index=False)
