import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from itertools import combinations


def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    return df


csvData = 'oddsData.csv'

df2019 = pd.read_csv(csvData, delimiter=',', encoding='utf-8')
df2019 = df2019[df2019['season'] == 2019].reset_index(drop=True)

# My data has a duplicate for each game, f.e.: Utah vs. Miami and Miami vs. Utah
# Row count went 2460 -> 1230
rows_to_drop = []
for i in range(len(df2019)):
    gameDate = df2019.loc[i, "date"]
    homeTeam = df2019.loc[i, "team"]

    j = i + 1
    while j < len(df2019) and df2019.loc[j, "date"] == gameDate:
        if df2019.loc[j, "opponent"] == homeTeam:
            rows_to_drop.append(j)
            break
        j += 1

df2019 = df2019.drop(rows_to_drop).reset_index(drop=True)

df2019 = df2019.drop(columns=["season", "date", "home/visitor", "team", "opponent"])
continuous = ["score", "opponentScore", "moneyLine", "opponentMoneyLine", "total", "spread", "secondHalfTotal"]

df2019 = df2019.dropna()


# print(df2019.head())

scaler = MinMaxScaler()

df2019[continuous] = scaler.fit_transform(df2019[continuous])

# print(df2019)

df2019 = remove_outliers(df2019, continuous)

for f1, f2 in itertools.combinations(continuous, 2):
    att_pair = df2019[[f1, f2]]

    for i in range(2, 9):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=5)
        labels = kmeans.fit_predict(att_pair)

results_list = []

for f1, f2, f3 in itertools.combinations(continuous, 3):
    att_pair = df2019[[f1, f2, f3]]

    for i in range(2, 9):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=5)
        labels = kmeans.fit_predict(att_pair)

        inertia = kmeans.inertia_
        sil_score = silhouette_score(att_pair, labels)

        results_list.append({
            "attr_pair": f"{f1}, {f2}, {f3}",
            "n_clusters": i,
            "inertia": inertia,
            "silhouette": sil_score
        })

results_df = pd.DataFrame(results_list)

print(results_df.to_string())
results_df.to_csv("kmeans_results.csv", index=False, sep=';')

best_pairs = results_df.loc[results_df.groupby('attr_pair')['silhouette'].idxmax()]

top3_pairs = best_pairs.sort_values(by='silhouette', ascending=False).head(3)
print(top3_pairs)

palette = sns.color_palette("tab10")  # up to 10 unique colors

for idx, row in top3_pairs.iterrows():
    f1, f2 = row['attr_pair'].split(', ')
    k_opt = row['n_clusters']

    X_pair = df2019[[f1, f2]]

    labels = KMeans(n_clusters=k_opt, n_init=5, random_state=42).fit_predict(X_pair)

    # Simple scatter plot, color by cluster labels
    # plt.figure(figsize=(6, 4))
    # plt.scatter(X_pair[f1], X_pair[f2], c=labels, cmap='tab10', alpha=0.6)
    # plt.xlabel(f1)
    # plt.ylabel(f2)
    # plt.title(f"Klasteriai {f1} vs {f2}")
    # plt.show()

    # ----- Inertia line plot -----
    # subset = results_df[results_df['attr_pair'] == row['attr_pair']].sort_values('n_clusters')
    # plt.figure(figsize=(6, 4))
    # plt.plot(subset['n_clusters'], subset['inertia'], marker='o', color='red')
    # plt.xticks(subset['n_clusters'])
    # plt.xlabel("Number of Clusters (k)")
    # plt.ylabel("Inertia")
    # plt.title(f"Inertia vs Number of Clusters for {f1} vs {f2}")
    # plt.grid(True)
    # plt.show()

    # sil_values = silhouette_samples(X_pair, labels)
    # sil_avg = silhouette_score(X_pair, labels)
    #
    # plt.figure(figsize=(6, 4))
    # y_lower = 10
    #
    # for i in range(k_opt):
    #     cluster_vals = sil_values[labels == i]
    #     cluster_vals.sort()
    #
    #     size = cluster_vals.shape[0]
    #     y_upper = y_lower + size
    #
    #     plt.fill_betweenx(
    #         np.arange(y_lower, y_upper),
    #         0,
    #         cluster_vals,
    #         alpha=0.7
    #     )
    #
    #     plt.text(-0.05, y_lower + size / 2, f"C. {i + 1}")
    #     y_lower = y_upper + 10
    #
    # plt.axvline(sil_avg, linestyle='--', color='red',
    #             label=f"Vid. siluetas = {sil_avg:.2f}")
    #
    # plt.xlabel("Silueto koeficientas")
    # plt.ylabel("Duomenų taškai")
    # plt.title(f"Silueto diagrama: {f1} vs {f2} (k={k_opt})")
    # plt.legend()
    # plt.show()