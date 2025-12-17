import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples


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

scaler = MinMaxScaler()
df2019[continuous] = scaler.fit_transform(df2019[continuous])
df2019 = remove_outliers(df2019, continuous)

X = df2019[continuous].values
results_all_features = []

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
    labels = kmeans.fit_predict(X)

    inertia = kmeans.inertia_
    sil_score = silhouette_score(X, labels)

    results_all_features.append({
        "k": k,
        "inertia": inertia,
        "silhouette": sil_score
    })

output_row = {'Attribute': 'All 7 Attributes'}

for res in results_all_features:
    k = res['k']
    output_row[f'k={k} Inertia'] = res['inertia']
    output_row[f'k={k} Silhouette'] = res['silhouette']

formatted_df = pd.DataFrame([output_row])

filename = "kmeans_results_all_7.csv"
formatted_df.to_csv(filename, index=False, sep=';')

print(formatted_df.to_string(index=False))

results_df = pd.DataFrame(results_all_features)
top3_k = results_df.sort_values(by="silhouette", ascending=False).head(3)

print(top3_k.to_string(index=False))

for idx, (index, row) in enumerate(top3_k.iterrows()):
    k = int(row['k'])
    score = row['silhouette']

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
    cluster_labels = kmeans.fit_predict(X)

    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    silhouette_avg = silhouette_score(X, cluster_labels)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(X) + (k + 1) * 10])

    y_lower = 10
    for i in range(k):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / k)

        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10

    ax.set_title(
        f"TOP {idx + 1}: Silueto diagrama (Visi 7 atributai)\n(K={k}, Vidutinis siluetas={silhouette_avg:.3f})")
    ax.set_xlabel("Silueto koeficientas")
    ax.set_ylabel("Klasterio numeris")

    ax.axvline(x=silhouette_avg, color="red", linestyle="--", label="Vidurkis")
    ax.legend()

    fig.canvas.manager.set_window_title(f"Top {idx + 1}: k={k}")

    plt.tight_layout()
    plt.show()

plt.figure(figsize=(8, 6))
plt.plot(results_df['k'], results_df['inertia'], marker='o', linestyle='-', color='tab:blue', linewidth=2)

plt.title("Alkūnės grafikas (Visi 7 atributai)")
plt.xlabel("Klasterių skaičius (k)")
plt.ylabel("Inercija")
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(2, 9))

plt.tight_layout()
plt.show()
