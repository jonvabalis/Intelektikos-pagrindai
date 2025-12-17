import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from clustergram import Clustergram

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
df_scaled = pd.DataFrame(scaler.fit_transform(df2019[continuous]), columns=continuous)

df_scaled = remove_outliers(df_scaled, continuous)

df_scaled = df_scaled.reset_index(drop=True)

cgram = Clustergram(range(1, 9), method='kmeans', backend='sklearn')

cgram.fit(df_scaled)

fig, ax = plt.subplots(figsize=(12, 8))

cgram.plot(ax=ax)

ax.set_title("NBA Duomenų Clustergram (Matthias Schonlau metodas)", fontsize=14)
ax.set_xlabel("Klasterių skaičius (k)", fontsize=12)
ax.set_ylabel("Klasterio vidurkis (PCA svertinė projekcija)", fontsize=12)

plt.tight_layout()
plt.show()