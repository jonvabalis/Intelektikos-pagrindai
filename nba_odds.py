import pandas as pd
import numpy as np

csvData = 'oddsData.csv'

df2019 = pd.read_csv(csvData, delimiter=',', encoding='utf-8')
df2019 = df2019[df2019['season'] == 2019].reset_index(drop=True)


def print_relevant_data(_df):
    print("Dataset shape:", _df.shape)
    print("\nColumn names:", _df.columns.tolist())
    print("\nFirst few rows:")
    print(_df.head())
    print("\nData types:")
    print(_df.dtypes)


def american_to_european(odds):
    if odds > 0:
        return 1 + (odds / 100)
    else:
        return 1 + (100 / abs(odds))


df2019["moneyLine"] = df2019["moneyLine"].apply(american_to_european)
df2019["opponentMoneyLine"] = df2019["opponentMoneyLine"].apply(american_to_european)

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

# I've decided to use 8 columns only
df2019 = df2019.drop(columns=['date', 'season', 'total', 'secondHalfTotal'])

# As my data has no missing entries, let's make a few fields missing
df2019.loc[10, "score"] = np.nan
df2019.loc[100, "opponentScore"] = np.nan
df2019.loc[1000, "team"] = np.nan

rowCount = len(df2019)
rowCountMissing = df2019.isnull().any(axis=1).sum()
missingPercentage = (rowCountMissing / rowCount) * 100

print(f"Current row count: {rowCount}, missing {rowCountMissing} rows,"
      f" thus missing data row percentage is {missingPercentage:.2f}%")

# As I have many rows of data, I will drop the rows with missing data
# Row count went 1230 -> 1227
df2019 = df2019.dropna()

print_relevant_data(df2019)
