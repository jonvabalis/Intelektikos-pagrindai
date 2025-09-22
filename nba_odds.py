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


def print_continuous_attribute_data(df, attribute):
    rowCount = len(df[attribute])
    rowCountMissing = df[attribute].isnull().sum()
    missingPercentage = (rowCountMissing / rowCount) * 100

    print(f"-----{attribute}-----")
    print(f"Row count: {len(df[attribute]) - df[attribute].isnull().sum()}")
    print(f"Missing data row percentage: {missingPercentage:.2f}%")
    print(f"Cardinality: {df[attribute].nunique()}")
    print(f"Min value: {df[attribute].min()}")
    print(f"Max value: {df[attribute].max()}")
    print(f"1st quantile: {df[attribute].quantile(0.25):.2f}")
    print(f"3rd quantile: {df[attribute].quantile(0.75):.2f}")
    print(f"Average: {df[attribute].mean():.2f}")
    print(f"Median: {df[attribute].median():.2f}")
    print(f"Standard deviation: {(df[attribute].std()):.2f}")


def print_categorical_attribute_data(df, attribute):
    rowCount = len(df[attribute])
    rowCountMissing = df[attribute].isnull().sum()
    missingPercentage = (rowCountMissing / rowCount) * 100

    print(f"-----{attribute}-----")
    print(f"Row count: {len(df[attribute]) - df[attribute].isnull().sum()}")
    print(f"Missing data row percentage: {missingPercentage:.2f}%")
    print(f"Cardinality: {df[attribute].nunique()}")
    print(f"Mode: {df[attribute].value_counts().index[0]}")
    print(f"Mode frequency value: {df[attribute].value_counts().iloc[0]}")
    print(f"Mode frequency percentage: {((df[attribute].value_counts().iloc[0] / len(df[attribute])) * 100):.2f}%")
    print(f"2nd mode: {df[attribute].value_counts().index[1]}")
    print(f"2nd mode frequency value: {df[attribute].value_counts().iloc[1]}")
    print(f"2nd mode frequency percentage: {((df[attribute].value_counts().iloc[1] / len(df[attribute])) * 100):.2f}%")


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
df2019.loc[20, "team"] = np.nan
df2019.loc[200, "home/visitor"] = np.nan
df2019.loc[1200, "team"] = np.nan
df2019.loc[1202, "team"] = np.nan
df2019.loc[1203, "team"] = np.nan
df2019.loc[1204, "team"] = np.nan
df2019.loc[1205, "team"] = np.nan

# Continuous attributes
print("Continuous attributes")
continuous_attributes = ["score", "opponentScore", "moneyLine", "opponentMoneyLine", "spread"]
for attribute in continuous_attributes:
    print_continuous_attribute_data(df2019, attribute)

print("")

# Categorical attributes
print("Categorical attributes")
continuous_attributes = ["team", "home/visitor", "opponent"]
for attribute in continuous_attributes:
    print_categorical_attribute_data(df2019, attribute)

# As I have many rows of data, I will drop the rows with missing data
# Row count went 1230 -> 1227
df2019 = df2019.dropna()

print_relevant_data(df2019)
