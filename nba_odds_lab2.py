import pandas as pd

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

df2019["won"] = df2019[['score', 'opponentScore']].max(axis=1)
df2019["pointCategory"] = df2019["won"].apply(
    lambda x: "very low" if x <= 85
    else ("low" if x < 100
          else ("average" if x < 130
                else "more"))
)

# I've decided to use 7 columns only: 'team', 'home/visitor', 'opponent',
# 'moneyLine', 'opponentMoneyLine', 'spread' and `pointCategory`
df2019 = df2019.drop(columns=['date', 'season', 'total', 'secondHalfTotal', 'score', 'opponentScore', 'won'])

print_relevant_data(df2019)
