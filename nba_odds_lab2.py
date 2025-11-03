import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt

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
    lambda x: "very low" if x <= 90
    else ("low" if x < 105
          else ("average" if x < 130
                else "high"))
)

# I've decided to use 7 columns only: 'team', 'home/visitor', 'opponent',
# 'moneyLine', 'opponentMoneyLine', 'spread' and 'pointCategory'
df2019 = df2019.drop(columns=['date', 'season', 'total', 'secondHalfTotal', 'score', 'opponentScore', 'won'])

all_teams = pd.concat([df2019['team'], df2019['opponent']])
le_team = LabelEncoder()
le_team.fit(all_teams)

df2019['team'] = le_team.transform(df2019['team'])
df2019['opponent'] = le_team.transform(df2019['opponent'])
for i, row in df2019.iterrows():
    if row["home/visitor"] == "@":
        df2019.at[i, 'home/visitor'] = -1
    else:
        df2019.at[i, 'home/visitor'] = -2

df_train = df2019.sample(frac=0.7, random_state=42)
df_test = df2019.drop(df_train.index)

df_train_result = df_train["pointCategory"]
df_train.drop(columns=["pointCategory"], inplace=True)

df_test_result = df_test["pointCategory"]
df_test.drop(columns=["pointCategory"], inplace=True)

model = DecisionTreeClassifier(random_state=42)
model.fit(df_train, df_train_result)

dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=df2019.drop(columns=['pointCategory']).columns,
    class_names=['very low', 'low', 'average', 'high'],
    filled=True,
    rounded=True,
    special_characters=True
)
graph = graphviz.Source(dot_data)
graph.render("decision_tree")


print("---------------------------------------")
accuracy = model.score(df_test, df_test_result)
y_pred = model.predict(df_test)
f1 = f1_score(df_test_result, y_pred, average='macro')
print(f"Tree accuracy: {accuracy * 100:.2f}%")
print(f"Tree f1 score: {f1 * 100:.2f}%")

cm = confusion_matrix(df_test_result, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

