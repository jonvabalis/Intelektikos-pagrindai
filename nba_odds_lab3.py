import numpy as np
import pandas as pd
from keras.src.losses import sparse_categorical_crossentropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, mean_squared_error, accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import SGD
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.optimizers
from tensorflow.keras.optimizers import SGD

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
df2019['home/visitor'] = df2019['home/visitor'].map({'@': -1}).fillna(-2).astype(int)
le_points = LabelEncoder()
df2019['pointCategory'] = le_points.fit_transform(df2019['pointCategory'])

df2019y = df2019['pointCategory']
df2019x = df2019.drop(columns=['pointCategory'])

# print(df2019x.dtypes)
#
# print(df2019.isna().sum())

acc_scores = []
f1_scores = []

layers = (16, 8)
learning_rate = 0.001
activation = "linear"
epochs = 100

kf = KFold(n_splits=10, shuffle=True, random_state=42)
i = 0

for train_idx, test_idx in kf.split(df2019x):
    i = i + 1
    X_train, X_test = df2019x.iloc[train_idx], df2019x.iloc[test_idx]
    y_train, y_test = df2019y.iloc[train_idx], df2019y.iloc[test_idx]

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(df2019x.shape[1],)))
    for n in layers:
        model.add(tf.keras.layers.Dense(n, activation="relu"))
    model.add(tf.keras.layers.Dense(4, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(X_train, y_train,
              validation_split=0.1,
              epochs=epochs,
              batch_size=32,
              verbose=0)

    y_pred_prob = model.predict(X_test, verbose=0)
    # Get the highest chance of class probability
    y_pred = np.argmax(y_pred_prob, axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Fold {i} - Accuracy: {acc:.4f}, F1: {f1:.4f}")

    acc_scores.append(acc)
    f1_scores.append(f1)

print("\nAverage Accuracy:", np.mean(acc_scores))
print("Average F1 Score:", np.mean(f1_scores))
