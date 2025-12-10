import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.optimizers import SGD


df = pd.read_csv('sunspot.txt', sep='\t', header=None, names=['year', 'sunspots'], dtype={'year': int, 'sunspots': int})


# def split_data(data):
#     sunspots = data['sunspots'].values
#
#     p = []
#     t = []
#
#     for i in range(2, len(sunspots)):
#         p.append([sunspots[i - 2], sunspots[i - 1]])
#         t.append(sunspots[i])
#
#     return np.array(p), t
#
#
# P, T = split_data(df)
#
#
# xfull = P
# xtrain = P[:200, :]
# xtest = P[200:, :]
# yfull = np.array(T)
# ytrain = np.array(T[:200])
# ytest = np.array(T[200:])
#
# model = tf.keras.Sequential([
#     tf.keras.Input(shape=(2,)),
#     tf.keras.layers.Dense(1, activation='linear')
# ])
#
# error_goal = 310
# max_epochs = 1000
# learning_rate = 0.0001
#
# model.compile(optimizer=SGD(learning_rate=learning_rate), loss='mean_squared_error')
#
# for epoch in range(1, max_epochs + 1):
#     model.fit(xtrain, ytrain, epochs=1, verbose=0)
#
#     Y_pred_train = model.predict(xtrain, verbose=0).flatten()
#     MSE_train = np.mean((ytrain - Y_pred_train) ** 2)
#
#     Y_pred_full = model.predict(xfull, verbose=0).flatten()
#     MSE_full = np.mean((yfull - Y_pred_full) ** 2)
#
#     if epoch % 100 == 0 or MSE_full <= error_goal:
#         W = model.layers[0].get_weights()[0].flatten()
#         b = model.layers[0].get_weights()[1][0]
#         print(f"Epoch {epoch}: Treniravimo MSE={MSE_train:.2f}, "
#               f"Bendras MSE={MSE_full:.2f}, w={W}, b={b:.5f}")
#
#     if MSE_full <= error_goal:
#         print(f"\nMokymas sustabdytas: MSE_full={MSE_full:.2f} ≤ {error_goal}")
#         break
#
# W = model.layers[0].get_weights()[0].flatten()
# b = model.layers[0].get_weights()[1][0]
#
# # Treniravimo metrikos
# Y_pred_train = model.predict(xtrain, verbose=0).flatten()
# MSE_train = np.mean((ytrain - Y_pred_train) ** 2)
# MAD_train = np.median(np.abs(ytrain - Y_pred_train))
#
# # Testavimo metrikos
# Y_pred_test = model.predict(xtest, verbose=0).flatten()
# MSE_test = np.mean((ytest - Y_pred_test) ** 2)
# MAD_test = np.median(np.abs(ytest - Y_pred_test))
#
# print(f"\nGalutiniai svoriai: w={W}, b={b:.5f}")
# print(f"Treniravimo MSE = {MSE_train:.2f}, MAD = {MAD_train:.2f}")
# print(f"Testavimo  MSE = {MSE_test:.2f}, MAD = {MAD_test:.2f}")
# print(f"Bendras    MSE = {MSE_full:.2f}, "
#       f"MAD = {np.median(np.abs(yfull - Y_pred_full)):.2f}")


def split_data(data):
    sunspots = data['sunspots'].values

    p = []
    t = []

    for i in range(10, len(sunspots)):
        p.append([sunspots[i - 10], sunspots[i - 9], sunspots[i - 8], sunspots[i - 7], sunspots[i - 6], sunspots[i - 5], sunspots[i - 4], sunspots[i - 3], sunspots[i - 2], sunspots[i - 1]])
        t.append(sunspots[i])

    return np.array(p), t


P, T = split_data(df)


xfull = P
xtrain = P[:200, :]
xtest = P[200:, :]
yfull = np.array(T)
ytrain = np.array(T[:200])
ytest = np.array(T[200:])

model = tf.keras.Sequential([
    tf.keras.Input(shape=(10,)),
    tf.keras.layers.Dense(1, activation='linear')
])

error_goal = 200
max_epochs = 1000
learning_rate = 0.00001

model.compile(optimizer=SGD(learning_rate=learning_rate), loss='mean_squared_error')

for epoch in range(1, max_epochs + 1):
    model.fit(xtrain, ytrain, epochs=1, verbose=0)

    Y_pred_train = model.predict(xtrain, verbose=0).flatten()
    MSE_train = np.mean((ytrain - Y_pred_train) ** 2)

    Y_pred_full = model.predict(xfull, verbose=0).flatten()
    MSE_full = np.mean((yfull - Y_pred_full) ** 2)

    if epoch % 100 == 0 or MSE_full <= error_goal:
        W = model.layers[0].get_weights()[0].flatten()
        b = model.layers[0].get_weights()[1][0]
        print(f"Epoch {epoch}: Treniravimo MSE={MSE_train:.2f}, "
              f"Bendras MSE={MSE_full:.2f}, w={W}, b={b:.5f}")

    if MSE_full <= error_goal:
        print(f"\nMokymas sustabdytas: MSE_full={MSE_full:.2f} ≤ {error_goal}")
        break

W = model.layers[0].get_weights()[0].flatten()
b = model.layers[0].get_weights()[1][0]

# Treniravimo metrikos
Y_pred_train = model.predict(xtrain, verbose=0).flatten()
MSE_train = np.mean((ytrain - Y_pred_train) ** 2)
MAD_train = np.median(np.abs(ytrain - Y_pred_train))

# Testavimo metrikos
Y_pred_test = model.predict(xtest, verbose=0).flatten()
MSE_test = np.mean((ytest - Y_pred_test) ** 2)
MAD_test = np.median(np.abs(ytest - Y_pred_test))

print(f"\nGalutiniai svoriai: w={W}, b={b:.5f}")
print(f"Treniravimo MSE = {MSE_train:.2f}, MAD = {MAD_train:.2f}")
print(f"Testavimo  MSE = {MSE_test:.2f}, MAD = {MAD_test:.2f}")
print(f"Bendras    MSE = {MSE_full:.2f}, "
      f"MAD = {np.median(np.abs(yfull - Y_pred_full)):.2f}")

