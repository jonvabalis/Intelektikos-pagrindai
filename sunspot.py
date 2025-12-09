import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('sunspot.txt', sep='\t', header=None, names=['year', 'sunspots'], dtype={'year': int, 'sunspots': int})
print(df)


plt.figure(figsize=(10, 5))
plt.plot(df['year'], df['sunspots'])
plt.xlabel('Year')
plt.ylabel('Sunspots')
plt.title('Saulės dėmių skaičius metų bėgyje')


plt.show()

def split_data(data):
    sunspots = data['sunspots'].values

    p = []
    t = []

    for i in range(2, len(sunspots)):
        p.append([sunspots[i - 2], sunspots[i - 1]])
        t.append(sunspots[i])

    return np.array(p), t


P, T = split_data(df)


x = P[:, 0]
y = P[:, 1]
z = T

fig = plt.figure(figsize=(10, 7))
d3d = fig.add_subplot(111, projection='3d')

d3d.scatter(x, y, z)

d3d.set_xlabel("t-2 metų saulės dėmės")
d3d.set_ylabel("t-1 metų saulės dėmės")
d3d.set_zlabel("t metų saulės dėmės")
d3d.set_title("3D duomenų diagrama")

plt.show()

Pt = P[:200, :]
Pr = P[200:, :]
Tt = T[:200]
Tr = T[200:]

lr = LinearRegression()
lr.fit(Pt, Tt)

print("Coeficients:", lr.coef_)
print("Bias:", lr.intercept_)

pred_train = lr.predict(Pt)
pred_test = lr.predict(Pr)

plt.figure(figsize=(10,5))
plt.plot(df['year'][:200], Tt, label="Tikros reikšmės")
plt.plot(df['year'][:200], pred_train, label="Prognozuota")
plt.xlabel("Metai")
plt.ylabel("Sunspots")
plt.title("Modelio prognozė (mokymo duomenys)")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(df['year'][202:], Tr, label="Tikros reikšmės")
plt.plot(df['year'][202:], pred_test, label="Prognozuota")
plt.xlabel("Metai")
plt.ylabel("Sunspots")
plt.title("Modelio prognozė (testo duomenys)")
plt.legend()
plt.show()

e_train = np.array(Tt) - pred_train
e_test = np.array(Tr) - pred_test

plt.figure(figsize=(10,5))
plt.plot(df['year'][:200], e_train, label="Vektoriaus reikšmė su treniravimo duomenim")
plt.plot(df['year'][202:], e_test, label="Vektoriaus reikšmė su testavimo duomenim")
plt.xlabel("Metai")
plt.ylabel("Prognozės vektoriaus reikšmė")
plt.title("Prognozės vektoriaus reikšmės")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(e_train, bins=10, alpha=0.7)
plt.title("Treniravimo prognozės klaidos histograma")
plt.xlabel("Klaida")
plt.ylabel("Dažnis")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(e_test, bins=10, alpha=0.7)
plt.title("Testavimo prognozės klaidos histograma")
plt.xlabel("Klaida")
plt.ylabel("Dažnis")
plt.legend()
plt.show()

mse_train = mean_squared_error(Tt, pred_train)
mse_test = mean_squared_error(Tr, pred_test)
print("MSE train:", mse_train)
print("MSE test:", mse_test)

mad_train = np.median(np.abs(np.array(Tt) - pred_train))
mad_test = np.median(np.abs(np.array(Tr) - pred_test))
print("MAD train:", mad_train)
print("MAD test", mad_test)


