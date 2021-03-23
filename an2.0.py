import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math

names = ['kazan', 'yakutsk', 'spb', 'altay', 'dagestan', 'kostroma', 'osetiya', 'chechnya']
ages = list(range(7,18))

files = [x+ '.csv' for x in names]
df = pd.DataFrame()
for elem in files:
    df2 = pd.read_csv(elem, skiprows = 1, header = None)
    ind = []
    for i in df2[0].tolist():
        ind.append(i +elem[:2])
    df2[0] = ind
    df = df.append(df2, ignore_index=True)
print(df)

for n in names:
    for gender in ['b', 'g']:
        for index, rows in df.iterrows():
            if len(tag) == 5:
                curra = int(tag[0])
            else:
                curra = int(tag[0:1])





#fig = plt.figure()
#fig.subplots_adjust(hspace=0.4, wspace=0.4)
for it in range(len(names)):
    #ax = fig.add_subplot(2, 4, it+1)

    x_regs[it] = np.array(x_regs[it])
    y_regs[it] = np.array(y_regs[it])
    x_regs[it] = x_regs[it][np.logical_not(np.isnan(x_regs[it]))]
    y_regs[it] = y_regs[it][np.logical_not(np.isnan(y_regs[it]))]
    x_regs[it] = x_regs[it].reshape(((-1, 1)))

    model_reg = LinearRegression()
    model_reg.fit(x_regs[it], y_regs[it])
    r_sq = model_reg.score(x_regs[it], y_regs[it])


    #ax.plot(x_regs[it], model_reg.predict(x_regs[it]), color="green")

    print('________', names[it], 'GIRLS,','height to weight' '________')
    print('means: ', y_regs[it].mean(), x_regs[it].mean())
    print('standart deviance:', y_regs[it].std())
    print('coefficient of determination:', math.sqrt(r_sq))
    print('sigma * determination coef:', math.sqrt(r_sq)*y_regs[it].std())
    print('coefficients:', model_reg.coef_[0])

    print()



print(x_regs)
print(y_regs)
print(x_all)
print(y_all)

#r_sq = model.score(x_all, y_all)
#print('coefficient of determination:', r_sq)
#print(x_all.std(), y_all.std())
#print('coefficients:', model.coef_)
#print('means: ',x_all.mean(), y_all.mean())






plt.scatter(x_all, y_all, color = "red")
plt.plot(x_all, model.predict(x_all), color = "green")
plt.title("Рост и Вес детей (вся выборка)")
plt.xlabel("Рост")
plt.ylabel("Вес")
plt.show()
