import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
from scipy import stats
from collections import OrderedDict
from openpyxl import Workbook, load_workbook

names = ['kazan', 'yakutsk', 'spb', 'altay', 'dagestan', 'kostroma', 'osetiya', 'chechnya', 'new']

nums = 25
agg = list(range(7,18))
#agg = [12]

files = [x+ '.csv' for x in names ]
df = pd.DataFrame()
for elem in files:
    df2 = pd.read_csv(elem, skiprows = 1, header = None)
    ind = []
    for i in df2[0].tolist():
        ind.append(i +elem[:2])
    df2[0] = ind
    df = df.append(df2, ignore_index=True)
print(df)

x_all_listlog = []
M_listlog = []
sigma_listlog = []
coef_var_listlog = []
r_listlog = []
Rym_listlog = []
sigma_R_listlog = []

def my_corrcoef1(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x)
    std_y = np.std(y)
    n = len(x)
    return ((x - mean_x) * (y - mean_y)).sum() / n / (std_x * std_y)

genderr = 'g'
for ages in agg:
    x_all = []
    y_all = []
    for index, rows in df.iterrows():
        tag = df.iat[index, 0]
        curra = 0

        if len(tag) == 5:
            curra = int(tag[0])
        else:
            curra = int(tag[0:2])
        if tag[-3] == 'h' and tag[-4]==genderr and curra == ages :
            r = rows.tolist()
            r = r[1:]
            x_all.append(r)
        if tag[-3] == 'w' and tag[-4]==genderr and curra == ages :
            r = rows.tolist()
            r = r[1:]
            y_all.append(r)
        #    for ii in range(len(r)):
        #        if not math.isnan(r[ii]):   #GET AGE AS X
        #            x_all.append(curra)

    x_all = np.array(x_all)
    y_all = np.array(y_all)
    x_all = x_all[np.logical_not(np.isnan(x_all))]
    y_all = y_all[np.logical_not(np.isnan(y_all))]
    x_all = x_all.reshape(-1, 1)

    model = LinearRegression()

    model.fit(x_all, y_all)

    min = int(int(x_all.mean()) - 2.5 * int(x_all.std()))
    max = int(int(x_all.mean()) + 2.5 * int(x_all.std()))
    x_standart = list(range(min, max))
    x_standart = [x_standart[x:x +1] for x in range(0, len(x_standart), 1)]

    dependency_of_sigma_list = ['']* len(x_standart)
    for idx, xs in enumerate(x_standart):
        if xs[0] < (x_all.mean() - 2 * int(x_all.std())):
            dependency_of_sigma_list[idx] = 'Меньше 2 сигм'
        elif xs[0] < (x_all.mean() - 1 * int(x_all.std())):
            dependency_of_sigma_list[idx] = 'От -2 до -2 сигм'
        elif xs[0] < (x_all.mean() + 1 * int(x_all.std())):
            dependency_of_sigma_list[idx] = 'От -1 до +1 сигм'
        elif xs[0] < (x_all.mean() + 2 * int(x_all.std())):
            dependency_of_sigma_list[idx] = 'От +1 до +2 сигм'
        else:
            dependency_of_sigma_list[idx] = 'Больше 2 сигм'


    mean_mas = list(model.predict(np.array(x_standart)))


    #x_standart = np.array(x_standart, dtype="float")
    #x = x_standart.flat
    #mean_mas = np.array(mean_mas, dtype="float")
    x_standart = [val[0]  for val in x_standart]
    x_standart = np.array(x_standart)
    mean_mas = np.array(mean_mas)
    x_standart = x_standart.reshape(-1,1)
    r_coef = math.sqrt(model.score(x_all, y_all))
    print(x_standart, mean_mas)
    print(r_coef)

    R = 0
    print('M = ', x_all.mean(), y_all.mean())
    print('σ = ', x_all.std(), y_all.std())

    R = r_coef * y_all.std()/x_all.std()
    print(R)
    print(len(x_all), len(y_all))
    print(x_all.mean(), y_all.mean())

    sigm_R = y_all.std() * math.sqrt(1- r_coef*r_coef)

    less_2_mas = mean_mas - 2*sigm_R - 0.01
    less_2_mas = np.around(less_2_mas, decimals= 2)

    filler_mas = ['--'] * len(mean_mas)
    filler_mas1 = filler_mas
    filler_mas2 = filler_mas
    filler_mas3 = filler_mas
    from_2_to_1_low_mas = mean_mas - 2 * sigm_R
    from_2_to_1_low_mas = np.around(from_2_to_1_low_mas, decimals=2)

    from_2_to_1_high_mas = mean_mas - 1 * sigm_R - 0.01
    from_2_to_1_high_mas = np.around(from_2_to_1_high_mas, decimals=2)

    from_1_to_1_low_mas = mean_mas - 1 * sigm_R
    from_1_to_1_low_mas = np.around(from_1_to_1_low_mas, decimals=2)

    from_1_to_1_high_mas = mean_mas + 1 * sigm_R - 0.01
    from_1_to_1_high_mas = np.around(from_1_to_1_high_mas, decimals=2)

    from_1_to_2_low_mas = mean_mas + 1 * sigm_R
    from_1_to_2_low_mas = np.around(from_1_to_2_low_mas, decimals=2)

    from_1_to_2_high_mas = mean_mas + 2 * sigm_R
    from_1_to_2_high_mas = np.around(from_1_to_2_high_mas, decimals=2)

    more_2_mas = mean_mas + 2*sigm_R + 0.01
    more_2_mas = np.around(more_2_mas, decimals=2)



    #more_mas = mean_mas + 1 * y_all.std()/2
    #more_mas = np.around(more_mas, decimals=2)
    #mean_mas = np.around(mean_mas, decimals=2)
    l = ['Ср. арифм (M)', 'Сигма (σ)', 'Част.сигма (σR)', 'Коэф. регр. (Ry/x)']


    dataset = pd.DataFrame(OrderedDict({'Границы сигмальных отклонений': dependency_of_sigma_list,'Рост ': list(range(min, max)), 'до -2σR': less_2_mas,
                            'от -2σR': from_2_to_1_low_mas,'--': filler_mas1,'до -1σR': from_2_to_1_high_mas,
                            'от -1σR': from_1_to_1_low_mas,'-- ': filler_mas2,'до +1σR': from_1_to_1_high_mas,
                            'от +1σR': from_1_to_2_low_mas,'--   ': filler_mas3,'до +2σR': from_1_to_2_high_mas,
                            'от +2σR': more_2_mas }))
    dataset.reset_index(drop=True)
    geno = 'BOYS' if genderr == 'b' else 'GIRLS'
    name = geno +' '+ str(ages) + '.xlsx'
    dataset.to_excel(name)

    workbook = load_workbook(filename=name)
    sheet = workbook.active
    n1 =len(dependency_of_sigma_list)
    y1 = 'B'+ str(n1+2)
    y2 = 'B' + str(n1 + 3)
    y3 = 'B' + str(n1 + 4)
    y4 = 'B' + str(n1 + 5)
    sheet[y1] = 'Ср. арифм (M)'
    sheet[y2] = 'Сигма (σ)'
    sheet[y3] = 'Част.сигма (σR)'
    sheet[y4] = 'Коэф. регр. (Ry/x)'

    y11 = 'C'+ str(n1+2)
    y12 = 'C' + str(n1 + 3)
    sheet[y11] = np.round(x_all.mean(), decimals=2)
    sheet[y12] = np.round(x_all.std(), decimals=2)

    y21 = 'I'+ str(n1+2)
    y22 = 'I' + str(n1 + 3)
    y23 = 'I' + str(n1 + 4)
    y24 = 'I' + str(n1 + 5)
    sheet[y21] = np.round(y_all.mean(), decimals=2)
    sheet[y22] = np.round(y_all.std(), decimals=2)
    sheet[y23] = np.round(sigm_R, decimals=2)
    sheet[y24] = np.round(R, decimals=2)

    sheet.column_dimensions['B'].width = 17
    for col in sheet.columns:
        for cell in col:
            # openpyxl styles aren't mutable,
            # so you have to create a copy of the style, modify the copy, then set it back
            alignment_obj = cell.alignment.copy(horizontal='center', vertical='center')
            cell.alignment = alignment_obj
    workbook.save(filename=name)

    #TO GET X DATA => CHANGE ALL Y TO X
    x_all_listlog.append(len(y_all))
    M_mistake = np.round(y_all.std()* y_all.std()/len(y_all), decimals= 2)
    s_temp = str((np.round(y_all.mean(), decimals=2))) + ' ± ' + str(M_mistake)
    M_listlog.append(s_temp)
    sigma_listlog.append(np.round(y_all.std(), decimals=2))
    coef_var_listlog.append(np.round(y_all.std()/y_all.mean()*100, decimals= 2))
    r_listlog.append(str(np.round(r_coef,decimals=2)) + ' ± '+ str(M_mistake))
    Rym_listlog.append(np.round(R, decimals=2))
    sigma_R_listlog.append(np.round(sigm_R, decimals=2))

print(len(agg), len(coef_var_listlog))

df2 = pd.DataFrame({'Возраст': agg, 'N': x_all_listlog, 'М±m': M_listlog, 'σ': sigma_listlog, 'V': coef_var_listlog, 'r±m':r_listlog, 'Ry/m': Rym_listlog, '±σR': sigma_R_listlog })
df2.reset_index(drop=True)
name = geno + ' STANDARTS'+ '.xlsx'
df2.to_excel(name)


x_regs = []
y_regs = []
iter = 0
gen = 'b'
for region in names:
    x_regs.append([])
    y_regs.append([])
    for index, rows in df.iterrows():
        curra=0
        tag = df.iat[index, 0]
        if len(tag)==5:
            curra = int(tag[0])
        else:
            curra = int(tag[0:2])
        for age in agg:
            #if tag[-2:-1] == region[0:1] and gen == tag[-4] and str(age) == str(curra):
            if tag[-2:-1] == region[0:1] and gen == tag[-4] and str(age) == str(curra):
                if tag[-3] == 'h':
                    r = rows.tolist()
                    r = r[1:]
                    y_regs[iter].append(r)
                    for ii in range(len(r)):
                        if not math.isnan(r[ii]):
                            x_regs[iter].append(curra)

                #if tag[-3] == 'h':
                #    r = rows.tolist()
                #    r = r[1:]
                #    x_regs[iter].append(r)
    iter+=1


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

    #ax.scatter(x_regs[it], y_regs[it], color = "red")
    #ax.plot(x_regs[it], model_reg.predict(x_regs[it]), color="green")
    #ax.set_title(names[it])
    #ax.set_ylim([100, 190])
    #ax.set_xlim([5, 18])

    print('________', names[it], 'BOYS,','age to height' '________')
    print('means: ', y_regs[it].mean(), x_regs[it].mean())
    print('standart deviance:', y_regs[it].std())
    print('coefficient of determination:', math.sqrt(r_sq))
    print('sigma * determination coef:', math.sqrt(r_sq)*y_regs[it].std())
    print('coefficients:', model_reg.coef_[0])

    print()



type_gen = 'BOYS' if genderr == 'b' else 'GIRLS'

rmse = mean_squared_error(y_all, model.predict(x_all))
r_sq = model.score(x_all, y_all)
print('________', 'ALL', type_gen, 'height to weight' '________')
print('RMSE based on full dataset: ', rmse)
print('means: ', y_all.mean(), x_all.mean())
print('standart deviance:', y_all.std())
print('coefficient of determination:', math.sqrt(r_sq))
print('sigma * determination coef:', math.sqrt(r_sq) * y_all.std())
print('coefficients:', model.coef_[0])




type_gen2 = 'мальчиков' if genderr == 'b' else 'девочек'

plt.scatter(x_all, y_all, color = "red")
plt.plot(x_all, model.predict(x_all), color = "green")
title = "Рост и вес"+ type_gen2+ " (Вся выборка)"
plt.title(title)
plt.xlabel("Рост")
plt.ylabel("Вес")
plt.show()
