import pandas as pd
import numpy as np

df_in = pd.read_excel('new boys.xlsx', index_col = 0)
df_in_2 = pd.read_excel('new girls.xlsx', index_col = 0)

df_in_tr = df_in.transpose()
print(df_in_tr.values)
df_in_tr2 = df_in_2.transpose()

#print(data.columns.ravel())
print(df_in_tr.dropna().values.tolist())

iter = 0
dict = {}
yo_list = ['6bw', '6bh','7bw', '7bh','8bw', '8bh','9bw', '9bh','10bw', '10bh','11bw', '11bh','12bw', '12bh','13bw', '13bh','14bw', '14bh','15bw', '15bh','16bw', '16bh','17bw', '17bh']
yo_list2 = ['6gw', '6gh','7gw', '7gh','8gw', '8gh','9gw', '9gh','10gw', '10gh','11gw', '11gh','12gw', '12gh','13gw', '13gh','14gw', '14gh','15gw', '15gh','16gw', '16gh','17gw', '17gh']

nd = df_in_tr.values.tolist()
nd2 = df_in_tr2.values.tolist()


#nd = nd[1:-1]
#nd = np.array(nd)
#print(nd.std())
for it in nd:
    if it[0] != 'Индекс кетле' and iter<= len(yo_list):
        dict[yo_list[iter]] = it[1:]
        iter+=1

iter=0
for item in nd2:
    if item[0] != 'Индекс кетле' and iter< len(yo_list2):
        dict[yo_list2[iter]] = item[1:]
        iter+=1

print(dict)
for i in dict.keys():
    for value in dict[i]:
        if len(i) ==3:
            curra = i[2]
        else:
            curra = i[3]
        if curra == 'w' and (value >110 or value<20):
            inde = dict[i].index(value)
            dict[i].remove(value)
            fr = i
            fr = fr[:-1]+'h'
            dict[fr].pop(inde)
        if curra == 'h' and (value >210 or value<100):
            inde = dict[i].index(value)
            dict[i].remove(value)
            fr = i
            fr = fr[:-1]+'w'
            dict[fr].pop(inde)


newdf = pd.DataFrame.from_dict(dict, orient= 'index')
print(newdf)
newdf.to_csv('new.csv')
