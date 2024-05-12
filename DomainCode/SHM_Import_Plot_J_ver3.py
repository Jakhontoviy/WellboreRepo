#import file capillary_test_1.xls as pandas dataframe from subfoler named 06_Капиллярки
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, MeanShift, OPTICS
from scipy.optimize import curve_fit
df = pd.read_excel(r'Files\Capillaty_example_1.xlsx', header=0)

#pc, bar
#sw, v/v

def func(x, a, b, c):
    return a * np.power(x, b) + c

pk = df.filter(regex='^Pk').values
sw = df.filter(regex='^Sw').values

num_columns = pk.shape[1]
num_rows = pk.shape[0]

df_new = pd.DataFrame()
dict = ['DEPTH', 'SampleID', 'Perm', 'Poro']

for name in dict:
    if name in df.columns:
        array_all = np.tile(df[name].values[:, None], num_columns).flatten()
        df_new[name] = array_all

df_new['PC_lab'] = pk.flatten()
df_new['SW'] = sw.flatten()

#capillary pressure input parameters
sigma_lab = 72 #IFT, dyn/cm
sigma_res = 30 #IFT, dyn/cm
costh_lab = 1
costh_res = 0.87

#расчет J функции в системе PC_IFT (с учетом лабораторных условий)
#коэффициент 0.31832 используется для случая PC[bar], Sw[v/v], Por[v/v], Perm[mD]
df_new['J_ift'] = 0.31832*(df_new['PC_lab']/(sigma_lab*costh_lab))*((df_new['Perm']/df_new['Poro'])**0.5)

#расчет пластового давления по J функции (с учетом лабораторных условий)
df_new['PC_res'] = (df_new['J_ift']*sigma_res*costh_res)/(0.31832*((df_new['Perm']/df_new['Poro'])**0.5))

np_new = df_new['DEPTH'].unique()
a = np.array([])
b = np.array([])
c = np.array([])
for iter in range(np_new.size):
    xdata = df_new['J_ift'][df_new['DEPTH'] == np_new[iter]]
    ydata = df_new['SW'][df_new['DEPTH'] == np_new[iter]]
    popt, pcov = curve_fit(func, xdata, ydata, p0=[0.01, 0.01, 0.01], maxfev=10000)
    a1=np.full(8, popt[0])
    b1=np.full(8, popt[1])
    c1=np.full(8, popt[2])
    a = np.append(a, a1)
    b = np.append(b, b1)
    c = np.append(c, c1)

#тут можно прописать условия на обрезание данных если необходимо
df_new['a'] = np.where(df_new['Poro']>0.2, a, np.NaN)
df_new['b'] = np.where(df_new['Poro']>0.2, b, np.NaN)
df_new['c'] = np.where(df_new['Poro']>0.2, c, np.NaN)

#среднее арифметическое для построения графиков и вычислений средних
df_new['a_avg'] = a_mean = df_new['a'].mean()
df_new['b_avg'] = b_mean =df_new['b'].mean()
df_new['c_avg'] = c_mean = df_new['c'].mean()

#расчет водонасыщенности со средними параметрами a,b,c (a,b,Swi) в фунукции J
df_new['SW_synth'] = df_new['a_avg']*(df_new['J_ift']**df_new['b_avg'])+df_new['c_avg']

height = 5
rhob_wat = 1
rhob_hc = 0.8
por_log = 20
perm_log = 130
pc_res_h = 0.001 * height * 9.81 * (rhob_wat - rhob_hc)
j_res = 0.031415 * (pc_res_h / (sigma_res * costh_res)) * (np.sqrt(perm_log / por_log))
sw_shm = a_mean * (j_res**b_mean) + c_mean
print(f'Зависимость для расчета SW: sw_shm = {round(a_mean,5)} * (j_res**{round(b_mean,4)}) + {round(c_mean,5)}')

#кластеризация данных по пористости/проницаемости
numb_clust = 8
X = df_new[['Poro', 'Perm']].values
groups = KMeans(n_clusters=numb_clust, random_state=0).fit(X)
df_new['group'] = groups.labels_

print(df_new)

#графический вывод всех результатов
import plotly.express as px
fig_abc = px.scatter_matrix(df_new, dimensions=["Poro", 'Perm', "a", "b", "c"], color="group")
fig_por_perm = px.scatter(x=df_new['Poro'], y=df_new['Perm'], color=df_new['group'], log_y=True, color_continuous_scale=px.colors.sequential.Turbo)
fig_pc_lab = px.line(df_new, x='SW', y='PC_lab', color='group', line_group='DEPTH', log_y=True)
fig_pc_lab_synth = px.line(df_new, x='SW_synth', y='PC_lab', color='group', line_group='DEPTH', log_y=True)
fig_j = px.line(df_new, x='SW', y='J_ift', color='DEPTH', line_group='DEPTH', log_y=True)
fig_pc_res = px.line(df_new, x='SW', y='PC_res', color='DEPTH', log_y=True)
fig_abc_parallel = px.parallel_coordinates(df_new, dimensions=["Poro", 'Perm', "a", "b", "c"], color="group", color_continuous_midpoint=2, labels='group',)
fig_abc_parallel_categ = px.parallel_categories(df_new, dimensions=["Poro", 'Perm', "a", "b", "c"], color="group")

#fig_por_perm.show() #зависимость проницаемости от пористости по исходным данным
#fig_abc.show() #вывод матрицы графиков a, b, c, Poro, Perm
#fig_abc_parallel_categ.show() #вывод матрицы графиков a, b, c, Poro, Perm
#fig_abc_parallel_categ.show()
fig_pc_lab.show()
#fig_pc_lab_synth.show() #зависимость PC_lab от SW

#fig_j.show()
fig_pc_res.show()


#вывод всех результатов в эксель
#df_new.to_excel('SHM.xlsx', sheet_name="Report")