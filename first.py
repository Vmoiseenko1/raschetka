import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import metrics
from math import sqrt
import openpyxl
from openpyxl import load_workbook


df = pd.read_excel('C:/Users/USA_GAS.xls')
FORMAT = ['Year', 'Value']
df_selected = df[FORMAT] # датафрэйм для добычи газа в США с 1971 по 1990
#print(df_selected)


def plot_df(df_selected, x, y, title="", style='bmh'):
    with plt.style.context(style):
        df_selected.plot(x, y, figsize=(16, 5))
        plt.gca().set(title=title)
        plt.show()
plot_df(df_selected, "Year", "Value", title='Monthly gas production (billion cubic feet) in USA from 1971 to 1990')

my_columns = list(df_selected.columns)
my_columns[0] = 'Gas'
df_selected.columns = my_columns

t = list(range(1,len(df_selected)+1))
def t_2(x):
    s = []
    for i in list(range(1, len(x)+1)):
        s.append(i**2)
    return s

t = pd.DataFrame(t)
t_sq = pd.DataFrame(t_2(df))

col_t = list(t.columns)
col_t_sq = list(t_sq.columns)

col_t[0] = 't'
col_t_sq[0] = 't^2'

t.columns = col_t
t_sq.columns = col_t_sq

t.set_index(df_selected.index, inplace = True)
t_sq.set_index(df_selected.index, inplace = True)

df = pd.concat([df_selected, t, t_sq], axis=1) #выводим нормальную таблицу с t и t**2
print(df)

def tsplot(y, lags=None, figsize=(15,7), style='bmh'):
    """ACF и PACF"""
    if not isinstance(y, pd.Series): #Ставим формат Series для ряда
        y = pd.Series(y)

    with plt.style.context(style):

        fig = plt.figure(figsize=figsize)

        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax = ts_ax, color = 'black')
        ts_ax.set_title('Time Series Analysis Plots')

        sm.graphics.tsa.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5, color = 'red')
        sm.graphics.tsa.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5, method='ywm', color = 'red')

        plt.tight_layout()
        plt.show()
    return

np.random.seed(1)
tsplot(df['Value'], lags=30)


# Выделим сезонные переменные (январь, сентябрь и июль)
## January
y = []
for i in list(range(len(df.index))):
    if i % 12 == 0:
        y.append(1)
    else:
        y.append(0)
y = pd.DataFrame(y)
y.set_index(df.index, inplace = True)
January = y

col_January = list(January.columns)

col_January[0] = 'January'

January.columns = col_January

## September
x = []
for i in list(range(len(df.index))):
    if i % 12 == 8:
        x.append(1)
    else:
        x.append(0)
x = pd.DataFrame(x)
x.set_index(df.index, inplace = True)
September = x

col_September = list(September.columns)

col_September[0] = 'September'

September.columns = col_September

df = pd.concat([df, January, September], axis = 1)
print(df)
#print(df.January.head(20))

## Квадратичный тренд
y1 = df['Value']
x1 = df[['t', 't^2']]
estimator1 = LinearRegression()
estimator1.fit(x1, y1)
y_pred1 = estimator1.predict(x1)
#print(f'Slope 1 : {format(round(estimator1.coef_[0],2))}') #угол наклона перед t, a1
#print(f'Slope 2 : {format(round(estimator1.coef_[1],4))}') #угол наклона перед t^2, a2
#print(f'Intercept : {format(round(estimator1.intercept_,2))}') #константа
#print(f'R^2 : {round(estimator1.score(x1,y1),2)}')

## Линейный тренд
x2 = df[['t']]
y2 = df['Value']
estimator2 = LinearRegression()
estimator2.fit(x2, y2)
y_pred2 = estimator2.predict(x2)
#print(f'Slope : {format(round(estimator2.coef_[0],2))}') #a1
#print(f'Intercept : {format(round(estimator2.intercept_,2))}') #const
#print(f'R^2 : {round(estimator2.score(x2,y2),2)}')

## Квадратичный тренд с сезонными фиктивными переменными
x3 = df[['t', 't^2', 'January', 'September']]
y3 = df['Value']
estimator3 = LinearRegression()
estimator3.fit(x3, y3)
y_pred3 = estimator3.predict(x3)
print(f'Slope 1: {format(round(estimator3.coef_[0],2))}') #a1
print(f'Slope 2: {format(round(estimator3.coef_[1],4))}') #a2
print(f'Slope 3: {format(round(estimator3.coef_[2],4))}') # угол наклона перед January (a3)
print(f'Slope 4: {format(round(estimator3.coef_[2],4))}') # угол наклона перед September (a4)
print(f'Intercept : {format(round(estimator3.intercept_,2))}') # константа
print(f'R^2 : {round(estimator3.score(x3, y3),2)}')

## Линейный тренд с сезонными фиктивными переменными
x4 = df[['t', 'January', 'September']]
y4 = df['Value']
estimator4 = LinearRegression()
estimator4.fit(x4, y4)
y_pred4 = estimator4.predict(x4)
print(f'Slope 1 : {format(round(estimator4.coef_[0],2))}') #a1
print(f'Slope 2 : {format(round(estimator4.coef_[2],4))}') #a2
print(f'Slope 3 : {format(round(estimator4.coef_[2],4))}') #a3
print(f'Intercept: {format(round(estimator4.intercept_,2))}') #const
print(f'R^2 : {round(estimator4.score(x4, y4),2)}')

## Стат значимость моделей
model_s = smf.ols('Value ~ t + t^2 + January + September', data=df)
res1 = model_s.fit()
print(res1.summary())

## Графики с трендами
# Линейный тренд
#axs[0,0].plot(y_pred2, color='red')
#axs[0,0].plot(list(y2), color='blue')
#axs[0,0].set_title('Linear trend')
#axs[0,0].set_xlabel('Year')
#axs[0,0].set_ylabel('Monthly Gas Production in USA')


















