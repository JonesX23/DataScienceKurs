import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


x = np.array([84,1323,282,957,1386,810,396,474,501,660,1260,1005,1110,1290]) #Studying time
y = np.array([44,97,30,51,95,51,44,41,21,40,90,83,61,92]) #result (punkte)


def mse(y_pred,y):
    return (y_pred-y)**2

def sse(y_pred,y):
    return np.mean((y_pred-y)**2)

def rmse(y_pred,y):
    return np.sqrt(np.mean((y_pred-y)**2))



#reg = LinearRegression().fit(x,y)

x_axis = np.linspace(0,2000)
plt.figure(figsize=(9,7))
plt.scatter(x,y,c="#1acc94")
#for w in np.linspace(0.05,0.09,5):
plt.plot(x_axis,x_axis*0.068892845,label="$w$={}".format(0.068892845),ls="--")
plt.title("Visualisierung verschiedener $w$-Werte")
plt.xlabel("Vorbereitungszeit auf Klausur in Minuten")
plt.ylabel("Klausurergebnis")
plt.legend()
plt.show()
