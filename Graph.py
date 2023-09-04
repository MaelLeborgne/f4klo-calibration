#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import Externe
from math import *
import matplotlib.pyplot as plt
import numpy as np
import statistics as st
import scipy as sp

import Classe
from Classe import Transit, Sensor, Spectrum
from main import *

# Graph storage directory
GSD = "../Graph_download/"

def Obs1(nObs, mode):
    '''
       Affiche une Observation numéro nObs
       Selon le mode choisi: temporel 'T' ou frequentiel 'F'
TEST:
    '''
    self = Transit.instance[nObs]# choix de l'objet étudié
    self.read_Tot()

    fig, ax = plt.subplots(layout="constrained")
    if mode == 'F':
         self.read_Freq()
         self.graph_Freq(ax)
    elif mode == 'T':
         self.read_Temp()
         self.graph_Temp(ax)
    fig.savefig('/home/mael/Bureau/StageMael2023/Graph_download/usefull/Sun.png',bbox_inches='tight',dpi=200)
    plt.show()

def Obs2(a):
    '''
       Comparaison des Transit n° a et b
TEST:
    '''
    fig, ax = plt.subplots(1,2,figsize=(15,15), layout="constrained")
    
    transit = Transit.instance[a] # choix de l'objet étudié
    transit.read_Tot() ### NE PAS OUBLIER
    transit.read_Temp()
    transit.read_Freq()
    transit.graph_Freq(ax[0])
    transit.graph_Temp(ax[1])
    
    #self = Transit.instance[b] # choix de l'objet étudié
    #Classe.Transit.read_Tot(self) ### NE PAS OUBLIER
    #self.graph_Freq(ax[1,0])
    #self.graph_Temp(ax[1,1])
    
    print('\n')
    plt.show()

def Obs9(LObs):
    """
       Affichage en 3*3 de la liste de nObs donnée
TEST:
    """
    fig, ax = plt.subplots(3,3,figsize=(15,15), layout="constrained")
    i=0
    j=0
    for nObs in LObs:
        print(i)
        self = Transit.instance[nObs]
        self.read_Tot() ### NE PAS OUBLIER
        self.read_Temp()
        self.graph_Temp(ax[i,j])
        i = i+1
        if i == 3:
            i = 0
            j = j+1
    
    print('\n')
    plt.show()

def Sensor4():
    '''
       Affiche les données influx de 4 varibales
TEST:
    '''
    fig, ax = plt.subplots(2,2,figsize=(15,15), layout="constrained")
    
    self = Sensor.fileList[3]
    self.read_Tot()
    self.graph(ax[0,0])
    
    self = Sensor.fileList[6]
    self.read_Tot()
    self.graph(ax[0,1])
    
    self = Sensor.fileList[15]
    self.read_Tot()
    self.graph(ax[1,0])
    
    self = Sensor.fileList[17]
    self.read_Tot()
    self.graph(ax[1,1])
    
    print('\n')
    plt.show()


def PF_read():
    """
       affiche tout les spectres
       Possibilité d'afficher moyenne, mediane, ...
       affichage possible en mode histograme (desité de point),
       afin d'y voir plus clair
TEST:
    """
    fig, ax = plt.subplots(figsize=(15,15))

    # read from 
    df_PF_f = pd.read_csv('PF4_dataFrame.csv',index_col=0) # freq at columns
    df_PF_n = df_PF_f.T # nObs at columns

    # freq to list of float
    freqScale = df_PF_f.columns.astype(float).to_numpy()
    minF = freqScale[0]
    maxF = freqScale[-1]

    # iter on spectrum and simple plot (PF_spec8.png)
    #df_PF_n = pd.DataFrame(df_PF_n.iloc[:,:8]) pour ne pas tout afficher
    for (label,content) in df_PF_n.items(): 
        ax.plot(freqScale,list(content),color='b')

#    # plot mediane
#    mediane = df_PF_f.median(axis=0) # by freq
#    ax.plot(freqScale, mediane, color='red', label='mediane') # Plot
#
#    # plot moyenne
#    moy = df_PF_f.mean() # by freq
#    ax.plot(freqScale, moy, color='orange', label='moyenne') # Plot
#
#    # plot mediane resampled
#    med_res = sp.signal.resample(medianeGlobale,60)
#    freq_new = np.linspace(freqScale[0], freqScale[-1], 60, endpoint=False)
#    ax.plot(freq_new, med_res, color='purple')

#    # plot all spec in density of point (PF_all)
#    # next 15 lines from https://matplotlib.org/stable/gallery/statistics/time_series_histogram.html#sphx-glr-gallery-statistics-time-series-histogram-py
#
#    # Now we will convert the multiple time series into a histogram.
#    # Linearly interpolate between the points in each time series
#    num_fine = 8191 # num of value in a spectrum
#    num_series = len(list(df_PF_n))
#    x = freqScale
#    Y = df_PF_f.to_numpy()
#    x_fine = np.linspace(x.min(), x.max(), num_fine)
#    y_fine = np.empty((num_series, num_fine), dtype=float)
#    for i in range(num_series):
#        y_fine[i, :] = np.interp(x_fine, x, Y[i, :])
#    y_fine = y_fine.flatten()
#    import numpy.matlib
#    x_fine = np.matlib.repmat(x_fine, num_series, 1).flatten()
#
#    # Plot (x, y) points in 2d histogram with linear colorscale
#    from copy import copy
#    cmap = copy(plt.cm.binary)
#    cmap.set_bad(cmap(0))
#    h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[400, 100])
#    pcm = ax.pcolormesh(xedges, yedges, h.T, cmap=cmap,
#                         vmax=1.5e2, rasterized=True)
#    fig.colorbar(pcm, ax=ax, label="# points", pad=0)
#
    # final setting for plot
    ax.grid(True)
    ax.ticklabel_format(useOffset=False)
    plt.xticks(np.arange(minF, maxF+0.00001, 0.2), fontsize='large')
    ax.set_xlabel('Fréquence (en MHz)',fontsize='xx-large')
    ax.set_ylabel('Intensité du signal',fontsize='xx-large')
    ax.set_ylim(0,0.0005)
    ax.legend()
    plt.show()
#    fig.savefig('/home/mael/Bureau/StageMael2023/Graph_download/usefull/FP_spec8.png',bbox_inches='tight',dpi=200)

def PF_exmin():
    '''
      filtre les spectres d'entrée "y"
      calcule le rapport y_filter/y
    '''

def PF_DC(fileName):
    """
       Test de lissage du spectre avec resample
       calcule le spectre médian et la moyenne de chaque spectre resamplé
TEST:
from main import *
#Write.PF_Spec(CarnetLabo,'PF1_dataFrame.csv') #en modifiant la fonction pour selectionner les données voulues
PF_DC('PF1_dataFrame')
    """
    fig, ax = plt.subplots(figsize=(15,15))

    # read all PF Spectrum (by freq in col)
    df_PF = pd.read_csv('PF_dataFrame.csv',index_col=0)

    # frequency as array of float
    freqScale = df_PF.columns.astype(float).to_numpy()
    minF = freqScale[0]
    maxF = freqScale[-1]
    # resample frequency
    freq_new = np.linspace(minF, maxF, 60, endpoint=False)

    df_PF_new = pd.DataFrame(index=freq_new)
    # Spectrum resampler
    for (label,content) in df_PF.T.items(): # by nObs in col
        spec_res = sp.signal.resample(list(content),60) # resample with scipy
        ax.plot(freq_new, spec_res, color='b') # plot
        mean = np.average(spec_res) # mean
        ax.plot([minF, maxF],[mean, mean], color='purple') # plot mean
        df_PF_new[label] = spec_res # storage by nObs in col

    df_PF_new = df_PF_new.T # by Freq in col
    df_PF_new["mean"] = df_PF_new.mean(1) # in new column write mean on nObs
    print(df_PF_new)

    # Mediane of rsampled
    medianeGlobale = df_PF_new.iloc[:,:-1].median(axis=0) # by freq
    ax.plot(freq_new, medianeGlobale, color='red', label='mediane') # Plot
    medianeGlobale.to_csv('spec_med.csv') # save median Spectrum

    # show
    ax.set_xlabel('Fréquence (en MHz)')
    ax.set_ylabel('Intensité du signal')
    plt.xticks(np.arange(minF, maxF+0.00001, 0.2))
    ax.ticklabel_format(useOffset=False)
    ax.grid(True)
    ax.legend()
    plt.show()
    
    # Save
    df_PF_new.to_csv('PF_DC_save.csv')

def PF_DC_Temp(fileName):
    '''
       take file of dataframe with all PF Spec
       plot mean by spectrum / mean Temperature
       fit linéraire pour prédiction hauteur de Spectre PF
TEST:
from main import *
#Graph.PF_DC(CarnetLabo,'PF1_dataFrame.csv')
Graph.PF_DC_Temp('PF1_dataFrame.csv')
    '''
    from Classe import Transit
    df_PF = pd.read_csv(fileName,index_col=0) # nObs in index
    nObsRange = list(df_PF.index.astype(int))

    # Import the Data of sensor on each Observation
    df_M_T = pd.DataFrame(index=['mean','ext']) # filled by loop
    for nObs in nObsRange:
        # test if exist and request influx if not
        transit = Transit.instance[nObs]
        transit.import_Sensor('ext','obj','temp','f4klo')
        # pick values
        temp = transit.sensorFrame.T['value']['ext']
        temp = np.average(temp)
        mean = df_PF.loc[nObs,:].mean() #on spec
        # add to DataFrame
        df_M_T[nObs] = [mean,temp] # new column

    # Prepare data
    df_M_T = df_M_T.T.sort_values(by='ext',ascending=True) # nObs in index
    Spec_M = df_M_T['mean'].to_numpy()
    Temp = df_M_T['ext'].to_numpy() # put nObs in index

    # fit with scikit learn
    # 1d-list to 2d-list
    T_2d = [[t] for t in list(Temp)]
    M_2d = [[m] for m in list(Spec_M)]

    # Fit Polynomiale
    (Y_pred_2d, [x0,x1,x2], mape, mse) = fit2D(T_2d,M_2d)

    # 2d-list to 1d-list
    Y_pred_1d = np.array([element[0] for element in Y_pred_2d])

    return (Temp, Spec_M, Y_pred_1d, [x0,x1,x2], mape, mse)

def PF_DC_Temp_plot1():
    '''
       Plot le(s) résultat de la fonction PF_DC_Temp
       Permet la séparation des différents points froids
TEST:
from main import *
#Graph.PF_DC()
Graph.PF_DC_Temp_plot1()
    '''
    #plot
    (Temp, Spec, Pred, coeff, mape, mse) = PF_DC_Temp('PF_dataFrame.csv')
    # figure
    fig, ax = plt.subplots(figsize=(15,15),layout='constrained')
    ax.plot(Temp, Spec, 'o')
    plt.plot(Temp, Pred,color='red',label='fit quad')
    plt.text(0.3, 0.9, fr'$y = {coeff[2]:.3}x^2 + {coeff[1]:.3}x + {coeff[0]:.3}$',
               transform=plt.gca().transAxes, fontsize='x-large')
    plt.text(0.3, 0.85, fr'$mape = {mape:.3}$',
               transform=plt.gca().transAxes, fontsize='x-large')
    plt.text(0.3, 0.8, fr'$mse = {mse:.3}$',
               transform=plt.gca().transAxes, fontsize='x-large')
    plt.xlabel("Température °C", fontsize='x-large')
    plt.ylabel("moyenne Spectre", fontsize='x-large')
    plt.legend()
    plt.show() 

def PF_DC_Temp_plot3():
    '''
       Plot le(s) résultat de la fonction PF_DC_Temp
       Permet la séparation des différents points froids
TEST:
from main import *
# une par une en changant le filtre mauellement:
#Write.PF_Spec(CarnetLabo,'PF1_dataFrame.csv')
#Write.PF_Spec(CarnetLabo,'PF3_dataFrame.csv') 
#Write.PF_Spec(CarnetLabo,'PF4_dataFrame.csv')
Graph.PF_DC_Temp_plot3()
    '''
    # Data
    (PF1_Temp, PF1_Spec, PF1_Pred, PF1_coeff, PF1_mape, PF1_mse) = PF_DC_Temp('PF1_dataFrame.csv')
    (PF3_Temp, PF3_Spec, PF3_Pred, PF3_coeff, PF3_mape, PF3_mse) = PF_DC_Temp('PF3_dataFrame.csv')
    (PF4_Temp, PF4_Spec, PF4_Pred, PF4_coeff, PF4_mape, PF4_mse) = PF_DC_Temp('PF4_dataFrame.csv')

    # plot
    fig, ax = plt.subplots(figsize=(15,15),layout='constrained')
    # Exp
    ax.plot(PF1_Temp, PF1_Spec, 'o', color='orange')
    ax.plot(PF3_Temp, PF3_Spec, 'o', color='green')
    ax.plot(PF4_Temp, PF4_Spec, 'o', color='blue')
    # Model
    plt.plot(PF1_Temp, PF1_Pred, color='orange',label='fit PF1')
    plt.plot(PF3_Temp, PF3_Pred, color='green',label='fit PF3')
    plt.plot(PF4_Temp, PF4_Pred, color='blue',label='fit PF4')
#    plt.text(0.1, 0.9, fr'$y = {coeff[2]:.3}x^2 + {coeff[1]:.3}x + {coeff[0]:.3}$',
#               transform=plt.gca().transAxes, fontsize=16)
    # taux d'erreur de chaque model
    # PF1
    plt.text(0.27, 0.9, fr'$mape = {PF1_mape:.3}$',
               transform=plt.gca().transAxes, fontsize='large', color='orange')
    plt.text(0.27, 0.85, fr'$mse = {PF1_mse:.3}$',
               transform=plt.gca().transAxes, fontsize='large', color='orange')
    # PF3
    plt.text(0.45, 0.9, fr'$mape = {PF3_mape:.3}$',
               transform=plt.gca().transAxes, fontsize='large', color='green')
    plt.text(0.45, 0.85, fr'$mse = {PF3_mse:.3}$',
               transform=plt.gca().transAxes, fontsize='large', color='green')
    # PF4
    plt.text(0.63, 0.9, fr'$mape = {PF4_mape:.3}$',
               transform=plt.gca().transAxes, fontsize='large', color='blue')
    plt.text(0.63, 0.85, fr'$mse = {PF4_mse:.3}$',
               transform=plt.gca().transAxes, fontsize='large', color='blue')
    # légende
    plt.xlabel("Température °C", fontsize='x-large')
    plt.ylabel("moyenne Spectre", fontsize='x-large')
    plt.legend()
    plt.show() 

def PF_all_removed():
    """
       Plot all points froids after removing prediction of median spectrum
TEST:

    """
    from Classe import Transit
    df_PF = pd.read_csv('PF_DC_save.csv',index_col=0)
    nObsRange = list(df_PF.index.astype(int))
    result = pd.DataFrame(index=nObsRange)
    for nObs in nObsRange:
        transit = Transit.instance[nObs]
        transit.read_Tot()
        modifFreq = transit.modif_Tot()
        result[nObs] = modifFreq
    result.plot()
    plt.show()

def Elev_read():
    """
       Extaction de données affichable pour chaque trajet en azimut constante
TEST:
    """
    from Classe import Elev
    # Création des données
    CopieLabo = pd.read_csv('./CarnetLabo.csv',index_col=0)
    for az in Elev.fileList[0:35]:
        az.read_Tot() # contenu du fichier de Coordonnée
        nObsRange = Classe.Elev.linkTransit(az) # Récupère les nObs correspondants
        az.pointValue = [CopieLabo.loc[int(nObs),'pointValue'] for nObs in nObsRange]
        print(az.elev)
        print(az.pointValue)

def Elev_fit():
    '''
       fit elevation/pointValue 

TEST:
from main import *
#Write.pointValue() # à effectuer sur tout les transit conserné
Graph.Elev_read()
Graph.Elev_fit()
    '''
    from Classe import Elev
    # Reccup en liste
    E = []
    PV = []
    for az in Elev.fileList[0:35]:
        E = E + az.elev
        PV = PV + az.pointValue

    # 1d-list to 2d-list
    E_2d = [[e] for e in E]
    PV_2d = [[pv] for pv in PV]
    
    # fit
    (Y_pred_2d, coeff, err) = fit2D(E_2d,PV_2d)

    # 2d-list to 1d-list
    X_new = [element[0] for element in E_2d]
    Y_new = [element[0] for element in PV_2d]
    Y_pred_new = [element[0] for element in Y_pred_2d]

    #plot
    fig, ax = plt.subplots()
    plt.scatter(X_new, Y_new)
    plt.plot(X_new, Y_pred_new,'o',color='red',label='fit quad')
    ax.set_yscale('log')
    ax.set_ylim(9e-5, 2e-4)
    plt.legend()
    plt.show()

def fit2D(X,Y):
    '''
       take [[],[],...] and [[],[],...]
       give [[],[]...]
       return prédiction for Y
    '''
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
    
    # Transformer les données d'entrée pour inclure le terme quadratique
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    
    # Ajuster le modèle linéaire aux données transformées
    model = LinearRegression()
    model.fit(X_poly, Y)
    
    # Imprimer le résultat de l'ajustement
    print(f"Coefficient pour le terme linéaire : {model.coef_[0][0]}")
    print(f"Coefficient pour le terme quadratique : {model.coef_[0][1]}")
    print(f"Ordonnée à l'origine (biais) : {model.intercept_[0]}")
    coeff = [model.intercept_[0],
             model.coef_[0][0],
             model.coef_[0][1]]
    
    # Prédiction
    Y_pred = model.predict(X_poly)
    
    # Calculer l'erreur quadratique moyenne (MSE)
    mse = mean_squared_error(Y, Y_pred)
    print(f"Erreur Quadratique Moyenne (MSE) : {mse}")
    mape = mean_absolute_percentage_error(Y,Y_pred)
    print(f"Pourcentage d'Erreur Absolue Moyenne (MAPE) : {mape}")
    
    return (Y_pred, coeff, mape, mse)

def Elev_fit3D():
    '''
       Affiche le bruit de fond en fonction des coordonnée azimut/elevation
       plusieurs méthodes de formatages des données (DataFrame, Series)
       2 plot différents: 2D et surface 3D

TEST:
from main import *
Graph.Elev_read()
Graph.Elev_fit2D()

    '''
    from Classe import Elev
    
    # Import Data from class Elev
    E = [] # élévation
    A = [] # azimut
    PV = [] # pointValue
    for az in Elev.fileList[0:35]:
        E = E + az.elev
        A = A + az.azimut
        PV = PV + az.pointValue
    
    ## méthode 2: Assembling series in big DataFrame 
    #Elev.SerList = []
    #for az in Elev.fileList[0:35]:
    #    Elev.SerList.append(
    #            pd.Series(az.pointValue, index=az.elev, name=str(az.azimut[0]))
    #            ) 
    #
    #df = pd.DataFrame(Elev.SerList)
    
    # meth3: namedTuple to DataFrame
    from collections import namedtuple
    Point3D = namedtuple('Point3D','elevation azimut value') # give name
    tuple_Exp = zip(E, A, PV) # simple tuple
    Ntuple_Exp = [Point3D(t[0],t[1],t[2]) for t in tuple_Exp] # namedtuple
    df_Exp = pd.DataFrame(Ntuple_Exp) # DataFrame
    df_Exp.to_csv('pointValue_az_elev.csv')
    
    # Removing the 5 biggest value as aberations
    df_Exp_clean = df_Exp.sort_values(by='value').iloc[:-5,:]
    df_Exp_clean = df_Exp_clean.groupby(['azimut','elevation'],as_index=False).mean()
    df_Exp_clean.to_csv('pointValue_az_elev_cleaned.csv')
    
    # Shaping for scikit library
    XY = [[e, a] for (e,a) in zip(df_Exp_clean.elevation, df_Exp_clean.azimut)]
    PV = [[pv] for pv in df_Exp_clean.value]
    
    # Scikit Polynomial model  
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
    
    # Transformer les données d'entrée pour inclure le terme quadratique
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    XY_poly = poly_features.fit_transform(XY)
    
    # Ajuster le modèle linéaire aux données transformées
    model = LinearRegression()
    model.fit(XY_poly, PV)
    
    # Imprimer le résultat de l'ajustement
    print(f"Coefficient pour le terme linéaire : {model.coef_[0][0]}")
    print(f"Coefficient pour le terme quadratique : {model.coef_[0][1]}")
    print(f"Ordonnée à l'origine (biais) : {model.intercept_[0]}")
    
    # Prédiction
    PV_pred = model.predict(XY_poly)
    PV_pred_list = [element[0] for element in PV_pred] # model 1-dimention list
    df_Exp_clean['model'] = PV_pred_list
    #df_Exp_clean.to_csv('DataFrame_elev_az_value_model.csv')
    
    # Group, split, mean, sort
    df_group = df_Exp_clean.groupby("azimut")
    DF = [] # list of DataFrame by azimut
    for (name,group) in df_group:
        #if name != 60 and name != 75: # azimut à vérifier
        DF.append(group)
        print(name)

    # plot 2D
    fig, ax = plt.subplots(figsize=(15,15),layout='constrained')
    for frame in DF:
        #ax.plot(frame["elevation"] , frame["value"], label=frame["azimut"].iloc[0])
        ax.scatter(frame["elevation"] , frame["value"],color='k',marker='+')
        ax.plot(frame["elevation"], frame["model"])
        #ax.scatter(frame["azimut"], frame["elevation"], color='b')

    # Calculer l'erreur quadratique moyenne (MSE)
    mse = mean_squared_error(PV, PV_pred)
    print(f"Erreur Quadratique Moyenne (MSE) : {mse}")
    mape = mean_absolute_percentage_error(PV,PV_pred)
    print(f"Pourcentage d'Erreur Absolue Moyenne (MAPE) : {mape}")
   
    # 2D plot setting
    #ax.set_yscale('log')
    ax.set_ylim(9e-5, 10.5e-5)
    plt.tick_params(labelsize='x-large')
    ax.set_xlabel('Élevation',fontsize='xx-large')
    ax.set_ylabel('Intensité moyenne du Signal',fontsize='xx-large')
    #ax.set_xlabel('Azimut',fontsize='xx-large')
    #ax.set_ylabel('Élevation',fontsize='xx-large')
    ax.grid(True)
    plt.text(25, 0.0001035, fr'$mape = {mape:.3}$',fontsize=15)
    plt.text(25, 0.0001025, fr'$mse = {mse:.3}$',fontsize=15)
    #ax.legend()
    
#    fig.savefig('/home/mael/Bureau/StageMael2023/Graph_download/usefull/Bruit_Elev_poly.png',bbox_inches='tight',dpi=200)
    plt.show()
    
#    # To 2d-array
#    E_array = np.array([[part1,part2] for (part1,part2) in zip(E_new[:108],E_new[108:])])
#    A_array = np.array([[part1,part2] for (part1,part2) in zip(A_new[:108],A_new[108:])])
#    PV_array = np.array([[part1,part2] for (part1,part2) in zip(PV_new[:108],PV_new[108:])])
#    PV_pred_array = np.array([[part1,part2] for (part1,part2) in zip(PV_pred_list[:108], PV_pred_list[108:])])
    
#    # meth2: DataFrame avec np.nan => chamboule les coordonnées
#    PV_pred_list = [element[0] for element in PV_pred] # model 1-dimention list
#    SerListModel = list(Elev.SerList) # Copy experiment pd.Series data
#    for PV in PV_pred_list:
#         for s in SerListModel:
#             serLen = len(list(s.index)) # series len()
#             s.loc[:] = PV_pred_list[:serLen] # replace values in Series
#             del PV_pred_list[:serLen] # del in list of model data
#    
#    df_model = pd.DataFrame(SerListModel)
#    PV_pred_array = df_model.to_numpy()
#    print(PV_pres_array)
    
#    # plot surface 3D 
#    from mpl_toolkits.mplot3d import Axes3D
#    from matplotlib import cm
#    fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(15,15))
#    ax.set_zscale('log')
#    ax.set_zlim(9e-5, 11e-5)
#    plt.tick_params(labelsize='x-large')
#    ax.set_zticks([]) # no zticks
#    ax.set_zticks([], minor=True)
#    ax.set_xlabel('Élevation',fontsize='xx-large')
#    ax.set_ylabel('Azimut',fontsize='xx-large')
#    ax.text2D(0.05, 0.95, fr'$mape = {mape:.3}$', fontsize=15, transform=ax.transAxes)
#    ax.text2D(0.05, 0.90, fr'$mse = {mse:.3}$', fontsize=15, transform=ax.transAxes)
#    
#    ax.scatter3D(df_Exp_clean.elevation ,df_Exp_clean.azimut ,df_Exp_clean.value ,zdir='z',color='b')
#    surf = ax.plot_trisurf(df_Exp_clean.elevation, df_Exp_clean.azimut, df_Exp_clean.model, cmap=cm.jet, linewidth=0.1)
#    cb = fig.colorbar(surf, shrink=0.5, aspect=5)
#    cb.set_label(label='Intensité moyenne du signal', size=15)
#    plt.show()
#    fig.savefig('/home/mael/Bureau/StageMael2023/Graph_download/usefull/Bruit_3D_sansAberation.png',bbox_inches='tight',dpi=200)
        

def PF_all_deltaPara(sizeWanted, Sensor_Area):
    '''
       Corélation des Spectre de Points Froid avec les sensors: 
       - Coéfficients avec SALib
       - Coordonnée Parallèle avec plotly

       Pour réduire les dimentions on moyenne:
       - soit sur chaque fréquence et seconde 
       (Afin de faire correspondre les donnée en nombres de points,
       on utilise un fonction d'interpolation) 

       - soit sur chaque spectre et capteur 
       (ici pas besoin de redimentionnement)
       
TEST:
from main import *
Graph.PF_all_deltaPara(100,
[('ra','coord','indi','f4klo'),
('dec','coord','indi','f4klo'),
('ext','obj','temp','f4klo'),
('cav','obj','temp','f4klo'),
('preamp2','obj','temp','f4klo'),
('LFPG','station','metar','weather')])

erreur avec az, elev
    '''
    import plotly.graph_objects as go
    from Classe import Transit

    # Read PF Spectrum Data
    df_PF = pd.read_csv('PF4_dataFrame.csv',index_col=0)
    # Read PF frequency Scale
    freqScale = df_PF.columns.astype(float).to_numpy()
    minF = freqScale[0]
    maxF = freqScale[-1]

    # Reshape Spec
    x_freq = freqScale
    y_freq = df_PF.T.mean().to_numpy() # Transpose to mean on each Spectrum
    print(y_freq)
    #y_freq = Transit.instance[0].resize(x_freq, y_freq , sizeWanted) #TODO: sortir le resize de la classe Transit
    print('Taille de l\'échantillon', len(y_freq))

    # Build Sensor Data
    Sensor_ParaDict = [] # for plotly library
    for sensor in Sensor_Area: # shaping sensors data
        # Empty df for adding  
        df_sen = pd.DataFrame()
        sen_list = []
        nObsRange = list(df_PF.index.astype(int)) 
        # Import the Data of sensor on each Observation
        for nObs in nObsRange:
            transit = Transit.instance[nObs]
            transit.import_Sensor(sensor[0], sensor[1], sensor[2], sensor[3])
            # Resizing
            x_sen = transit.sensorFrame.T['time'][sensor[0]]
            y_sen = transit.sensorFrame.T['value'][sensor[0]]
            #y_sen = transit.resize(x_sen, y_sen , sizeWanted)
            sen_list.append(y_sen)

        # Transpose to mean on each sensor
        sen_df = pd.DataFrame(sen_list).T 
        sen_mean = sen_df.mean().to_numpy()
        # Each columns represent a sensor
        df_sen[str(sensor)] = sen_mean 
        print(df_sen)
        # Parallèle Coordinate graph parameters
        Sensor_ParaDict.append(
                dict(range = [np.min(sen_mean), np.max(sen_mean)],
                     label = transit.sensorFrame.T['name'][sensor[0]],
                     values = sen_mean)
                )
    # end of for sensor loop

    ## Search delta indice
    from SALib.analyze import delta

    # Définir le problème
    problem = {
        'num_vars': len(Sensor_Area),
        'names': [d["label"] for d in Sensor_ParaDict],
        'bounds': [d["range"] for d in Sensor_ParaDict] #TODO
    }

    # sensors value in list of list
    Values = [list(d["values"]) for d in Sensor_ParaDict]
    # to X 2d-array
    X = np.array(list(zip(*Values)))

    # Spectrum values array
    Y = np.array(y_freq)

    # Analyse Delta
    Si = delta.analyze(problem, X, Y, print_to_console=True)

    # Plot Parallel Coordinate Graph with plotly library
    fig = go.Figure(data=
        go.Parcoords(
            line_color='blue',
            dimensions = list([
                dict(range = [np.min(y_freq),np.max(y_freq)],
                     label = 'Spectrum value', values = y_freq)])
                + Sensor_ParaDict
            )# each dict make a dimention
        )
    fig.show()

