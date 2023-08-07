#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import Externe
from math import *
import matplotlib.pyplot as plt
import numpy as np
import statistics as st
import scipy as sp

import Classe
from Classe import *
from main import *

# Graph storage directory
GSD = "../Graph_download/"

def Obs1(nObs, mode):
    '''
    Affiche une Observation numéro nObs
    Selon le mode choisi: temporel 'T' ou frequentiel 'F'
    '''
    self = Transit.instance[nObs]# choix de l'objet étudié
    self.read_Tot()

    fig, ax = plt.subplots()
    if mode == 'F':
         self.read_Freq()
         self.graph_Freq(ax)
    elif mode == 'T':
         self.read_Temp()
         self.graph_Temp(ax)
    plt.show()

def Obs2(a,b):
    """Comparaison des Transit n° a et b"""
    fig, ax = plt.subplots(2,2,figsize=(15,15), layout="constrained")
    
    self = Transit.instance[a] # choix de l'objet étudié
    self.read_Tot(self) ### NE PAS OUBLIER
    self.graph_Freq(ax[0,0])
    self.graph_Temp(ax[0,1])
    
    self = Transit.instance[b] # choix de l'objet étudié
    Classe.Transit.read_Tot(self) ### NE PAS OUBLIER
    self.graph_Freq(ax[1,0])
    self.graph_Temp(ax[1,1])
    
    print('\n')
    plt.show()

def Obs9(LObs):
    """affichage en 3*3 de la liste de nObs donnée"""
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
    '''Affiche les données influx de 4 varibales'''
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
       Puis calule et affiche la mediane en chaque point
    """
    fig, ax = plt.subplots(figsize=(15,15))

    # read
    df_PF = pd.read_csv('PF_dataFrame.csv',index_col=0)
    # to list of float
    freqScale = df_PF.columns.astype(float).to_numpy()

    # iter on spectrum and plot
    for (label,content) in df_PF.T.items(): 
        ax.plot(freqScale,list(content),color='b')

    # Mediane
    medianeGlobale = df_PF.median(axis=0) # by freq
    ax.plot(freqScale, medianeGlobale, color='red', label='mediane') # Plot

    # mediane resampled
    med_res = sp.signal.resample(medianeGlobale,60)
    freq_new = np.linspace(freqScale[0], freqScale[-1], 60, endpoint=False)
    ax.plot(freq_new, med_res, color='purple')

    # show
    ax.set_title("Mediane des spectres de tout les tracking de points froids\n(gain à 38)", fontsize=20)
    #ax.ticklabel_format(useOffset=False)
    ax.grid(True)
    ax.legend()
    plt.show()

def PF_DC():
    """
       Identifi le spectre continu de chaque point froids avec resample
       calcule le spectre médian
       calcule la moyenne de chaque spectre
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

    # show
    ax.set_xlabel('Fréquence (en MHz)')
    ax.set_ylabel('Puissance du signal')
    plt.xticks(np.arange(minF, maxF+0.00001, 0.2))
    ax.ticklabel_format(useOffset=False)
    ax.grid(True)
    ax.legend()
    plt.show()
    
    # Save
    df_PF_new.to_csv('PF_DC_save.csv')

def PF_DC_Temp():
    '''
       plot mean by spectrum / mean Temperature
TEST:
from main import *
Graph.PF_DC()
Graph.PF_DC_Temp()
    '''
    df_PF = pd.read_csv('PF_DC_save.csv',index_col=0)
    nObsRange = list(df_PF.index.astype(int))
    # Import the Data of sensor on each Observation
    df_sen = pd.DataFrame(index=['ext']) # filled by loop
    for nObs in nObsRange:
        # test if exist and request influx if not
        transit = Transit.instance[nObs]
        transit.import_Sensor('ext','obj','temp','f4klo')
        # pick values
        y_sen = transit.sensorFrame.T['value']['ext']
        y_sen = np.average(y_sen)
        # add to DataFrame
        df_sen[nObs] = y_sen # new column

    # Prepare data
    Spec_M = df_PF['mean']
    Spec_M = Spec_M.sort_values(ascending=True).to_numpy()
    Temp = df_sen.T # put nObs in index
    Temp = Temp.sort_values(by='ext',ascending=True).to_numpy()

    # figure
    fig, ax = plt.subplots(figsize=(15,15))
    ax.plot(Temp, Spec_M)

    # fit with scikit learn
    # 1d-list to 2d-list
    T_2d = list(Temp)
    print(T_2d)
    M_2d = [[m] for m in list(Spec_M)]
    print(M_2d)

    # Fit Polynomiale
    (Y_pred_2d, [x0,x1,x2], err) = fit2D(T_2d,M_2d)

    # 2d-list to 1d-list
    Y_pred_1d = [element[0] for element in Y_pred_2d]

    #plot
    plt.plot(Temp, Y_pred_1d,'o',color='red',label='fit quad')
    plt.text(0.1, 0.9, fr'$y = {x2:.3}x^2 + {x1:.3}x + {x0:.3}$',
               transform=plt.gca().transAxes, fontsize=16)
    plt.legend()
    plt.show()


def Elev_read():
    """
       Extaction de données affichable pour chaque trajet en azimut constante
    """
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
Graph.Elev_read()
Graph.Elev_fit()
    '''
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
    
    return (Y_pred, coeff, mape)

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
    
    # Removing the 5 biggest value as aberations
    df_Exp_clean = df_Exp.sort_values(by='value').iloc[:-10,:]
    df_Exp_clean = df_Exp_clean.groupby(['azimut','elevation'],as_index=False).mean()
    
    # Shaping for scikit library
    XY = [[e, a] for (e,a) in zip(df_Exp_clean.elevation, df_Exp.azimut)]
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
    df_Exp_clean.to_csv('DataFrame_elev_az_value_model.csv')
    
    # Group, split, mean, sort
    df_group = df_Exp_clean.groupby("azimut")
    DF = [] # list of DataFrame by azimut
    for (name,group) in df_group:
        if name != 60 and name != 75:
            DF.append(group)
            print(name)
    
    # plot 2D
    fig, ax = plt.subplots()
    for frame in DF:
        ax.plot(frame["elevation"] , frame["value"], label=frame["azimut"].iloc[0])
        ax.scatter(frame["elevation"] , frame["value"],color='k',marker='+')
        #ax.plot(frame["elevation"], frame["model"])
    
    ax.set_yscale('log')
    ax.set_ylim(9e-5, 11e-5)
    ax.set_xlabel('Elevation')
    ax.set_ylabel('Signal')
    plt.legend()
    plt.show()
    
    # Calculer l'erreur quadratique moyenne (MSE)
    mse = mean_squared_error(PV, PV_pred)
    print(f"Erreur Quadratique Moyenne (MSE) : {mse}")
    mape = mean_absolute_percentage_error(PV,PV_pred)
    print(f"Pourcentage d'Erreur Absolue Moyenne (MAPE) : {mape}")
    
#    # To 2d-array
#    E_array = np.array([[part1,part2] for (part1,part2) in zip(E_new[:108],E_new[108:])])
#    A_array = np.array([[part1,part2] for (part1,part2) in zip(A_new[:108],A_new[108:])])
#    PV_array = np.array([[part1,part2] for (part1,part2) in zip(PV_new[:108],PV_new[108:])])
#    PV_pred_array = np.array([[part1,part2] for (part1,part2) in zip(PV_pred_list[:108], PV_pred_list[108:])])
    
#    # meth2: DataFrame Nan chamboule les coordonnées
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
#    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#    ax.set_zscale('log')
#    ax.set_zlim(9e-5, 11e-5)
#    ax.set_xlabel('Élevation')
#    ax.set_ylabel('Azimut')
#    ax.set_zlabel('Background Noise')
#    
#    ax.scatter3D(df_Exp_clean.elevation ,df_Exp_clean.azimut ,df_Exp_clean.value ,zdir='z',color='b')
#    surf = ax.plot_trisurf(df_Exp_clean.elevation, df_Exp_clean.azimut, df_Exp_clean.model, cmap=cm.jet, linewidth=0.1)
#    fig.colorbar(surf, shrink=0.5, aspect=5)
#    #plt.savefig('{}Elev_fit2_{}.png'.format(GSD,next(iter(os.popen('date \"+%y%m%d%H%M%S\"')))[:-1]))
#    plt.show()
#        
'''
# Erreur Quadratique Moyenne:
# degré 2 (MSE) : 2.8544429484034086e-10
# degré 3 (MSE) : 2.7562763785302257e-10
# degré 4 (MSE) : 2.6445467564188925e-10
# degré 8 (MSE) : 2.5372139821167926e-10

# Pourcentage d'Erreur Absolue Moyenne 
# degré 2 (MAPE) : 0.027332006694158084
# degré 3 (MAPE) : 0.022868027542152735
# degré 4 (MAPE) : 0.021643753017180075
'''

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
('Pluto','sdr','gain','f4klo'),
('LFPG','station','metar','weather')])

erreur avec az, elev
    '''
    import plotly.graph_objects as go

    # Read PF Spectrum Data
    df_PF = pd.read_csv('PF_dataFrame.csv',index_col=0)
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
    print(np.size(X))

    # Spectrum values array
    Y = np.array(y_freq)
    print(np.size(y_freq))

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

