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

def Graph_Obs1(nObs, mode):
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

def Graph_Obs2(a,b):
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

def Graph_Obs9(LObs):
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

def Graph_PF_write(CarnetLabo):
    """
       écrit Tout les spectres des Points Froids 4 3 1
       Dans un fichier csv
    """

    # Extrait les objet transit tous les point froids du carnet de Labo
    PFobjet = [] 
    # Select by astre name
    astre = CarnetLabo.groupby('astre')
    PF1 = astre.get_group('PF1')
    PF3 = astre.get_group('PF3')
    PF4 = astre.get_group('PF4')

    df = pd.concat([PF1,PF3,PF4]) # single DataFrame
    #df = df.groupby('gain').get_group('38.0') # gain filter
    print(df)
        
    PFobjet = list(df['instance'])  # storage of class Transit objects

    # initialisation des données de capture
    for PF in PFobjet:
        PF.read_Tot()
        PF.read_Freq()
    
    # Storage of spectrum in CarnetLabo
    CarnetLabo["freqScale"] = [transit.freqScale for transit in Transit.instance]
    CarnetLabo["Spectrum"] = [transit.modeFreq for transit in Transit.instance]

    CarnetLabo.loc[df_PF].to_csv('PF_dataFrame.csv')

def Graph_PF_read():
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

def Graph_PF_DC():
    """
       Identifi le spectre continu de chaque point froids avec resample
       calcule le spectre médian
       calcule la moyenne de chaque spectre
    """
    fig, ax = plt.subplots(figsize=(15,15))

    # read
    df_PF = pd.read_csv('PF_dataFrame.csv',index_col=0)
    # to list of float
    freqScale = df_PF.columns.astype(float).to_numpy()
    minF = freqScale[0]
    maxF = freqScale[-1]

    # iter on spectrum and plot spectrum resampled
    freq_new = np.linspace(minF, maxF, 60, endpoint=False)
    df_PF_res = pd.DataFrame(index=freq_new)
    for (label,content) in df_PF.T.items():
        spec_res = sp.signal.resample(list(content),60) # resample with scipy
        ax.plot(freq_new, spec_res, color='b')
        mean = np.average(spec_res)
        ax.plot([minF, maxF],[mean, mean], color='purple')
        df_PF_res[label] = spec_res # storage

    df_PF_res = df_PF_res.T 
    #df_PF_res["mean"] = df_PF_res.mean(1)
    print(df_PF_res)

    # Mediane of rsampled
    medianeGlobale = df_PF_res.median(axis=0) # by freq
    ax.plot(freq_new, medianeGlobale, color='red', label='mediane') # Plot

    # show
    #ax.set_title("Mediane des spectres de tout les tracking de points froids\n(gain à 38)", fontsize=20)
    ax.set_xlabel('Fréquence (en MHz)')
    ax.set_ylabel('Puissance du signal')
    plt.xticks(np.arange(minF, maxF+0.00001, 0.2))
    ax.ticklabel_format(useOffset=False)
    ax.grid(True)
    ax.legend()
    plt.show()

def Graph_Sensor1():
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

def Graph_Paral_Coord():
    '''Coordonnée Parallèle multi-dimentions'''
    import plotly.graph_objects as go

    # extraction des données
    transit = Transit.instance[258]
    transit.read_Tot()
    transit.read_Freq()

    # extraction des données
    transit = Transit.instance[259]
    transit.read_Tot()
    transit.read_Freq()

    # extraction des données
    transit = Transit.instance[260]
    transit.read_Tot()
    transit.read_Freq()

    minmax = [transit.freqScale[0], transit.freqScale[-1]]
 
    fig = go.Figure(data=
        go.Parcoords(
            line_color='blue',
            dimensions = list([
                dict(range = minmax,
                     label = 'Fréquence', values = Transit.instance[258].freqScale),
                dict(range = [0,0.0005],
                     label = 'Spectre PF4', values = Transit.instance[258].modeFreq),
                dict(range = [0,0.0005],
                     label = 'Spectre PF3', values = Transit.instance[259].modeFreq),
                dict(range = [0,0.0005],
                     label = 'Spectre PF1', values = Transit.instance[260].modeFreq)
                ])
            )
        )
    fig.show()

def Graph_Paral_Sensor(nObs, sizeWanted, Sensor_Area):
    '''
       Coordonnée Parallèle multi-dimentions des sensors
       + SALib delta corélation indice
TEST:
from main import *
Graph_Paral_Sensor(258,100,[('ra','coord','indi'),('dec','coord','indi'),('ext','obj','temp'),('cav','obj','temp')])
    '''
    import plotly.graph_objects as go

    # Spectre
    transit = Transit.instance[nObs]
    transit.read_Tot()
    transit.read_Freq()
    # Resize Freq
    x = transit.freqScale
    y = transit.modeFreq
    Freq_resiz = transit.resize(x, y , sizeWanted)
    print('Taille de l\'échantillon', len(Freq_resiz))

    # Sensors
    Sensor_Area_defaut = [('ra','coord','indi'),
                          ('dec','coord','indi'),
                          ('ext','obj','temp'),
                          ('cav','obj','temp')]

    Sensor_ParaDict = [] # for plotly library
    for sensor in Sensor_Area: # shaping sensors data
        transit.import_Sensor(sensor[0], sensor[1], sensor[2])
        # Resizing
        x = transit.sensorFrame.T['time'][sensor[0]]
        y = transit.sensorFrame.T['value'][sensor[0]]
        y = transit.resize(x, y , sizeWanted)
        # Storage
        Sensor_ParaDict.append(
                dict(range = [np.min(y), np.max(y)],
                     label = str(sensor), 
                     values = y)
                )

    # Giving delta indice
    from SALib.analyze import delta
    
    # Définir le problème
    problem = {
        'num_vars': 4,
        'names': ['ra', 'dec', 'ext', 'cav'],
        'bounds': [d["range"] for d in Sensor_ParaDict] #TODO
    }
    
    # Remplacer par vos propres valeurs
    Values = [list(d["values"]) for d in Sensor_ParaDict]
    print(Values)
    X = np.array(list(zip(*Values))) 
    print(X)
    Y = np.array(Freq_resiz)
    # Analyse Delta
    Si = delta.analyze(problem, X, Y, print_to_console=True)

#    # Plot with plotly library
#    fig = go.Figure(data=
#        go.Parcoords(
#            line_color='blue',
#            dimensions = list([
#                dict(range = [0,0.0005],
#                     label = 'Spectrum value', values = Freq_resiz)]) 
#                + Sensor_ParaDict
#            )# each dict make a dimention
#        )
#    fig.show()

def Graph_PFall_deltaPara(sizeWanted, Sensor_Area):
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
Graph_PFall_deltaPara(100,
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

    # Resample
    x_freq = freqScale
    y_freq = df_PF.T.mean().to_numpy() # Transpose to mean on each Spectrum
    print(y_freq)
    #y_freq = Transit.instance[0].resize(x_freq, y_freq , sizeWanted) #TODO: sortir le resize de la classe Transit
    print('Taille de l\'échantillon', len(y_freq))

    # Build Sensor Data
    Sensor_ParaDict = [] # for plotly library
    for sensor in Sensor_Area: # shaping sensors data
        df_sen = pd.DataFrame()
        sen_list = []
        nObsRange = list(df_PF.index.astype(int))
        for nObs in nObsRange:
            transit = Transit.instance[nObs]
            transit.import_Sensor(sensor[0], sensor[1], sensor[2], sensor[3])
            # Resizing
            x_sen = transit.sensorFrame.T['time'][sensor[0]]
            y_sen = transit.sensorFrame.T['value'][sensor[0]]
            #y_sen = transit.resize(x_sen, y_sen , sizeWanted)
            sen_list.append(y_sen)

        sen_df = pd.DataFrame(sen_list).T # Transpose to mean on each sensor
        sen_mean = sen_df.mean().to_numpy()
        df_sen[str(sensor)] = sen_mean # each columns represent a sensor
        print(df_sen)
        # Storage
        Sensor_ParaDict.append(
                dict(range = [np.min(sen_mean), np.max(sen_mean)],
                     label = transit.sensorFrame.T['name'][sensor[0]],
                     values = sen_mean)
                )

    # Giving delta indice
    from SALib.analyze import delta

    # Définir le problème
    problem = {
        'num_vars': len(Sensor_Area),
        'names': [d["label"] for d in Sensor_ParaDict],
        'bounds': [d["range"] for d in Sensor_ParaDict] #TODO
    }

    # Remplacer par vos propres valeurs
    Values = [list(d["values"]) for d in Sensor_ParaDict]
    print(Values)
    X = np.array(list(zip(*Values)))
    print(np.size(X))
    Y = np.array(y_freq)
    print(np.size(y_freq))
    # Analyse Delta
    Si = delta.analyze(problem, X, Y, print_to_console=True)

    # Plot with plotly library
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


def Write_all_pointValue(begin,end):
    '''
       Calcul pour chaque transit la moyenne de toutes les valeurs du signal réccupéré
       puis l'enregistre dans la colone de pointValue 
       Cette version permet de séparer en intervale d'observation [begin:end]
       afin de ne pas faire planter votre ordianteur
    '''
    CopieLabo = pd.read_csv('./CarnetLabo.pointValue.csv',index_col=0)
    for transit in Transit.instance[begin:end]: 
        transit.read_Tot()
        df = pd.DataFrame(transit.modeTot)
        point = df.mean().mean()
        CopieLabo.loc[int(transit.nObs), 'pointValue'] = point
    CopieLabo.to_csv('./CarnetLabo.pointValue.csv',index_label='nObs')

def Graph_Elev_read():
    """Extaction de données affichable pour chaque trajet en azimut constante"""
    # Création des données
    CopieLabo = pd.read_csv('./CarnetLabo.csv',index_col=0)
    for az in Elev.fileList[0:35]:
        az.read_Tot() # contenu du fichier de Coordonnée
        nObsRange = Classe.Elev.linkTransit(az) # Récupère les nObs correspondants
        az.pointValue = [CopieLabo.loc[int(nObs),'pointValue'] for nObs in nObsRange]
        print(az.elev)
        print(az.pointValue)

def Graph_Elev_fit1D():
    # Mise en forme des donnée d'entrée
    E = []
    PV = []
    for az in Elev.fileList[0:35]:
        E = E + az.elev
        PV = PV + az.pointValue
    E = [[e] for e in E]
    PV = [[pv] for pv in PV]
    print(E[0:4])
    print(PV[0:4])

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error
    
    # Transformer les données d'entrée pour inclure le terme quadratique
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    E_poly = poly_features.fit_transform(E)
    
    # Ajuster le modèle linéaire aux données transformées
    model = LinearRegression()
    model.fit(E_poly, PV)
    
    # Imprimer le résultat de l'ajustement
    print(f"Coefficient pour le terme linéaire : {model.coef_[0][0]}")
    print(f"Coefficient pour le terme quadratique : {model.coef_[0][1]}")
    print(f"Ordonnée à l'origine (biais) : {model.intercept_[0]}")
    
    # Prédiction
    PV_pred = model.predict(E_poly)
    
    # Calculer l'erreur quadratique moyenne (MSE)
    mse = mean_squared_error(PV, PV_pred)
    print(f"Erreur Quadratique Moyenne (MSE) : {mse}")
    
    #plot
    E_new = [element[0] for element in E]
    PV_new = [element[0] for element in PV]
    PV_pred_new = [element[0] for element in PV_pred]

    fig, ax = plt.subplots()
    plt.scatter(E_new, PV_new)
    plt.plot(E_new, PV_pred_new,'o',color='red',label='fit quad')
    ax.set_yscale('log')
    ax.set_ylim(9e-5, 2e-4)
    plt.legend()
    plt.show()

def Graph_Elev_fit2D():
    '''
       Affiche le bruit de fond en fonction des coordonnée azimut/elevation
       plusieurs méthodes de formatages des données (DataFrame, Series)
       2 plot différents: 2D et surface 3D

TEST:
from main import *
Graph_Elev_read()
Graph_Elev_fit2D()

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

