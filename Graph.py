#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import Externe
from math import *
import matplotlib.pyplot as plt
import numpy as np
import statistics as st
import pickle

import Classe
from Classe import *

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


def Graph_PF9():
    """
       Affiche dans 9 plot séparés les spectres des Points Froids 4 3 1
       à 3 instants différents chacun
    """
       
    fig, ax = plt.subplots(3,3,figsize=(15,15), layout="constrained")
    
    # Numéros des observations considérées
    nObsPF4 = [258, 261, 264]
    nObsPF3 = [259, 262, 265]
    nObsPF1 = [260, 263, 266]
    nObsPF = [nObsPF4, nObsPF3, nObsPF1]
    titre = np.array(['PF4', 'PF3', 'PF1'])

    # Parcour des numéros d'Observation et remplissage du subplot
    for PF in nObsPF:
        j = nObsPF.index(PF)
        for nObs in PF:
            i = PF.index(nObs)  
            self = Transit.instance[nObs] # choix de l'objet étudié
            self.read_Tot() ### NE PAS OUBLIER
            self.read_Freq()
            self.graph_Freq(ax[i,j])
            ax[i,j].set_title(titre[j])
            ax[i,j].ticklabel_format(useOffset=False)

    print('\n')
    plt.show()


def Graph_PF_mediane():
    """
       Affiche en un seul plot
       Tout les spectres des Points Froids 4 3 1
       Puis calule et affiche la mediane en chaque point (en rouge)
    """

    fig, ax = plt.subplots(figsize=(15,15))

    # Numéros des observations considérées
    nObsPF4 = [258, 261, 264]
    nObsPF3 = [259, 262, 265]
    nObsPF1 = [260, 263, 266]
    nObsPF = [nObsPF4, nObsPF3, nObsPF1]
    PFobjet = []

    # Tout les spectres sur un meme figure
    for PF in nObsPF:
        j = nObsPF.index(PF)
        for nObs in PF:
            i = PF.index(nObs)
            self = Transit.instance[nObs] # choix de l'objet étudié
            self.read_Tot() ### NE PAS OUBLIER
            self.read_Freq()
            self.graph_Freq(ax)
            PFobjet.append(self)

    # Ajout de la médiane en chaque point (en rouge)
    # La médiane n'étant pas un objet de la classe Transit
    # la fonction self.graph_Freq() ne peut pas être utilisé
    ref = PFobjet[0] # un des objet est utilisé comme référence
    minF = ref.freqC - (1/2)*ref.BP
    maxF = ref.freqC + (1/2)*ref.BP
    freqScale = np.linspace( minF, maxF, len(ref.modeFreq))
    # Mise en parallèle des modeFreq
    freqParallèle = zip(*[PFobjet[i].modeFreq for i in range(len(PFobjet))])
    # Médiane pour chaque frequence
    medianeGlobale = [st.median(x) for x in freqParallèle]
    ax.plot(freqScale, medianeGlobale, color='red', label='mediane') # Plot

    # Paramètres d'affichage 
    ax.set_title("Mediane des spectres de trois points froids avec 3 prises chacun", fontsize=20)
    ax.ticklabel_format(useOffset=False)
    ax.legend()
    print('\n')
    plt.show()

def Graph_PF_all(CarnetLabo):
    """
       Affiche en un seul plot
       Tout les spectres des Points Froids 4 3 1
       Puis calule et affiche la mediane en chaque point (en rouge)
    """

    fig, ax = plt.subplots(figsize=(15,15))

    # Numéros des observations considérées
    PFobjet = CarnetLabo.Sort(astre=('PF1','PF3','PF4'), gain=38.0)

    for PF in PFobjet:
        PF.read_Tot()
        PF.read_Freq()
        PF.graph_Freq(ax)

    # Ajout de la médiane en chaque point (en rouge)
    # La médiane n'étant pas un objet de la classe Transit
    # la fonction self.graph_Freq() ne peut pas être utilisé
    ref = PFobjet[0] # un des objet est utilisé comme référence
    minF = ref.freqC - (1/2)*ref.BP
    maxF = ref.freqC + (1/2)*ref.BP
    freqScale = np.linspace( minF, maxF, len(ref.modeFreq))
    # Mise en parallèle des modeFreq
    freqParallèle = zip(*[PFobjet[i].modeFreq for i in range(len(PFobjet))])
    # Médiane pour chaque frequence
    medianeGlobale = [st.median(x) for x in freqParallèle]
    ax.plot(freqScale, medianeGlobale, color='red', label='mediane') # Plot

    # Paramètres d'affichage 
    ax.set_title("Mediane des spectres de tout les tracking de points froids\n(gain à 38)", fontsize=20)
    ax.ticklabel_format(useOffset=False)
    ax.legend()
    print('\n')
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
    '''Coordonnée Parallèle multi-dimentions des sensors'''
    import plotly.graph_objects as go

    # extraction du transit
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
    Sensor_Value = []
    Sensor_ParaDict = []
    for sensor in Sensor_Area: # on met en forme les data des sensor
        transit.import_Sensor(sensor[0], sensor[1], sensor[2])
        # Resizing
        x = transit.PDframe.T['time'][sensor[0]]
        y = transit.PDframe.T['value'][sensor[0]]
        ynew = transit.resize(x, y , sizeWanted)
        # Storage
        Sensor_Value.append(ynew)
        Sensor_ParaDict.append(
                dict(range = [np.min(ynew), np.max(ynew)],
                     label = str(sensor), 
                     values = ynew)
                )

    fig = go.Figure(data=
        go.Parcoords(
            line_color='blue',
            dimensions = list([
                dict(range = [0,0.0005],
                     label = 'Spectrum value', values = Freq_resiz)]) 
                + Sensor_ParaDict
            )# Ici chaque dictionnaire génère une dimention
        )
    fig.show()

def Write_all_pointValue():
    '''
       Calcul pour chaque transit la moyenne de toutes les valeurs du signal réccupéré
       puis l'enregistre dans la colone de pointValue 
    '''
    for obj in Transit.instance:
        obj.read_modeTot()
        obj.read_point()
    # pas fini

def Graph_Elev_write():
    """Extaction de données affichable pour chaque trajet en azimut constante"""
    # Création des données
    for az in Elev.fileList:
        az.read_Tot() # contenu du fichier de Coordonnée
        nObsRange = Classe.Elev.linkTransit(az) # Récupère les nObs correspondants
        for nObs in nObsRange: # Itère les nObs des transits
            Transit.instance[nObs].read_Tot() # Extrait le contenu du fichier du Transit
            Transit.instance[nObs].read_point() # average de modTot
        az.pointValue = [Transit.instance[nObs].pointValue for nObs in nObsRange]
        print(az.elev)
        print(az.pointValue)

def Graph_Elev_read():
    """Plot les parcours en azimut constante""" 
    fig, ax = plt.subplots()
    for az in Elev.fileList:
        plt.plot(az.elev, az.pointValue, 'b')
        plt.xlabel('élévation')
    ax.set_yscale('log')
    ax.set_ylim(9e-5, 2e-4)
    plt.show()
    
