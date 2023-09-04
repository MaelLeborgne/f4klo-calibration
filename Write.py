#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import Externe
from math import *
import numpy as np
import statistics as st
import scipy as sp

#import Classe
from Classe import *
from main import *

#-------- Write.py -----------

def pointValue(begin,end):
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

def PF_Spec(CarnetLabo, fileName): #TODO: CarnetLabo autrement que argument
    """
       écrit Tout les spectres des Points Froids 4 3 1
       Dans un fichier csv
TEST:
from main import *
Write.PF_Spec(CarnetLabo,'PF1_dataFrame.csv')
    """

    # Extrait les objet transit tous les point froids du carnet de Labo
    PFobjet = [] 

    # Filtre le CarnetLabo
    # Select by astre name
    astre = CarnetLabo.groupby('astre')
    #PF1 = astre.get_group('PF1')
    #PF3 = astre.get_group('PF3')
    PF4 = astre.get_group('PF4')
    #df = pd.concat([PF1,PF3,PF4]) # single DataFrame
    df = PF4

    # 2ème filtre
    df = df.groupby('gain').get_group('38.0') # gain filter
    print(df)
        
    # storage of class Transit objects
    PFobjet = [Transit.instance[nObs] for nObs in df.index]  

    # initialisation des données de capture
    for PF in PFobjet:
        PF.read_Tot()
        PF.read_Freq()
    
    # Storage of spectrum in CarnetLabo
    liste = [list(transit.modeFreq) for transit in PFobjet]
    keys = [PF.nObs for PF in PFobjet]
    print(keys)
    dico = dict(zip(keys,liste))

    df_PF = pd.DataFrame(dico, index = PFobjet[0].freqScale)

    print(df_PF)
    # Saving
    df_PF.T.to_csv(fileName) # freq in columns


