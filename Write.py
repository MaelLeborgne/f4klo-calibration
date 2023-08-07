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

def PF_Spec(CarnetLabo):
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
    df_PF = pd.DataFrame(
            [pd.Series(transit.modeFreq) for transit in PFobjet], 
            index = [PF.nObs for PF in PFobjet],
            columns = PFobjet[0].freqScale)

    print(df_PF)
    # Saving
    df_PF.to_csv('PF_dataFrame.csv')


