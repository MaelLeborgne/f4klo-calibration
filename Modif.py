'''Modif.py'''
from math import *
import pandas as pd
import matplotlib.pyplot as plt

def by_PF_Spectrum(transit, modifTot):
    '''Soustrait le spectre des points froid aux données d'entrée'''
    # mediane des spectres
    spec_med = pd.read_csv('spec_med.csv',index_col=0) # freq in index
    # normalization
    spec_med = spec_med / spec_med.mean()
    # resize
    spec_med = transit.resize(list(spec_med.index),list(spec_med['0']),8191)
    spec_med = pd.Series(spec_med)

    # Prediction of quantity of signal to reemove
    pred_moy_PF = transit.temp_pred()
    # Soustrait la mediane à modifTot
    newTot = modifTot - spec_med * pred_moy_PF
    return newTot

def Ref(f):
    '''
       Renvoi un disctionnaire des astres de références et leur Brillance 
       en fonction des la fréquence f
       D'après le document Rec UIT-R S.733-1
    '''
    # Densité spectrale de Puissance Surfacique de l'étoile sur la terre à la fréquence f 
    Psf={ 
            'Cas_A': 1067*10**(-26)*(f/4)**(-0.792) ,
            'Tau_A': 679*10**(-26)*(f/4)**(-0.287)
    }
    return Psf

def JanskiToWatt(values):
    '''Conversition des Janski en Watt'''
    Diametre = 10 # m
    Surface = pi*Diametre**2  # m^2
    for i in range(len(values)):
        values[i]= values[i] *Surface *10**6
    return values

def Compare(capture, ref):
    '''renvoi la différence delta entre la capture et la référence'''
    return delta
