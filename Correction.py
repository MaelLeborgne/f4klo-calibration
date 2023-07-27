'''Correction.py'''
from math import *

def ModifBy_PF_Spectrum(modifTot):
    '''Soustrait le spectre des points froid aux données d'entrée'''
    # Calcul de la mediane des spectres
    # Soustrait la mediane à modifTot
    return modifTot

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
