#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print('###   MAIN   ###')

from Classe import carnet
import pandas as pd

# Extraction du carnet de Labo
global CarnetLabo 
CarnetLabo = pd.read_csv('./CarnetLabo.csv')

print('    Extraction du carnet de Labo ... {}'
        .format(len(list(CarnetLabo.index))))

# Création des classes
import Classe 
from Classe import *

# List of DataFrame to list of Dict
Frame = CarnetLabo.T # transposing
Series = [ S for S in list(dict(Frame).values())] # series of attibuts
DicoList = [ dict(S) for S in Series if S['nObs'] != 'XXX'] # series to dict

# Dict create Transit objects
for dico in DicoList:
    Transit.instance.append(Transit(**dico)) 
    # Attention à ne pas changer les entete, car elles sont les nom des attributs

# Series as attributs of transit instances
for transit in Transit.instance:
    transit.s = Series[int(transit.nObs)]

# Cleaned CarnetLabo
CarnetLabo = pd.DataFrame(DicoList)
CarnetLabo.pop('nObs') # replaced by index column
# enrichissement du DataFrame 
CarnetLabo['instance'] = Transit.instance
CarnetLabo['gain'] = CarnetLabo['fileName'].str.split('_').str.get(2)


print('Pour afficher la liste des Observations du carnet de Labo,',
        '\nentrez: Transit.display_Obs_carnet()\n')

# Vérification de la cohérences entre Transit.fileList et CarnetLabo 
# Transit.Verif(CarnetLabo) 

# Fonctions interactives
import Graph
from Graph import *
import Write 

def help():
    '''
       Affiche à l'utilisateur les possibilitées du mode interactif
    '''
    print('\n###  Mode interactif   ###')
    from inspect import getmembers, isfunction

    print('Graph.')
    for (name,obj) in getmembers(Graph):
        if isfunction(obj):
            if obj.__module__ == 'Graph': 
                print('    ',name, '()')

    print('Write.')
    for (name,obj) in getmembers(Write):
        if isfunction(obj):
            if obj.__module__ == 'Write': 
                print('    ',name, '()')

    print('\nPour Afficher la doc d\'une fonction entrez: print(nomFonction.__doc__)')
    print('Pour Afficher les paramètres d\'une Observation n°nObs, entrez:   CarnetLabo.iloc[nObs]')
    print('\nPour afficher cette aide tapez: help()')

help() 

## Début de calibration
#import Correction as Corr
#print('###  Dictionnaire Astres de référence  ###')
#RefJanski = Corr.Ref(1.4204) 
#print(RefJanski)
#RefWatt = Corr.JanskiToWatt(list(RefJanski.values()))
#print(RefWatt)

