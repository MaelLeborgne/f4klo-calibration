#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print('###   MAIN   ###')

from Classe import carnet
import pandas as pd

# Extraction du carnet de Labo
global CarnetLabo 
CarnetLabo = pd.read_csv('./CarnetLabo.csv')

#PANDAS CarnetLabo.Read()
print('    Extraction du carnet de Labo ...')#PANDA {}'.format(len(CarnetLabo.DicoList)))

# Création des classes
import Classe 
from Classe import *

# List of DataFrame to list of Dict
Frame = CarnetLabo.T
Series = [ S for S in list(dict(Frame).values())]
DicoList = [ dict(S) for S in Series if S['nObs'] != 'XXX']

# Cleaned CarnetLabo
CarnetLabo = pd.DataFrame(DicoList)
CarnetLabo.pop('nObs')

# Dict create Transit objects
for dico in DicoList:#PANDA CarnetLabo.DicoList:
    Transit.instance.append(Transit(**dico)) 
    # Attention à ne pas changer les entete, car elles sont les nom des attributs

# Series as attributs of transit instances
for transit in Transit.instance:
    transit.s = Series[int(transit.nObs)]

print('Pour afficher la liste des Observations du carnet de Labo,',
        '\nentrez: Transit.display_Obs_carnet()\n')

# Vérification de la cohérences entre Transit.fileList et CarnetLabo 
#PANDAS Transit.Verif(CarnetLabo) 

# Fonctions interactives
import Graph
from Graph import *

def interactif():
    '''
       Affiche à l'utilisateur les possibilitées du mode interactif
    '''
    print('\n###  Mode interactif   ### TODO: afficher la __doc__ ')
    print('Fonctions utilisables:')
    for name in dir(Graph):
        if name[0:5] == 'Graph': # Garde uniquement les fonctions Graph_
            print(name, '()')

    print('\nPour Afficher la doc d\'une fonction entrez: print(Graph_.__doc__)')
    print('Pour Afficher les paramètres d\'une Observation entrez:   print(Transit.Data[nObs])')
    print('\nVous pourrez toujours afficher cette aide en entrant: interactif()')
interactif()

## Début de calibration
#import Correction as Corr
#print('###  Dictionnaire Astres de référence  ###')
#RefJanski = Corr.Ref(1.4204) 
#print(RefJanski)
#RefWatt = Corr.JanskiToWatt(list(RefJanski.values()))
#print(RefWatt)

