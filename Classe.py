from math import *
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime
from datetime import timedelta
import time
import pandas as pd

# my modules
#import Graph # erreur circular import
import Modif

classes = {} # dictionnaire class : objet de MetaData
Unit = [['Secondes', 'Ra', 'Dec'], #TODO: à récupérer dans influx?
      ['Secondes', 'Température (°C)'],
      ['Secondes', 'Température (°C)'],
      ['Secondes', 'Température (°C)']]

def fileToList(file, ponctuation):
    '''
       Convertion d'un fichier d'entrée en Array
    '''
    with open(file.path + file.fileName) as fileID:
        reader = csv.reader(fileID, delimiter=ponctuation)
        csvTable = []
        for line in reader:
            csvTable.append(line) # Chaque line est ajouté à une grande Table 
            file.lenLine = len(line)
            file.lenCol = len(csvTable)
    return csvTable

################################################################################# 
class carnet(): #TODO: Toujours utile?
    def __init__(self):
        self.path = './'
        self.fileName = 'CarnetLabo.csv'
        self.content = []
        self.enTete = []

    def Read(self):
        '''
           Lit le Carnet de Labo 
           et prépare les future objet transit dans une liste de dictionnaires
        '''
        # séparation de l'en-tête et du contenu
        content = fileToList(self, ',')
        self.entete = content[0]
        self.content = content[1:]

        # Chaque ligne devient un dictionnaire d'attributs 
        self.DicoList = [] # liste des dictionnaires d'attributs
        for ligne in self.content: 
            if ligne[0] != 'XXX': # sauf les ligne volontairement ignorées
                self.DicoList.append(dict(zip(self.entete, ligne)))

    def Write(self, att, liste):
        '''
           Écrit dans une copie du carnet de labo
        '''
        if self is CarnetLabo:
            print("Vous ne pouvez pas écrire sur le fichier de référence")
        else:
            # reccupère touts les attributs pour en faire des dictionnaire
            # recompose self.content et self.enTete
            # écrit leur contenu dans un nouveau fichier
            pass #TODO: fonction non terminé

    def Sort(self, **kwargs):
        '''
           Trie et isole les transit selon un dictionnaire de parmètre filtrants
           les paramètre s'ajoutent et ne se contredise pas
           le résultat est une liste d'objet transit
        '''
        new_list = []
        def simple_comparaison(dico, key, value, test):
            '''
               cherche une clef et sa valeur dans les dictionnnaire d'entré
               renvoi True ou false en fonctin du résultat du test
            '''
            if dico[key] == value and test == True: 
                print('OUI!', key, '=', dico[key])
                test = True
            else:
                print('NON! ', key, '=', dico[key], type(dico[key]))
                test = False
            return test

        # Itération des Dictionnaire de CarnetLabo
        for transit in Transit.instance:
            dico = transit.__dict__
            # kwargs est un dictionnaire d'attributs filtrants
            test = True # devient false si dico ne repond pas à toute les valueurs filtrante
            for key in kwargs.keys(): # itération sur les clés filtrantes 
                value = kwargs[key] # réccupération des valeurs filtrantes

                # si plusieur valeur pour une même clef
                if 'tuple' in str(type(value)):
                    for V in value:
                        test = simple_comparaison(dico, key, V, True)
                else: # par défault
                    test = simple_comparaison(dico, key, value, test)

            # Résultat du test
            if test == True:
                print('transit append to new_list')
                new_list.append(transit)

            print('taille de la nouvelle liste: ', len(new_list), '\n')

        return new_list
        
################################################################################# 
class MetaData(type):
    """
       Chaque type de fichiers est rangé dans une classe de MetaData
       Chaque classe contient une liste d'objet remplie par la fonction RgmtObs()
       Les objets des classes de MetaData ont au minimum:
       - un nom de fichier
       - un chemin d'accès
    """
    def __init__(cls, nom, bases, dict):
        """
           Est Exécuté à chaque fois qu'une classe crée dans la métaclasse
           cls : objet contenant la classe
           nom : contient le nom de la classe
           
           Initialisation des attributs de classe
        """
        type.__init__(cls, nom, bases, dict) # créer la classe
        classes[nom] = cls # rangement dans le dictionnaire classes
        
        # Atributs de classe
        cls.name = nom
        cls.path=''
        cls.fileList = []
        cls.instance = []

        # On rempli les attributs par classe
        if nom == 'Transit':
            cls.path  = './Data/'
            if os.path.exists(cls.path) == False:
                os.system("mkdir {}".format(cls.path))
                print("Un répertoire à été créer à", cls.path)
                print("Afin de stoquer les données")
            files = os.popen("cd {} && ls leborgne*.0V2.csv".format(cls.path))
            cls.fileList = cls.Rgmt(files) # Rempli par les fichiers du path contenu dans files 
            cls.instance = [] # Rempli par le carnet de Labo
        elif nom == 'Elev':
            cls.path  = './Data/'
            files = os.popen("cd {} && ls Coord*".format(cls.path))
            cls.fileList = cls.Rgmt(files)
        elif nom == 'Sensor':
            cls.path ='./Data/' 
            files = os.popen("cd {} && ls *.dat".format(cls.path))
            cls.fileList = cls.Rgmt(files)
        else:
            print('\n### La class ', nom, ' n\'a pas d\'attribu .path')

    def Rgmt(cls, liste):
        '''
           Rangement des fichiers de données dans un liste nommé fileList 
           à partir de la \'liste\' en argument,
           fournie par os.popen(\"cd...
        '''
        fileList = [] # Liste des données réccupérées

        # On liste les fichiers de données 
        for File in liste: # ajout des fichier dans la liste fileList
            File = File[:-1] # on retire le \n de la commande syst
            fileList.append(File)

        # Transformation des nom de fichier en objet de la classe 
        for nObs in range(len(fileList)):
            name = fileList[nObs] # on garde le nom
            fileList[nObs] = cls.__new__(cls) # on les remplace par des objets 
            # on leur transmet leurs paramètres généraux
            fileList[nObs].fileName = name
            fileList[nObs].path = cls.path
            fileList[nObs].nObs = nObs
            fileList[nObs].__init__()

        print('\n    Identification des fichiers de la class {} ... {}'.format(cls.name, len(fileList)))
        print('Pour afficher la liste de ces objets, entrez:   print({}.fileList)'.format(cls.name))
        return fileList   # list

    def Verif(cls, CarnetLabo):
        '''
           Vérifi la cohérence entre le carnet de Labo et les fichier de ../Data
           ne s'utilise pas avec la classe Sensor
        '''
        agreement = False # Change en True si les liste sont en accord
        fileLink = ''
        laboLink = ''
        nObs = ''
        for Obs in cls.fileList:
            fileLink = Obs.fileName.split('_')[1] # comparaison par dates
            laboLink = CarnetLabo.DicoList[Obs.nObs]['fileName'].split('_')[1]
            nObs = Obs.nObs
            if fileLink == laboLink:
                agreement = True

        # Check de sortie de boucle
        if cls.fileList == []:
            print("CarnetLabo.Verif(): Aucun fichier de la classe {} dans {}"
                    .format(cls.name, cls.path))
        else:
            if agreement == True:
                print('\nCarnetLabo.Verif():', 
                       'La liste des fichiers a été vérifié avec succès')
            else:
                raise NameError('CarnetLabo.Verif():', 
                        'Le carnet de labo contredit le rangement automatique\n',
                        'Class.fileList[{}]: {} \n'.format(nObs, fileLink),
                        'CarnetLabo[{}]: {}'.format(nObs, laboLink) ) 

    def display_Obs_carnet(cls): #TODO: tester la fonction
        for transit in cls.instance: 
            print('{}: {}'.format(transit.nObs, transit.fileName))

################################################################################ 
class Transit(metaclass=MetaData):
    '''
       Chaque fichier V2.csv est un instance de cette classe
    '''
    def __init__(self, **dico):
        self.__dict__.update(dico)

        #  Identification les paramètres dans le nom du fichier 
        file = self.fileName[0:-6]# on enlève le sufixe
        Param = file.split('_') # tout les paramètre

        self.user = Param[0]
        date = str(Param[1])
        self.date = "{}-{}-{}T{}:{}:{}".format(date[0:4], date[4:6], date[6:8],
                date[8:10], date[10:12], date[12:14])
        self.gain = float(Param[2])
        self.freqC = float(Param[3]) # en MHz
        self.BP = float(Param[4]) # en MHz
        self.freqE = float(Param[5]) # en MHz
        self.duree = round(float(Param[6])) # en secondes

        # Data of capture
        self.modeTot = [] #Total
        self.modeTemp = [] #Temporel
        self.modeFreq = [] #Fréquentiel
        self.freqScale = []
        self.Spectrum = []

        # Sensors informations
        self.sensorFrame = pd.DataFrame() # filling with import_Sensor()
        

    def read_Tot(self):
        '''
           Récupération Total des données dans self.modeTot
        '''
        if self.modeTot != []:
            print("###  Transit.instance[{}]: self.modTot a déjà été extraie".format(self.nObs))
        else:
            # Récupération du contenu brut 
            content = fileToList(self, ',')
            
            t = time.clock_gettime(time.CLOCK_REALTIME) # begin chrono
            # Formate le contenu
            self.lenLine = self.lenLine - 12 # -12 pour le taille de l'en-tête
            modeTotSize = (self.lenCol, self.lenLine) 
            self.modeTot = np.zeros(modeTotSize, dtype=float) # réservation de l'emplacement 
            i = 0 # compteur de lignes
            for line in content:
                self.modeTot[i] = line[6:-6]  # Suppression de l'entête 
                i += 1

            d = time.clock_gettime(time.CLOCK_REALTIME) - t # stop chrono
            print('###  Transit.instance[{}]:'.format(self.nObs),
                    'Extraction de self.modeTot  ###',
                    '   temps écoulé: ' + str(d) )
    
            # Temp unix du début de la capture
            self.t0 = os.popen("date -d '{} EDT' +%s".format(self.date)) 
            self.t0 = next(iter(self.t0)) # Unwrap os._wrap_close object


    def read_Temp(self):
        '''
           Récupération Temporelle dans self.modeTemp
        '''
        if self.modeTemp != []:
            print("###  Transit.fileList[{}]: self.modeTemp a déjà été extraie".format(self.nObs))
        else:
            t = time.clock_gettime(time.CLOCK_REALTIME) # begin chrono
            self.modeTemp = []
            for i in range(self.lenCol):
                self.modeTemp.append(np.average(self.modeTot[i,:]))
            self.modeTemp = np.array(self.modeTemp)
            d = time.clock_gettime(time.CLOCK_REALTIME) - t # stop chrono
            print('###  Transit.fileList[{}]:'.format(self.nObs),
                    'Extraction de self.modeTemp  ###',
                    '   temps écoulé: ' + str(d) )

            # Ordonnée Temporelle
            self.timeScale = np.linspace( 0, self.duree, len(self.modeTemp))
    
    def read_Freq(self):
        '''
           Récupération fréquentielle dans self.modeTemp
        '''
        if self.modeFreq != []:
            print('###  Transit.fileList[{}]:'.format(self.nObs),
                    'self.modeFreq a déjà été extraie')
        else:
            t = time.clock_gettime(time.CLOCK_REALTIME) # begin chrono

            # Average on  frequency and storage in a list
            self.modeFreq = []
            for j in range(self.lenLine):
                self.modeFreq.append(np.average(self.modeTot[:,j]))

            self.modeFreq = np.array(self.modeFreq) # to array

            d = time.clock_gettime(time.CLOCK_REALTIME) - t # stop chrono
            print('###  Transit.fileList[{}]:'.format(self.nObs),
                    'Extraction de self.modeFreq  ###',
                    '   temps écoulé: ' +  str(d) )

            # Ordonnée fréquentielle
            minF = self.freqC - (1/2)*self.BP
            maxF = self.freqC + (1/2)*self.BP
            self.freqScale = np.linspace( minF, maxF, len(self.modeFreq))

            dico ={
                    'minF': self.freqC - (1/2)*self.BP,
                    'maxF': self.freqC + (1/2)*self.BP,
                    'ech': len(self.modeFreq), # échantillonnage
                    'freq': self.freqScale,
                    'values': self.modeFreq
                    }

            # Storage in a Spectrum Object
            spec = Spectrum(**dico)
            Spectrum.instance.append(spec)
            self.spec = spec

    def read_point(self):
        '''
           Moyenne de toutes les valeurs du transit
        '''
        self.pointValue = np.average(np.average(self.modeTot)) 

    def graph_Temp(self, graph):
        '''
           Rempli le graph donné avec les données de l'objet self en mode Temporel
        '''
        T = self.modeTemp
        time = self.timeScale
        graph.plot( time, T, marker='+')
        graph.set_title("{}".format(self.astre), fontsize=18)
        graph.set_xlabel('Temps (en secondes)',fontsize='large')
        graph.set_ylabel('Puissance du signal',fontsize='large')
        graph.grid(True)
        plt.xticks(np.arange(0,self.duree,600))

    def graph_Freq(self, graph):
        '''
           Rempli le graph donné avec les données de l'objet self en mode Fréquentiel
        '''
        F = self.modeFreq
        freq = self.freqScale 
        graph.plot(freq, F, label=self.nObs, color='blue')
        graph.set_title('{}'.format(self.astre), fontsize=18)
        graph.set_xlabel('Fréquence (en MHz)',fontsize='large')
        graph.set_ylabel('Puissance du signal',fontsize='large')
        graph.grid(True)
        minF = self.freqScale[0]
        maxF = self.freqScale[-1]
        plt.xticks(np.arange(minF, maxF+0.00001, 0.2))

    def modif_Tot(self):
        '''
           Crée un copie de modeTot
           Et fait des modification en vu de la calibration
#TEST:
from main import *
self = Transit.instance[467]
self.read_Tot()
self.modif_Tot()
        '''
        import Modif
        modifTot = pd.DataFrame(self.modeTot)
        modifTot = Modif.by_PF_Spectrum(self, modifTot)
        modifFreq = modifTot.mean()
        return modifFreq

    def temp_pred(self):
        self.import_Sensor('ext','obj','temp','f4klo')
        # pick values
        T = self.sensorFrame.T['value']['ext']
        T = np.average(T)
        [x0,x1,x2] = Graph.PF_DC_Temp()
        return x0 + x1*T + x2*T**2

    def import_Sensor(self, sensorName, key, branch, database):
        '''
           Importe les data du capteur sensorName indiqué
           Puis les stoque dans un fichier
        '''
        fileName = "{}Sensor_{}_{}_{}Z.dat".format(Transit.path, branch, sensorName, self.date)
        print(fileName)
        if os.path.exists(fileName):
            print("le fichier existe déjà")
        else: 
            print("le fichier n'existe pas encore")
            duree = ''
            if database == 'weather':
                duree = '43200' # 1 mesures toute les 12h (43200) pour weather
            else:
                duree = self.duree
            # Commande influx
            initialPath = next(iter(os.popen('pwd')))[0:-1]
            os.system("cd {} && {}".format(Transit.path, initialPath) +
                    "/influx.sh {}Z {}s {} {} {} {} && " # duree en seconde
                    .format(self.date, duree, sensorName, key, branch, database) +
                    "echo \'le fichier vient d être créer!\'"
                    )

        # Attributs
        sensor_reader = pd.read_csv(fileName)
        value = [] # Data to retrieve in the table
        name = ''
        if database == 'weather': # Specification from database
            value = list(sensor_reader['humid'])
            name = sensorName + ' humid'
        else: # f4klo
            value = list(sensor_reader['value'])
            name = branch + ' ' + sensorName
        sensor_att = {
                'name':name,
                'key':key,
                'branch':branch,
                'fileName':fileName,
                'time':list(sensor_reader['time']),
                'value':value
        }
        # Ajout d'une ligne à self.sensorFrame
        self.sensorFrame[sensorName] = pd.Series(sensor_att)

    def resize(self, x, y, newSize): #TODO: return xnew, turn to general fonction
        '''
           Change la taille (newSize) de l'échantillonnage du "sensor"
           grace à une interpolation
        '''
        deg = 0 # degré 0 pour interp linéaire
        import scipy.interpolate as itp

        # initial data
        print(len(x))
        print(len(y))
        # parameters for ynew
        tck = itp.splrep(x, y, s=deg, k=2) # k doit être > que nbr de point
        # new vectors
        xnew = np.arange(min(x), max(x), (max(x)-min(x))/newSize) # arange as we want
        ynew = itp.BSpline(*tck)(xnew) # fit to xnew

        # plot
        #plt.plot(xnew, ynew, '-', label='interpolation') # interpol
        #plt.plot(x,y,'+', label='Data') # data points
        #plt.show()

        print('taille de l\'échantillon: ', len(ynew))
        return ynew

# inutilisable depuis que les instance sont dans le panda
#    def __str__(self):
#        '''
#           Affichage des paramètre dans la console
#        '''
#        D = self.__dict__
#        K = list(D.keys())
#        V = list(D.values())
#        disp = '\nTransit n°{}'.format(self.nObs) + "--"*20 + '\n'
#        for i in range(len(K)-4):
#            disp = disp + '{0:^20} | {1:^20}'.format(K[i],V[i]) + '\n'
#        return disp

############################################################################### 
class Sensor(metaclass=MetaData):
    def __init__(self, **dico):
        self.sensorName = ''
        self.key = ''
        self.branch = ''
        self.unit = ''
        self.__dict__.update(**dico)
        # TODO: est-elle utile?

############################################################################### 
class Elev(metaclass = MetaData):
    ''' 
       Chaque fichier Coordonnées est une instance de cette classe
    '''
    def __init__(self):
        self.date = self.fileName[12:-3] # extraction de la date dans le nom 
        self.nObsRange = []
        self.modeTot = []
        self.azimut = []
        self.elev = []
        self.pointValue = []

    def read_Tot(self):
        '''
           Extraction des paramètre dans les fichiers Cooordonnée..csv
        '''
        # Récupération Totale dans self.modeTot 
        if self.modeTot == []:
            print('\n###  Extraction des données de {}  ###'.format(self.fileName))
            content = fileToList(self, ',')
            content = content[2:] # suppression des commentaires
            self.lenCol = len(content)
            self.lenLine = len(content[0])
            self.modeTot = np.array(content) # array
            print(self.modeTot)
            
            if self.lenLine == 6:
                # Réccupération des Coordonnées de chaque point
                for line in range(self.lenCol):
                    self.azimut.append(float(self.modeTot[line, 4]))
                    self.elev.append(float(self.modeTot[line, 5]))
                print('Azimut: ', self.azimut)
                print('Élévation: ', self.elev)
            else:
                print('Le fichiers est érroné ou n\'est pas un Elev')
        else:
            print('self.modeTot est déjà rempli')

    def linkTransit(self):
        '''
           Recherche des numéros d'Observations 
           qui correspondent aux fichiers Coordonnées
        '''
        nObsRange = []
        fileList_copie = list(Transit.fileList) # créer une copie de la liste originale
        # parcour les ligne du fichier Coordonnée
        for line in range(len(self.elev)):
            # parcour les transit de Transit.fileList
            for Obs in Transit.fileList: 
                obsDate = datetime.strptime(Obs.date, "%Y-%m-%dT%H:%M:%S")
                coordDate = datetime.strptime(self.modeTot[line,0], "%Y-%m-%dT%H:%M:%S")
                delta = timedelta(minutes=5)
                if obsDate - coordDate >= delta : # si l'ecart est infèrieure à 5min 
                    nObsRange.append(Obs.nObs)
                    fileList_copie.pop(Transit.fileList.index(Obs))
                    break
            # on reprend à la ligne suivante dans Coordonnée.csv
        # Retour des nObsRange et des pointValue correspondants
        print(nObsRange)
        return nObsRange

    def linkTransit2(self):
        '''
           Essai de link avec CarnetLabo.csv
        '''
        nObsRange = []
        return nObsrange #TODO: fonction à finir pour plus de fiabilité

################################################################################## 
#TODO: class Spectrum():
class Spectrum(metaclass=MetaData):
    def __init__(self,**dico):
        self.minF = ''
        self.maxF = ''
        self.ech = ''
        self.freq = []
        self.values = []
        self.__dict__.update(**dico)

        pass







# Ancien contenu de la Classe Sensor
#
#            # convertion du temps unix
#            self.date = os.popen("date -u --date=\'@{}\' \"+%d/%m/%y %T\"".format(self.modeTot[0,0]))
#            self.date = next(iter(self.date)) # on appel le premier élement de l'objet renvoyé par os.popen
#            self.date = self.date[:-1] # on enlève le \n de la commande système
#            print('Convertion du temp unix: ',self.date)
#            print(self)
#
#    def graph(self, graph):
#        '''
#           rempli le graph donné avec les données de l'objet
#        '''
#        # temps depuis le début
#        time = self.modeTot[:,0] - self.modeTot[0,0]
#        # pour chaque chaque colonne après celle du temps 
#        for i in range(self.lenLine-1):
#            S = self.modeTot[:,1+i]
#            graph.plot(time, S, '+', label=self.fileName)
#            graph.plot(time, S, '--', color='brown')
#        graph.set_title('{} depuis le {} UTC'.format(self.fileName, self.date), fontsize=16)
#        graph.set_xlabel('Temps (en secondes)')
#        graph.legend()
#
#    def __str__(self):
#        '''
#           Affichage des paramètres dans la console
#        '''
#        D = self.__dict__
#        K = list(D.keys())
#        V = list(D.values())
#        disp = 'Sensor n°{}'.format(self.nObs) + "--"*20 + '\n'
#        for i in range(len(K)-3):
#            disp = disp + str('{0:^20} | {1:^20}'.format(K[i],V[i])) + '\n'
#        return disp


print("\n    Liste des classes de MetaData:\n", classes, '\n')
