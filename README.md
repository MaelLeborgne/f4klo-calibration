#  f4klo-calibration  #

Programme de calibration et d'affichage des observations du radiotelescope de
la villette.
f4klo est le nom de la station radiophonique installée par les radioamateurs
de l'association dimention parabole.


#  version fin de stage  #

Cette version permet la création de tout les graphiques présentés dans le rapport de stage. 
Dans ce programme, la base de donnée utilisée pour les capteurs est: giskard2.hd.free.fr .Cette adresse peut-être modifiée dans le fichier influx.sh


#  Structure Générale du programe  #
- main.py : Importation du Carnet de Laboratoire, exécution de Classe.py et ouverture aux commandes de Graph.py et Write.py.
- Classe.py : Création des Classes Transit et Elevation qui permettent le stockage des attributs et des méthodes pour l’ensemble des fichiers générés par les captures.
- Graph.py : Création de graphiques qui combinent les résultats de plusieurs captures
- Write.py : Fonctions d’enregistrements de résultats de calculs visant à réduir le temps d’exécution des fonctions de Graph.py .

#  Manuel d'utilisation:  #

Installation:
- Si ce n'est pas déjà fait, faite un git pull
- créer un dossier que vous nomerez Data dans le dossier principale
- placez y les fichier .csv de chaque Observations
- la liste des fichiers d'observation doit être identique à celle de
  CarnetLabo.csv pour permettre le lien entre les paramètre d'observations et
les fichiers d'observation

À chaque utilisation:
- dans un terminal entrez python3 pour ouvrir l'invite de commande de python
- pour initialiser le programme entrez: from main import *
- Exemple: pour afficher un graph de Coordonnée parallèle vous utiliserez la fonction
  Graph.Para_Sensor()
  Cette fonction prend en paramètres:
  -  le numéro de l'observation sur lequel vous investiguez
  -  la taille de l'échantillon (nombre de points pour chaque variables)
  -  les capteurs qui vous intéresse en tant que liste de tuples, exemple:
     [('ra','coord','indi'),
      ('dec','coord','indi'),
      ('ext','obj','temp'),
      ('cav','obj','temp')]
- Exemple final:
Graph.Paral_Sensor(258,100,[('ra','coord','indi'),('dec','coord','indi'),('ext','obj','temp'),('cav','obj','temp')])

  Le Résultat sera affiché dans un nouvel onglet de votre navigateur
internet.
  Vous pouvez ensuite déplacer certains éléments graphiques avec la souris.
  Plus d'informations sur: https://plotly.com/python/parallel-coordinates-plot/#parallel-coordinates-plot-with-plotly-express

# Documentation des fonctions #
Une grande partie des fonctions du module Graph.py ont une documentation print(nomDeLaFonction.\_\_doc\_\_) contenant entre autre un test censé fonctionner (les commande entre parathèse ne sont nécessaire que la première fois). 

