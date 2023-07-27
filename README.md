#  f4klo-calibration  #

Programme de calibration et d'affichage des observations du radiotelescope de
la villette.
f4klo est le nom de la station radiophonique installée par les radioamateurs
de l'association dimention parabole.


#  version Parallel  #

Cette version à été conçu pour permettre aux utilisateurs du telescope
d'afficher des graphiques en coordonnée parallèles, afin d'étudier la coorélation
entre le signal obtenu et les nombreux capteurs de la base de donnée influx.

Dans ce programme, la base de donnée utilisée est: giskard2.hd.free.fr
Cette adresse peut-être modifier dans le fichier influx.sh


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
- pour afficher un graph de Coordonnée parallèle vous utiliserez la fonction
  Graph_Para_Sensor()
  Cette fonction prend en paramètres:
  -  le numéro de l'observation sur lequel vous investiguez
  -  la taille de l'échantillon (nombre de points pour chaque variables)
  -  les capteurs qui vous intéresse en tant que liste de tuples, exemple:
     [('ra','coord','indi'),
      ('dec','coord','indi'),
      ('ext','obj','temp'),
      ('cav','obj','temp')]
- Exemple final:
Graph_Paral_Sensor(258,100,[('ra','coord','indi'),('dec','coord','indi'),('ext','obj','temp'),('cav','obj','temp')])

  Le Résultat sera affiché dans un nouvel onglet de votre navigateur
internet.
  Vous pouvez ensuite déplacer certains éléments graphiques avec la souris.
  Plus d'informations sur: https://plotly.com/python/parallel-coordinates-plot/#parallel-coordinates-plot-with-plotly-express


#  Structure Générale du programe  #
à venir..
