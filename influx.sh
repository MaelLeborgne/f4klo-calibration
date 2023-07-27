#!/bin/bash

echo "nom du script: $0"
echo "date: $1"
echo "duree: $2"
echo "variable: $3"
echo "clef: $4"
echo "branche: $5" # ne fonctionne pas dans la commande ci-dessous
branche=$5

influx -host giskard2.hd.free.fr -database f4klo -precision s -execute "select * from ${branche} where $4 =~ /$3/ and time > '$1' and time < '$1' + $2" -format csv > Sensor_$5_$3_$1.dat

# Comment afficher les séries?
# influx -host giskard2.hd.free.fr
# use f4klo
# show series
