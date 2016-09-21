#!/bin/bash

PS=" ./REED/channel_1.dat
     ./REED/channel_13.dat
     ./REED/channel_17.dat
     ./REED/channel_9.dat
"

GRANULARITYMIN=" 15
30
60
"

FORECASTINGHORIZONMIN=" 1440
720
360
180
60
30
15
"

for batch in `seq 5`
do
	for probl in $PS
	do
		for fh in $FORECASTINGHORIZONMIN
		do
	
		  for i in `seq 1`
		  do 
		     echo "Resolvendo o problema $probl, batch $i, forecasting Horizon $fh, granularityMin 15"		
		    ./Release/cuda-optframe-previsao $probl ./teste  15 $fh 1

		     echo "Resolvendo o problema $probl, batch $i, forecasting Horizon $fh, granularityMin 30"		
		    ./Release/cuda-optframe-previsao $probl ./teste  30 $fh 1


		     echo "Resolvendo o problema $probl, batch $i, forecasting Horizon $fh, granularityMin 60"		
		    ./Release/cuda-optframe-previsao $probl ./teste 60 $fh 1

		  done
	  done
	done
done

