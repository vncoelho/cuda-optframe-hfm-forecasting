#!/bin/bash

PS=" ./REED/channel_1.dat
"

SAMPLESTRAININGSET="5000
10000
50000
100000
200000
500000
1000000
"

FORECASTINGHORIZONMIN=" 1
2
5
10
60
360
720
1800
3600
7200
"

for batch in `seq 10`
do
	for probl in $PS
	do
		for fh in $FORECASTINGHORIZONMIN
		do
	
		  for nSamples in $SAMPLESTRAININGSET
		  do 
		     totalNSamples=$(($nSamples + $fh))
		     echo "Resolvendo o problema $probl forecastingHorizon $fh and number of samples $nSamples"		
		    ./Release/cuda-optframe-previsao $probl ./teste $totalNSamples $fh
	       
		  done
	  done
	done
done


