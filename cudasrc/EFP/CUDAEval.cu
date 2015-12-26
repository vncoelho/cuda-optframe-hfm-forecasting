/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#include "CUDAEval.cuh"

__global__ void kernelForecasts(int nThreads, CUDARep cudarep, float* dForecastings, int* dfSize, int maxLag, int stepsAhead, const int aprox, float* predicted)
//vector<double> returnForecasts(const RepEFP& rep, const vector<vector<double> >& vForecastings, int begin, int maxLag, int stepsAhead, const int aprox)
{
	int sizeSP = cudarep.size; //rep.singleIndex.size();

	int begin = threadIdx.x;
	if (begin >= nThreads)
		return;

	int offset = begin * nThreads;

	//int nForTargetFile = dfSize[0]; //vForecastings[0].size();

	//vector<double> predicteds;

	for (int pa = 0; pa < stepsAhead; pa++)			// passos a frente
	//for (int pa = 0; ((pa < stepsAhead) && (pa + begin < nForTargetFile)); pa++) //auxiliar loop for steps ahead
	{
		//vector<double> fuzzyWeights;
		//forecasting estimation
		double estimation = 0;
		double greaterAccepeted = 0;
		double lowerAccepted = 0;

		for (int nSP = 0; nSP < sizeSP; nSP++)
		{
			int file = cudarep.d_singleIndex[nSP].v1;		//rep.singleIndex[nSP].first;
			int K = cudarep.d_singleIndex[nSP].v2;		//rep.singleIndex[nSP].second;

			// BEGIN GET K VALUE
			// getKValue(K, file, begin, pa, vForecastings, predicteds);
			double singleValue;
			if ((pa >= K) && (file == 0) && (K > 0))
			{
				singleValue = predicted[offset + pa - K];
			}
			else
			{
				if ((begin + pa - K) < dfSize[file])
					singleValue = dForecastings[begin + pa - K]; // FILE ALWAYS ZERO!
				else
					singleValue = 0;
			}
			// END GET K VALUE

			double ruleGreater = cudarep.d_singleFuzzyRS[nSP].GREATER; //rep.singleFuzzyRS[nSP][GREATER];
			double greaterWeight = cudarep.d_singleFuzzyRS[nSP].GREATER_WEIGHT; //rep.singleFuzzyRS[nSP][GREATER_WEIGHT];
			double ruleLower = cudarep.d_singleFuzzyRS[nSP].LOWER; //rep.singleFuzzyRS[nSP][LOWER];
			double lowerWeight = cudarep.d_singleFuzzyRS[nSP].LOWER_WEIGHT; //rep.singleFuzzyRS[nSP][LOWER_WEIGHT];
			double ruleEpsilon = cudarep.d_singleFuzzyRS[nSP].EPSILON; //rep.singleFuzzyRS[nSP][EPSILON];
			FuzzyFunction repFuzzyPertinenceFunc = FuzzyFunction(cudarep.d_singleFuzzyRS[nSP].PERTINENCEFUNC); //rep.singleFuzzyRS[nSP][PERTINENCEFUNC]

			defuzzification(ruleGreater, greaterWeight, ruleLower, lowerWeight, ruleEpsilon, repFuzzyPertinenceFunc, singleValue, &estimation, &greaterAccepeted, &lowerAccepted);

			//fuzzyWeights.push_back(value);
		}

		//cout << "EstimationAntes:" << estimation << endl;
		double accepted = greaterAccepeted + lowerAccepted;
		if (accepted > 0)
			estimation /= accepted;

		//				Remove this for other forecast problem -- rain forecast
		//				if (estimation < 0)
		//					estimation = 0;

		//predicteds.push_back(estimation);
		predicted[offset + pa] = estimation;

	} //End of current iteration ... steps head
}

