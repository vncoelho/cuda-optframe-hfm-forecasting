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

static const int WORK_SIZE = 256;

vector<double> returnForecasts(const RepEFP& rep, const vector<vector<double> >& vForecastings, int begin, int maxLag, int stepsAhead, const int aprox)
{
	int sizeSP = rep.singleIndex.size();

	int nForTargetFile = vForecastings[0].size();

	vector<double> predicteds;

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
			int file = rep.singleIndex[nSP].first;
			int K = rep.singleIndex[nSP].second;

			// BEGIN GET K VALUE
			// getKValue(K, file, begin, pa, vForecastings, predicteds);
			double singleValue;
			if ((pa >= K) && (file == 0) && (K > 0))
			{
				singleValue = predicteds[pa - K];
			}
			else
			{
				if ((begin + pa - K) < vForecastings[file].size())
					singleValue = vForecastings[file][begin + pa - K];
				else
					singleValue = 0;
			}
			// END GET K VALUE

			double ruleGreater = rep.singleFuzzyRS[nSP][GREATER];
			double greaterWeight = rep.singleFuzzyRS[nSP][GREATER_WEIGHT];
			double ruleLower = rep.singleFuzzyRS[nSP][LOWER];
			double lowerWeight = rep.singleFuzzyRS[nSP][LOWER_WEIGHT];
			double ruleEpsilon = rep.singleFuzzyRS[nSP][EPSILON];
			FuzzyFunction repFuzzyPertinenceFunc = FuzzyFunction(rep.singleFuzzyRS[nSP][PERTINENCEFUNC]);

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

		predicteds.push_back(estimation);

	} //End of current iteration ... steps head

	return predicteds;
}

vector<double> returnTrainingSetForecasts(const RepEFP& rep, const vector<vector<double> >& vForecastings, int maxLag, int stepsAhead, const int aprox)
{
	int nForTargetFile = vForecastings[0].size();
	int nSamples = nForTargetFile - maxLag;

	vector<double> allForecasts;

	for (int i = maxLag; i < nForTargetFile; i += stepsAhead) // main loop that varries all the time series
	{
		vector<double> predicteds = returnForecasts(rep, vForecastings, i, maxLag, stepsAhead, aprox);

		for (int f = 0; f < predicteds.size(); f++)
			allForecasts.push_back(predicteds[f]);
	}

	//TODO do it in a better style
	if (allForecasts.size() > nSamples)
	{
		int nExtraForecasts = allForecasts.size() - nSamples;
		allForecasts.erase(allForecasts.begin() + allForecasts.size() - nExtraForecasts, allForecasts.end());
	}

	return allForecasts;
}

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */

__host__ __device__ unsigned int bitreverse1(unsigned int number)
{
	number = ((0xf0f0f0f0 & number) >> 4) | ((0x0f0f0f0f & number) << 4);
	number = ((0xcccccccc & number) >> 2) | ((0x33333333 & number) << 2);
	number = ((0xaaaaaaaa & number) >> 1) | ((0x55555555 & number) << 1);
	return number;
}

/**
 * CUDA kernel function that reverses the order of bits in each element of the array.
 */
__global__ void bitreverse1(void *data)
{
	unsigned int *idata = (unsigned int*) data;
	idata[threadIdx.x] = bitreverse1(idata[threadIdx.x]);
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int testx(void)
{
	void *d = NULL;
	int i;
	unsigned int idata[WORK_SIZE], odata[WORK_SIZE];

	for (i = 0; i < WORK_SIZE; i++)
		idata[i] = (unsigned int) i;

	CUDA_CHECK_RETURN(cudaMalloc((void** ) &d, sizeof(int) * WORK_SIZE));
	CUDA_CHECK_RETURN(cudaMemcpy(d, idata, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice));

	bitreverse1<<<1, WORK_SIZE, WORK_SIZE * sizeof(int)>>>(d);

	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaMemcpy(odata, d, sizeof(int) * WORK_SIZE, cudaMemcpyDeviceToHost));

	for (i = 0; i < WORK_SIZE; i++)
		printf("Input value: %u, device output: %u, host output: %u\n", idata[i], odata[i], bitreverse1(idata[i]));

	CUDA_CHECK_RETURN(cudaFree((void* ) d));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	return 0;
}
