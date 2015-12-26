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



__device__ int getKValue(const int K, const int file, const int i, const int pa, const vector<vector<double> >& vForecastings, const vector<double>& predicteds)
{
	double value = 0;

	if ((pa >= K) && (file == 0) && (K > 0))
	{
		value = predicteds[pa - K];
	}
	else
	{
		if ((i + pa - K) < vForecastings[file].size())
			value = vForecastings[file][i + pa - K];
		else
			value = 0;
	}

	return value;
}


__device__ void defuzzification(double ruleGreater, double greaterWeight, double ruleLower, double lowerWeight, double ruleEpsilon, FuzzyFunction fuzzyFunc, double value, double& estimation, double& greaterAccepeted, double& lowerAccepted)
{

	if (fuzzyFunc == Heavisde)
	{
		if (value > ruleGreater)
		{
			estimation += greaterWeight;
			greaterAccepeted += 1;
		}

		if (value < ruleLower)
		{
			estimation += lowerWeight;
			lowerAccepted += 1;
		}

	}

	double epsilon = ruleEpsilon;
	//Trapezoid Function
	if (fuzzyFunc == Trapezoid)
	{
		double est = 0;
		double a = ruleGreater;
		double mu = 0;
		if (value <= (a - epsilon))
			mu = 0;
		if (value > a)
			mu = 1;
		if ((value > (a - epsilon)) && value <= a)
		{
			double K1 = 1 / epsilon;
			double K2 = 1 - a * K1;
			mu = value * K1 + K2;
		}

		est = greaterWeight * mu;
		estimation += est;

		greaterAccepeted += mu;
	}

	if (fuzzyFunc == Trapezoid)
	{
		double b = ruleLower;
		double est = 0;
		double mu = 0;
		if (value >= (b + epsilon))
			mu = 0;
		if (value < b)
			mu = 1;
		if (value >= b && value < b + epsilon)
		{
			double K1 = 1 / epsilon;
			double K2 = 1 - b * K1;
			mu = value * K1 + K2;
		}
		est = lowerWeight * mu;
		estimation += est;

		lowerAccepted += mu;
	}

}


vector<double> returnForecasts(const RepEFP& rep, const vector<vector<double> >& vForecastings, int begin, int maxLag, int stepsAhead, const int aprox)
{
	int sizeSP = rep.singleIndex.size();
	int sizeMP = rep.averageIndex.size();
	int sizeDP = rep.derivativeIndex.size();

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

			double singleValue = getKValue(K, file, begin, pa, vForecastings, predicteds);

			double ruleGreater = rep.singleFuzzyRS[nSP][GREATER];
			double greaterWeight = rep.singleFuzzyRS[nSP][GREATER_WEIGHT];
			double ruleLower = rep.singleFuzzyRS[nSP][LOWER];
			double lowerWeight = rep.singleFuzzyRS[nSP][LOWER_WEIGHT];
			double ruleEpsilon = rep.singleFuzzyRS[nSP][EPSILON];
			FuzzyFunction repFuzzyPertinenceFunc = FuzzyFunction(rep.singleFuzzyRS[nSP][PERTINENCEFUNC]);

			defuzzification(ruleGreater, greaterWeight, ruleLower, lowerWeight, ruleEpsilon, repFuzzyPertinenceFunc, singleValue, estimation, greaterAccepeted, lowerAccepted);

			//fuzzyWeights.push_back(value);
		}

		for (int nMP = 0; nMP < sizeMP; nMP++)
		{
			vector < pair<int, int> > meansK = rep.averageIndex[nMP];
			int nAveragePoints = meansK.size();

			double mean = 0;
			for (int mK = 0; mK < nAveragePoints; mK++)
			{
				int file = meansK[mK].first;
				int K = meansK[mK].second;

				mean += getKValue(K, file, begin, pa, vForecastings, predicteds);
			}

			mean = mean / nAveragePoints;

			double ruleGreater = rep.averageFuzzyRS[nMP][GREATER];
			double greaterWeight = rep.averageFuzzyRS[nMP][GREATER_WEIGHT];
			double ruleLower = rep.averageFuzzyRS[nMP][LOWER];
			double lowerWeight = rep.averageFuzzyRS[nMP][LOWER_WEIGHT];
			double ruleEpsilon = rep.averageFuzzyRS[nMP][EPSILON];
			FuzzyFunction repFuzzyPertinenceFunc = FuzzyFunction(rep.averageFuzzyRS[nMP][PERTINENCEFUNC]);

			defuzzification(ruleGreater, greaterWeight, ruleLower, lowerWeight, ruleEpsilon, repFuzzyPertinenceFunc, mean, estimation, greaterAccepeted, lowerAccepted);

			//fuzzyWeights.push_back(mean);
		}

		for (int nDP = 0; nDP < sizeDP; nDP++)
		{
			vector < pair<int, int> > derivateK = rep.derivativeIndex[nDP];

			double d = 0;
			for (int dK = 0; dK < derivateK.size(); dK++)
			{
				int file = derivateK[dK].first;
				int K = derivateK[dK].second;

				double value = getKValue(K, file, begin, pa, vForecastings, predicteds);

				if (dK == 0)
					d += value;
				else
					d -= value;
			}

			//fuzzyWeights.push_back(d);

			double ruleGreater = rep.derivativeFuzzyRS[nDP][GREATER];
			double greaterWeight = rep.derivativeFuzzyRS[nDP][GREATER_WEIGHT];
			double ruleLower = rep.derivativeFuzzyRS[nDP][LOWER];
			double lowerWeight = rep.derivativeFuzzyRS[nDP][LOWER_WEIGHT];
			double ruleEpsilon = rep.derivativeFuzzyRS[nDP][EPSILON];
			FuzzyFunction repFuzzyPertinenceFunc = FuzzyFunction(rep.derivativeFuzzyRS[nDP][PERTINENCEFUNC]);

			defuzzification(ruleGreater, greaterWeight, ruleLower, lowerWeight, ruleEpsilon, repFuzzyPertinenceFunc, d, estimation, greaterAccepeted, lowerAccepted);
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


__host__ __device__ unsigned int bitreverse1(unsigned int number) {
	number = ((0xf0f0f0f0 & number) >> 4) | ((0x0f0f0f0f & number) << 4);
	number = ((0xcccccccc & number) >> 2) | ((0x33333333 & number) << 2);
	number = ((0xaaaaaaaa & number) >> 1) | ((0x55555555 & number) << 1);
	return number;
}

/**
 * CUDA kernel function that reverses the order of bits in each element of the array.
 */
__global__ void bitreverse1(void *data) {
	unsigned int *idata = (unsigned int*) data;
	idata[threadIdx.x] = bitreverse1(idata[threadIdx.x]);
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int testx(void) {
	void *d = NULL;
	int i;
	unsigned int idata[WORK_SIZE], odata[WORK_SIZE];

	for (i = 0; i < WORK_SIZE; i++)
		idata[i] = (unsigned int) i;

	CUDA_CHECK_RETURN(cudaMalloc((void**) &d, sizeof(int) * WORK_SIZE));
	CUDA_CHECK_RETURN(
			cudaMemcpy(d, idata, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice));

	bitreverse1<<<1, WORK_SIZE, WORK_SIZE * sizeof(int)>>>(d);

	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaMemcpy(odata, d, sizeof(int) * WORK_SIZE, cudaMemcpyDeviceToHost));

	for (i = 0; i < WORK_SIZE; i++)
		printf("Input value: %u, device output: %u, host output: %u\n",
				idata[i], odata[i], bitreverse1(idata[i]));

	CUDA_CHECK_RETURN(cudaFree((void*) d));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	return 0;
}
