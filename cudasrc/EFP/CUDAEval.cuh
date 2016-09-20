/*
 * CUDAEval.cuh
 *
 *  Created on: 26/12/2015
 *      Author: root
 */

#ifndef CUDAEVAL_CUH_
#define CUDAEVAL_CUH_

#include <vector>
#include <iostream>
#include <assert.h>

#include "CUDAEval.h"

using namespace std;

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

struct val2
{
	int v1;
	int v2;
};

struct val6
{
	float GREATER; // = 0; // A
	float GREATER_WEIGHT; // = 1; // V
	float LOWER; // = 2; // B
	float LOWER_WEIGHT; // = 3; // W
	float EPSILON; // = 4; // E
	float PERTINENCEFUNC; // = 5; // Pertinence function index
};

struct CUDARep
{
	//single index [ (file,t - k) ,...] ex:  [ (file = 0,t - 10),(file = 0,t - 2), ...]

	int size;

	//vector<pair<int, int> > singleIndex;
	val2* h_singleIndex;
	val2* d_singleIndex;

	//vector<vector<double> > singleFuzzyRS; //single inputs relationships
	val6* h_singleFuzzyRS;
	val6* d_singleFuzzyRS;

	int datasize; // forecastings

	//vector<vector<pair<int, int> > > averageIndex;
	//vector<vector<double> > averageFuzzyRS;
	//val6* averageFuzzyRS;

	//vector<vector<pair<int, int> > > derivativeIndex;
	//vector<vector<double> > derivativeFuzzyRS;
	//val6* derivativeFuzzyRS;
};

__host__ void transferRep(const RepEFP& rep, CUDARep& cudarep)
{
	cudarep.size = rep.singleIndex.size();
	cudarep.h_singleIndex = new val2[cudarep.size];
	cudarep.h_singleFuzzyRS = new val6[cudarep.size];

	for (unsigned i = 0; i < cudarep.size; i++)
	{
		cudarep.h_singleIndex[i].v1 = rep.singleIndex[i].first;
		cudarep.h_singleIndex[i].v2 = rep.singleIndex[i].second;

		cudarep.h_singleFuzzyRS[i].GREATER = rep.singleFuzzyRS[i][GREATER];
		cudarep.h_singleFuzzyRS[i].GREATER_WEIGHT = rep.singleFuzzyRS[i][GREATER_WEIGHT];
		cudarep.h_singleFuzzyRS[i].LOWER = rep.singleFuzzyRS[i][LOWER];
		cudarep.h_singleFuzzyRS[i].LOWER_WEIGHT = rep.singleFuzzyRS[i][LOWER_WEIGHT];
		cudarep.h_singleFuzzyRS[i].EPSILON = rep.singleFuzzyRS[i][EPSILON];
		cudarep.h_singleFuzzyRS[i].PERTINENCEFUNC = rep.singleFuzzyRS[i][PERTINENCEFUNC];
	}

	CUDA_CHECK_RETURN(cudaMalloc((void** ) &cudarep.d_singleIndex, sizeof(val2) * cudarep.size));
	CUDA_CHECK_RETURN(cudaMalloc((void** ) &cudarep.d_singleFuzzyRS, sizeof(val6) * cudarep.size));
	CUDA_CHECK_RETURN(cudaMemcpy(cudarep.d_singleIndex, cudarep.h_singleIndex, sizeof(val2) * cudarep.size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(cudarep.d_singleFuzzyRS, cudarep.h_singleFuzzyRS, sizeof(val6) * cudarep.size, cudaMemcpyHostToDevice));

}

__device__ void defuzzification(double ruleGreater, double greaterWeight, double ruleLower, double lowerWeight, double ruleEpsilon, FuzzyFunction fuzzyFunc, double value, double* estimation, double* greaterAccepted, double* lowerAccepted)
{

	if (fuzzyFunc == Heavisde)
	{
		if (value > ruleGreater)
		{
			*estimation += greaterWeight;
			*greaterAccepted += 1;
		}

		if (value < ruleLower)
		{
			*estimation += lowerWeight;
			*lowerAccepted += 1;
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
		*estimation += est;

		*greaterAccepted += mu;
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
		*estimation += est;

		*lowerAccepted += mu;
	}
}

//__host__ vector<double> returnForecasts(CUDARep cudarep, float* dForecastings, int begin, int maxLag, int stepsAhead, const int aprox);
__global__ void kernelForecasts(int nThreads, CUDARep cudarep, float* dForecastings, int* dfSize, int maxLag, int stepsAhead, const int aprox, float* predicted);

__host__ void initializeCudaItems(int datasize, int vForecastingSize, int* hfSize, const float* hForecastings, float** dForecastings, int** dfSize)
{

	CUDA_CHECK_RETURN(cudaMalloc((void** ) dForecastings, sizeof(float) * datasize));
	CUDA_CHECK_RETURN(cudaMemcpy(*dForecastings, hForecastings, sizeof(float) * datasize, cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMalloc((void** ) dfSize, sizeof(int) * vForecastingSize));
	CUDA_CHECK_RETURN(cudaMemcpy(*dfSize, hfSize, sizeof(int) * vForecastingSize, cudaMemcpyHostToDevice));

}

__host__ void freeInitializeCudaItems(float** dForecastings, int** dfSize)
{
	cudaFree(dForecastings);
	cudaFree(dfSize);
}

__host__ vector<double> returnTrainingSetForecasts(CUDARep cudarep, float* dForecastings, int* dfSize, int* hfSize, int maxLag, int stepsAhead, const int aprox)
{

	int nForTargetFile = hfSize[0];
	int nSamples = nForTargetFile - maxLag;

	int nThreads = ceil(nSamples / float(stepsAhead));

	int threadsPerBlock = 256; // tx
	int blocks;
	CUDA_CHECK_RETURN(cudaOccupancyMaxPotentialBlockSize(&blocks, &threadsPerBlock, kernelForecasts, 0, 0));
//	int threadsPerBlock = 768; // tx
//	int blocks=126;

	blocks = ceil(nThreads / float(threadsPerBlock));

//	cout << "nForTargetFile=" << nForTargetFile << endl;
//	cout << "maxLag=" << maxLag << endl;
//	cout << "nSamples=" << nSamples << endl;
//	cout << "stepsAhead=" << stepsAhead << endl;
//	cout << "nThreads=" << nThreads << endl;
//	cout << "threadsPerBlock=" << threadsPerBlock << endl;
//	cout << "blocks=" << blocks << endl;
//	getchar();

//	if (nThreads > threadsPerBlock)
//	{
//		printf("ERROR! MORE THAN %d threads per block! TOTAL: %d\n", threadsPerBlock, nThreads);
//		exit(1);
//	}

	int rsize = nThreads * stepsAhead;
	float* hrForecasts = new float[rsize];
	float* predicted;
	CUDA_CHECK_RETURN(cudaMalloc((void** ) &predicted, sizeof(float) * rsize));

//	cudaEvent_t start, stop;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//	cudaEventRecord(start, 0); //where 0 is the default stream
	kernelForecasts<<<blocks, threadsPerBlock>>>(nThreads, cudarep, dForecastings, dfSize, maxLag, stepsAhead, aprox, predicted);
	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
//	cudaEventRecord(stop, 0);
//	cudaEventSynchronize(stop);

	CUDA_CHECK_RETURN(cudaMemcpy(hrForecasts, predicted, sizeof(float) * rsize, cudaMemcpyDeviceToHost));
	cudaFree(predicted);
//	float dt_ms;
//	cudaEventElapsedTime(&dt_ms, start, stop);
//	cout << "ElapsedTime : " << dt_ms << endl;
//	getchar();

	/*
	 for (int i = maxLag; i < nForTargetFile; i += stepsAhead) // main loop that varries all the time series
	 {
	 vector<double> predicteds = returnForecasts(cudarep, dForecastings, i, maxLag, stepsAhead, aprox);

	 for (int f = 0; f < predicteds.size(); f++)
	 allForecasts.push_back(predicteds[f]);
	 }
	 */

	vector<double> allForecasts;

	for (unsigned k = 0; k < rsize; k++)
		allForecasts.push_back(hrForecasts[k]);

	//TODO do it in a better style
	if (allForecasts.size() > nSamples)
	{
		int nExtraForecasts = allForecasts.size() - nSamples;
		allForecasts.erase(allForecasts.begin() + allForecasts.size() - nExtraForecasts, allForecasts.end());
	}

	free(hrForecasts);

	return allForecasts;
}

vector<double> gpuTrainingSetForecasts(const RepEFP& rep, int maxLag, int stepsAhead, const int aprox, float* dForecastings, int* dfSize, int* hfSize, int datasize, const float* hForecastings)
{

	CUDARep cudarep;

	transferRep(rep, cudarep);

//	CUDA_CHECK_RETURN(cudaMalloc((void** ) dForecastings, sizeof(float) * datasize));

//	initializeCudaItems(datasize, 1, hfSize, hForecastings, &dForecastings, &dfSize);

//	vector<double> vr ;
	vector<double> vr = returnTrainingSetForecasts(cudarep, dForecastings, dfSize, hfSize, maxLag, stepsAhead, aprox);

//	CUDA_CHECK_RETURN(cudaDeviceReset());
//	cudaFree(dForecastings);
//	cudaFree(dfSize);
	cudaFree(cudarep.d_singleIndex);
	cudaFree(cudarep.d_singleFuzzyRS);
	free(cudarep.h_singleIndex);
	free(cudarep.h_singleFuzzyRS);

	return vr;
}

#endif /* CUDAEVAL_CUH_ */
