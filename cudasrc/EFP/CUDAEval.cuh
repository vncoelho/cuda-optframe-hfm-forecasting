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
	double GREATER; // = 0; // A
	double GREATER_WEIGHT; // = 1; // V
	double LOWER; // = 2; // B
	double LOWER_WEIGHT; // = 3; // W
	double EPSILON; // = 4; // E
	double PERTINENCEFUNC; // = 5; // Pertinence function index
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

	for (unsigned i = 0; i < rep.singleIndex.size(); i++)
	{
		cudarep.h_singleIndex[i].v1 = rep.singleIndex[i].first;
		cudarep.h_singleIndex[i].v2 = rep.singleIndex[i].second;
	}

	for (unsigned i = 0; i < rep.singleIndex.size(); i++)
	{
		cudarep.h_singleFuzzyRS[i].GREATER = rep.singleFuzzyRS[i][GREATER];
		cudarep.h_singleFuzzyRS[i].GREATER_WEIGHT = rep.singleFuzzyRS[i][GREATER_WEIGHT];
		cudarep.h_singleFuzzyRS[i].LOWER = rep.singleFuzzyRS[i][LOWER];
		cudarep.h_singleFuzzyRS[i].LOWER_WEIGHT = rep.singleFuzzyRS[i][LOWER_WEIGHT];
		cudarep.h_singleFuzzyRS[i].EPSILON = rep.singleFuzzyRS[i][EPSILON];
		cudarep.h_singleFuzzyRS[i].PERTINENCEFUNC = rep.singleFuzzyRS[i][PERTINENCEFUNC];
	}

	CUDA_CHECK_RETURN(cudaMalloc((void**) &cudarep.d_singleIndex, sizeof(val2) * cudarep.size));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &cudarep.d_singleFuzzyRS, sizeof(val6) * cudarep.size));
	CUDA_CHECK_RETURN(cudaMemcpy(cudarep.d_singleIndex, cudarep.h_singleIndex, sizeof(val2) * cudarep.size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(cudarep.d_singleFuzzyRS, cudarep.h_singleFuzzyRS, sizeof(val2) * cudarep.size, cudaMemcpyHostToDevice));
}

__device__ int getKValue(const int K, const int file, const int i, const int pa, const vector<vector<double> >& vForecastings, const vector<double>& predicteds);

__device__ void defuzzification(double ruleGreater, double greaterWeight, double ruleLower, double lowerWeight, double ruleEpsilon, FuzzyFunction fuzzyFunc, double value, double& estimation, double& greaterAccepeted, double& lowerAccepted);

__host__ vector<double> returnForecasts(const RepEFP& rep, const vector<vector<double> >& vForecastings, int begin, int maxLag, int stepsAhead, const int aprox);

__host__ vector<double> returnTrainingSetForecasts(const RepEFP& rep, const vector<vector<double> >& vForecastings, int maxLag, int stepsAhead, const int aprox);

vector<double> gpuTrainingSetForecasts(const RepEFP& rep, const vector<vector<double> >& vForecastings, int maxLag, int stepsAhead, const int aprox)
{
	CUDARep cudarep;
	transferRep(rep, cudarep);

	return returnTrainingSetForecasts(rep, vForecastings, maxLag, stepsAhead, aprox);
}

#endif /* CUDAEVAL_CUH_ */
