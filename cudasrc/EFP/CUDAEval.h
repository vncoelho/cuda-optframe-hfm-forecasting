#ifndef CUDAEVAL_H_
#define CUDAEVAL_H_

#include <vector>
#include <iostream>

#include "Representation.h"

vector<double> gpuTrainingSetForecasts(const RepEFP& rep, int maxLag, int stepsAhead, const int aprox, float* dForecastings, int* dfSize,int* hfSize,int datasize,const float* hForecastings);

void initializeCudaItems(int datasize, int vForecastingSize, int* hfSize, const float* hForecastings, float** dForecastings, int** dfSize);

#endif /* CUDAEVAL_H_ */
