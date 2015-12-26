
#ifndef CUDAEVAL_H_
#define CUDAEVAL_H_

#include <vector>
#include <iostream>

#include "Representation.h"

vector<double> gpuTrainingSetForecasts(const RepEFP& rep, const vector<vector<double> >& vForecastings, int maxLag, int stepsAhead, const int aprox);

#endif /* CUDAEVAL_H_ */
