// ===================================
// Main.cpp file generated by OptFrame
// Project EFP
// ===================================

#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <numeric>
#include "../../../OptFrame/RandGen.hpp"
#include "../../../OptFrame/Util/RandGenMersenneTwister.hpp"

#include "test.cuh"

using namespace std;
using namespace optframe;
using namespace EFP;

int rew2016CUDADemandForecasting(int argc, char **argv)
{
	cout << "Welcome to REW2016 -- REED " << endl;

	RandGenMersenneTwister rg;
	//long  1412730737
	long seed = time(NULL); //CalibrationMode
	//seed = 9;
	cout << "Seed = " << seed << endl;
	srand(seed);
	rg.setSeed(seed);

	if (argc != 6)
	{
		cout << "Parametros incorretos!" << endl;
		cout << "Os parametros esperados sao: instance output granularidade timeES argvforecastingHorizonteMinutes" << endl;
		exit(1);
	}

	const char* instance = argv[1];
	const char* outputFile = argv[2];
	int granularityMin = atoi(argv[3]);
	int argvforecastingHorizonteMinutes = atoi(argv[4]);
	int argvTimeES = atoi(argv[5]);

	int forecastingHorizonteMinutes = argvforecastingHorizonteMinutes;
	string nomeOutput = outputFile;
	string nomeInstace = instance;

	if(forecastingHorizonteMinutes < granularityMin)
	{
		cout<<"EXIT WITH ERROR! Forecasting horizon lower than granularity"<<endl;
		cout<<"granularityMin: "<<granularityMin<<"\t forecastingHorizonteMinutes:"<<forecastingHorizonteMinutes<<endl;
		return 0;
	}

	//===================================
	cout << "Parametros:" << endl;
	cout << "nomeOutput=" << nomeOutput << endl;


	treatREEDDataset alexandreTreatObj;


	vector<pair<double, double> > datasetDemandCut = alexandreTreatObj.cutData("./REED/channel_1.dat", "REED/saida.dat");
	cout << "data has ben cut with success" << endl;
	cout << datasetDemandCut.size() << endl;
//	getchar();

	vector<pair<double, double> > datasetInterpolated = alexandreTreatObj.interpolate(datasetDemandCut, 1);
	cout << "data has been interpolated  with success" << endl;
	cout << datasetInterpolated.size() << endl;
//	getchar();

	vector<pair<double, double> > datasetWithSpecificGranularity = alexandreTreatObj.separateThroughIntervals(datasetInterpolated, 60 * granularityMin, "./REED/instance", false);
	cout << "data has been split" << endl;
	cout << "dataSize =" << datasetWithSpecificGranularity.size() << endl;
	//cout<<dataset60Seconds<<endl;
	cout << "data saved with sucess" << endl;

	alexandreTreatObj.createInstance(datasetWithSpecificGranularity, "./REED/instance");
//	getchar();
//	getchar();

//dotest();
//getchar();
//getchar();

	//double argvAlphaACF = atof(argv[4]);

//
//	const char* caminho = argv[1];
//	const char* caminhoValidation = argv[2];
//	const char* caminhoOutput = argv[3];
//	const char* caminhoParameters = argv[4];
//	int instN = atoi(argv[5]);
//	int stepsAhead = atoi(argv[6]);
//	int mu = atoi(argv[7]);

	//cout << "argvAlphaACF=" << argvAlphaACF << endl;
//	getchar();
//	cout << "instN=" << instN << endl;
//	cout << "stepsAhead=" << stepsAhead << endl;
//	cout << "mu=" << mu << endl;
	//===================================

	//CONFIG FILES FOR CONSTRUTIVE 0 AND 1

	vector<string> explanatoryVariables;
	string instanceREED = "./REED/instance";

	vector<string> vInstances;
	vInstances.push_back(instanceREED);

	explanatoryVariables.push_back(vInstances[0]);

	treatForecasts rF(explanatoryVariables);

	//vector<vector<double> > batchOfBlindResults; //vector with the blind results of each batch

	/*int beginValidationSet = 0;
	 int nTrainningRoundsValidation = 50;
	 int nValidationSamples = problemParam.getNotUsedForTest() + nTrainningRoundsValidation * stepsAhead;
	 cout << "nValidationSamples = " << nValidationSamples << endl;
	 int nTotalForecastingsValidationSet = nValidationSamples;

	 vector<vector<double> > validationSet; //validation set for calibration
	 validationSet.push_back(rF.getPartsForecastsEndToBegin(0, beginValidationSet, nTotalForecastingsValidationSet));
	 validationSet.push_back(rF.getPartsForecastsEndToBegin(1, beginValidationSet, nTotalForecastingsValidationSet + stepsAhead));
	 validationSet.push_back(rF.getPartsForecastsEndToBegin(2, beginValidationSet, nTotalForecastingsValidationSet + stepsAhead));
	 */

//	int maxMu = 100;
//	int maxInitialDesv = 10;
//	int maxMutationDesv = 30;
//	int maxPrecision = 300;
	int nBatches = 1;

	vector<vector<double> > vfoIndicatorCalibration; //vector with the FO of each batch

	vector<SolutionEFP> vSolutionsBatches; //vector with the solution of each batch

	for (int n = 0; n < nBatches; n++)
	{
//		int contructiveNumberOfRules = rg.rand(maxPrecision) + 10;
//		int evalFOMinimizer = rg.rand(NMETRICS); //tree is the number of possible objetive function index minimizers
//		int evalAprox = rg.rand(2); //Enayatifar aproximation using previous values
//		int construtive = rg.rand(3);
//		double initialDesv = rg.rand(maxInitialDesv) + 1;
//		double mutationDesv = rg.rand(maxMutationDesv) + 1;
//		int mu = rg.rand(maxMu) + 1;
//		int lambda = mu * 6;

		//limit ACF for construtive ACF
//		double alphaACF = rg.rand01();
//		int alphaSign = rg.rand(2);
//		if (alphaSign == 0)
//			alphaACF = alphaACF * -1;

		// ============ FORCES ======================
//		initialDesv = 10;
//		mutationDesv = 20;
		int mu = 100;
		int lambda = mu * 6;
		int evalFOMinimizer = MAPE_INDEX;
		int contructiveNumberOfRules = 100;
		int evalAprox = 0;
		double alphaACF = 0.5;
		int construtive = 2;
		// ============ END FORCES ======================

		// ============= METHOD PARAMETERS=================
		methodParameters methodParam;
		//seting up Continous ES params
		methodParam.setESInitialDesv(10);
		methodParam.setESMutationDesv(20);
		methodParam.setESMaxG(100000);

		//seting up ES params
		methodParam.setESMU(mu);
		methodParam.setESLambda(lambda);

		//seting up ACF construtive params
		methodParam.setConstrutiveMethod(construtive);
		methodParam.setConstrutivePrecision(contructiveNumberOfRules);
		vector<double> vAlphaACFlimits;
		vAlphaACFlimits.push_back(alphaACF);
		methodParam.setConstrutiveLimitAlphaACF(vAlphaACFlimits);

		//seting up Eval params
		methodParam.setEvalAprox(evalAprox);
		methodParam.setEvalFOMinimizer(evalFOMinimizer);
		// ==========================================

		// ================== READ FILE ============== CONSTRUTIVE 0 AND 1
		ProblemParameters problemParam;
		//ProblemParameters problemParam(vParametersFiles[randomParametersFiles]);


		int nSA = forecastingHorizonteMinutes/granularityMin;
		problemParam.setStepsAhead(nSA);
		int stepsAhead = problemParam.getStepsAhead();

		int nTrainningDays = 7;
		double pointsPerHour = 60.0/ granularityMin;

		//========SET PROBLEM MAXIMUM LAG ===============
		problemParam.setMaxLag(pointsPerHour*24*3); // with maxLag equals to 2 you only lag K-1 as option
		int maxLag = problemParam.getMaxLag();

		//If maxUpperLag is greater than 0 model uses predicted data
		problemParam.setMaxUpperLag(0);
		int maxUpperLag = problemParam.getMaxUpperLag();
		//=================================================

		int numberOfTrainingPoints = 24 * pointsPerHour * nTrainningDays; //24hour * 7days * 4 points per hour
//		int nTotalForecastingsTrainningSet = maxLag + nTrainningRounds * stepsAhead;
		int nTotalForecastingsTrainningSet = maxLag + numberOfTrainingPoints;

		int beginTrainingSet = 1;

		int totalNumberOfSamplesTarget = rF.getForecastsSize(0);
		cout << "BeginTrainninningSet: " << beginTrainingSet << endl;
		cout << "\t #nTotalForecastingsTrainningSet: " << nTotalForecastingsTrainningSet << endl;
		cout << "#sizeTrainingSet: " << totalNumberOfSamplesTarget << endl;
		cout << "maxNotUsed: " << problemParam.getMaxLag() << endl;
		cout << "#StepsAhead: " << stepsAhead << endl;
		cout << "#forecastingHorizonteMinutes: " << forecastingHorizonteMinutes << endl;
		cout << "#granularityMin: " << granularityMin << endl << endl;

		int timeES = argvTimeES;
		vector<double> foIndicatorCalibration;
		vector<double> vectorOfForecasts;
		double averageError = 0;
		int countSlidingWindows = 0;

		for (int begin = 0; (nTotalForecastingsTrainningSet + begin + stepsAhead) <= totalNumberOfSamplesTarget; begin += stepsAhead)
		{
			vector<vector<double> > trainningSet; // trainningSetVector
			trainningSet.push_back(rF.getPartsForecastsBeginToEnd(0, begin, nTotalForecastingsTrainningSet));

			ForecastClass forecastObject(trainningSet, problemParam, rg, methodParam);

			pair<Solution<RepEFP>&, Evaluation&>* sol;
			sol = forecastObject.run(timeES, 0, 0);

			vector<vector<double> > validationSet; //validation set for calibration
			cout << "blind test begin: " << nTotalForecastingsTrainningSet + begin
					<< " end:" << nTotalForecastingsTrainningSet + begin + stepsAhead << endl;

			validationSet.push_back(rF.getPartsForecastsBeginToEnd(0, nTotalForecastingsTrainningSet + begin - maxLag, maxLag + stepsAhead));
			vector<double> foIndicatorsWeeks;
			foIndicatorsWeeks = forecastObject.returnErrors(sol, validationSet);
			foIndicatorCalibration.push_back(foIndicatorsWeeks[MAPE_INDEX]);
			averageError += foIndicatorsWeeks[MAPE_INDEX];

			vector<double> currentForecasts = forecastObject.returnForecasts(sol, validationSet);
			for (int cF = 0; cF < currentForecasts.size(); cF++)
				vectorOfForecasts.push_back(currentForecasts[cF]);

			cout << foIndicatorCalibration << "\t average:" << averageError / (countSlidingWindows + 1) << endl;
//			cout<<validationSet<<endl;
//			cout<<trainningSet<<endl;
//			getchar();

			countSlidingWindows++;
		}
		cout << foIndicatorCalibration << endl;

		double finalAverage = 0;
		for (int e = 0; e < foIndicatorCalibration.size(); e++)
			finalAverage += foIndicatorCalibration[e];
		finalAverage /= foIndicatorCalibration.size();

		vector<double> parametersResults;
		parametersResults.push_back(finalAverage);
		parametersResults.push_back(stepsAhead);
		parametersResults.push_back(forecastingHorizonteMinutes);
		parametersResults.push_back(granularityMin);
		parametersResults.push_back(numberOfTrainingPoints);
		parametersResults.push_back(maxLag);
		parametersResults.push_back(mu);
		parametersResults.push_back(lambda);
		parametersResults.push_back(timeES);
		parametersResults.push_back(seed);

		string calibrationFile = "./resultsREW2016_REED";

		FILE* fResults = fopen(calibrationFile.c_str(), "a");

		for (int i = 0; i < parametersResults.size(); i++)
			fprintf(fResults, "%.7f\t", parametersResults[i]);

		for (int i = 0; i < vectorOfForecasts.size(); i++)
			fprintf(fResults, "%.3f\t", vectorOfForecasts[i]);

		fprintf(fResults, "\n");

		fclose(fResults);

		//else
		//beginTrainingSet = nTotalForecastingsValidationSet + n * (nTotalForecastingsTrainningSet / 10);

		/*
		 while (beginTrainingSet + nTotalForecastingsTrainningSet > rF.getForecastsDataSize())
		 {
		 beginTrainingSet = 744;
		 nTrainningRounds = rg.rand(maxTrainningRounds) + 1;
		 nTotalForecastingsTrainningSet = maxNotUsedForTest + nTrainningRounds * stepsAhead;
		 }*/

//		cout << trainningSet << endl;
//		getchar();
//		for (int i = 0; i < sol->first.getR().singleIndex.size(); i++)
//		{
//			cout << sol->first.getR().singleIndex[i].second << endl;
//		}
//		int nValidationSamples = maxLag + nVR * stepsAhead;
//		vector<vector<double> > validationSet; //validation set for calibration
//		validationSet.push_back(rF.getPartsForecastsEndToBegin(0, beginTrainingSet + stepsAhead, nValidationSamples));
//		vector<double> foIndicatorsWeeks;
//		foIndicatorsWeeks = forecastObject.returnErrors(sol, validationSet);
//		foIndicatorCalibration.push_back(foIndicatorsWeeks[MAPE_INDEX]);
//		foIndicatorCalibration.push_back(foIndicatorsWeeks[RMSE_INDEX]);
//		cout << "maxLag:" << maxLag << endl;
//		cout << "nValidationSamples:" << nValidationSamples << endl;
//		//int beginValidationSet = 0;
//		for (int w = 3; w >= 0; w--)
//		{
//			vector<vector<double> > validationSet; //validation set for calibration
//			validationSet.push_back(rF.getPartsForecastsEndToBegin(0, w * 168, nValidationSamples));
//			vector<double> foIndicatorsWeeks;
//			foIndicatorsWeeks = forecastObject.returnErrors(sol, validationSet);
//			foIndicatorCalibration.push_back(foIndicatorsWeeks[MAPE_INDEX]);
//			//foIndicatorCalibration.push_back(foIndicatorsWeeks[RMSE_INDEX]);
//		}
//		int finalNRules = sol->first.getR().singleIndex.size();
//		cout<<"nRules:"<<finalNRules<<"/"<<contructiveNumberOfRules<<endl;
//		getchar();
//		double intervalOfBeginTrainningSet = double(beginTrainingSet) / double(rF.getForecastsDataSize());
////		foIndicatorCalibration.push_back(sol->second.evaluation());
//		foIndicatorCalibration.push_back(construtive);
//		foIndicatorCalibration.push_back(alphaACF);
//		foIndicatorCalibration.push_back(contructiveNumberOfRules);
////		foIndicatorCalibration.push_back(finalNRules);
//		foIndicatorCalibration.push_back(evalAprox);
//		foIndicatorCalibration.push_back(timeES);
//		foIndicatorCalibration.push_back(evalFOMinimizer);
//		foIndicatorCalibration.push_back(nTrainningRounds);
//		foIndicatorCalibration.push_back(beginTrainingSet);
//		foIndicatorCalibration.push_back(intervalOfBeginTrainningSet);
//		foIndicatorCalibration.push_back(nTotalForecastingsTrainningSet);
////		foIndicatorCalibration.push_back(initialDesv);
////		foIndicatorCalibration.push_back(mutationDesv);
//
//		foIndicatorCalibration.push_back(argvTargetTimeSeries);
//
//		foIndicatorCalibration.push_back(seed);
//		//getchar();
//		//cout << foIndicatorCalibration << endl;
////		vSolutionsBatches.push_back(sol->first);
	}

	return 0;
}

//
//for (int w = 4; w >= 1; w--)
//	{
//		vector<double> foIndicatorsMAPE;
//		vector<double> foIndicatorsRMSE;
//
//		for (int day = 1; day <= 7; day++)
//		{
//			vector<vector<double> > validationSet; //validation set for calibration
//			validationSet.push_back(rF.getPartsForecastsEndToBegin(0, w * 168 - stepsAhead * day, nValidationSamples));
//			vector<double> foIndicators;
//			foIndicators = forecastObject.returnErrors(sol, validationSet);
//			foIndicatorsMAPE.push_back(foIndicators[MAPE_INDEX]);
//			foIndicatorsRMSE.push_back(foIndicators[RMSE_INDEX]);
//		}
//		double sumMAPE = accumulate(foIndicatorsMAPE.begin(), foIndicatorsMAPE.end(), 0.0);
//		double sumRMSE = accumulate(foIndicatorsRMSE.begin(), foIndicatorsRMSE.end(), 0.0);
//
//		foIndicatorCalibration.push_back(sumMAPE/foIndicatorsMAPE.size());
//		foIndicatorCalibration.push_back(sumRMSE/foIndicatorsRMSE.size());
//	}
