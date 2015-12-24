// ===================================
// Main.cpp file generated by OptFrame
// Project MODM
// ===================================

#include <stdlib.h>
#include <math.h>
#include <iostream>

#include "../OptFrame/Loader.hpp"
#include "MODM/Evaluator.cpp"
#include "../OptFrame/Heuristics/VNS/MOVNSLevels.hpp"
#include "../OptFrame/Heuristics/2PPLS.hpp"
#include "../OptFrame/MultiEvaluator.hpp"
#include "../OptFrame/MultiObjSearch.hpp"
#include "../OptFrame/Util/UnionNDSets.hpp"

#include <string>
#include "MODM.h"

using namespace std;
using namespace optframe;
using namespace MODM;

char* execCommand(const char* command)
{

	FILE* fp;
	char* line = NULL;
	// Following initialization is equivalent to char* result = ""; and just
	// initializes result to an empty string, only it works with
	// -Werror=write-strings and is so much less clear.
	char* result = (char*) calloc(1, 1);
	size_t len = 0;

	fflush(NULL);
	fp = popen(command, "r");
	if (fp == NULL)
	{
		printf("Cannot execute command:\n%s\n", command);
		return NULL;
	}

	while (getline(&line, &len, fp) != -1)
	{
		// +1 below to allow room for null terminator.
		result = (char*) realloc(result, strlen(result) + strlen(line) + 1);
		// +1 below so we copy the final null terminator.
		strncpy(result + strlen(result), line, strlen(line) + 1);
		free(line);
		line = NULL;
	}

	fflush(fp);
	if (pclose(fp) != 0)
	{
		perror("Cannot close stream.\n");
	}

	return result;
}

double hipervolume(vector<vector<double> > v)
{
	int nSol = v.size();
	int nObj = v[0].size();
	string tempFile = "tempFileHipervolueFunc";
	FILE* fTempHV = fopen(tempFile.c_str(), "w");

	for (int s = 0; s < nSol; s++)
	{
		for (int o = 0; o < nObj; o++)
		{
			fprintf(fTempHV, "%.7f\t", v[s][o]);
		}
		fprintf(fTempHV, "\n");
	}

	fclose(fTempHV);
	stringstream ss;
	ss << "./hv\t -r \"" << 0 << " " << 0 << "\" \t" << tempFile.c_str();
	string hvValueString = execCommand(ss.str().c_str());
	double hvValue = atof(hvValueString.c_str());
	return hvValue;
}

int main(int argc, char **argv)
{

	int nOfArguments = 7;
	if (argc != (1 + nOfArguments))
	{
		cout << "Parametros incorretos!" << endl;
		cout << "Os parametros esperados sao: \n"
				"1 - instancia \n"
				"2 - saida - for saving solutions for each execution - type write\n"
				"3 - saida geral -- general file for savings all results - type append \n"
				"4 - timeILS\n"
				"5 - alpha Builder Int\n"
				"6 - alpha NS Int \n"
				"7 - initial population size \n" << endl;
		exit(1);
	}

	RandGenMersenneTwister rg;
	long seed = time(NULL);

	//seed = 10;
	srand(seed);
	rg.setSeed(seed);

	const char* instancia = argv[1];
	const char* saida = argv[2];
	const char* saidaGeral = argv[3];
	int timeILS = atoi(argv[4]);
	int alphaBuilderInt = atoi(argv[5]);
	int alphaNSInt = atoi(argv[6]);
	int pop = atoi(argv[7]);

	double alphaBuilder = alphaBuilderInt / 10.0;
	double alphaNeighARProduct = alphaNSInt / 10.0;

	string filename = instancia;
	string output = saida;
	string outputGeral = saidaGeral;
	cout << "filename = " << filename << endl;
	cout << "output = " << output << endl;
	cout << "outputGeral = " << outputGeral << endl;
	cout << "timeILS = " << timeILS << endl;
	cout << "alphaBuilder = " << alphaBuilder << endl;
	cout << "alphaNeighARProduct = " << alphaNeighARProduct << endl;
	cout << "initial population size = " << pop << endl;
	cout << "Seed = " << seed << endl;

	//filename = "./MyProjects/MODM/Instances/S3-15/S3-10-15-1-s.txt";
	//filename = "./MyProjects/MODM/Instances/L-5/L-10-5-1-l.txt";

	//string filename = "./MyProjects/MODM/Instances/L-5/L-15-5-2-s.txt";

	File* file;

	try
	{
		file = new File(filename);
	} catch (FileNotFound& f)
	{
		cout << "File '" << filename << "' not found" << endl;
		return false;
	}

	Scanner scanner(file);

	ProblemInstance p(scanner);

	// add everything to the HeuristicFactory 'hf'

	MODMADSManager adsMan(p);
	MODMEvaluator eval(p, adsMan);
	MODMRobustnessEvaluator evalRobustness(p, adsMan, rg);

	ConstructiveBasicGreedyRandomized grC(p, rg, adsMan);

	NSSeqSWAP nsseq_swap(rg, &p);
	NSSeqSWAPInter nsseq_swapInter(rg, &p);
	NSSeqInvert nsseq_invert(rg, &p);
	NSSeqARProduct nsseq_arProduct(rg, &p, alphaNeighARProduct);
	NSSeqADD nsseq_add(rg, &p);

	// ================ BEGIN OF CHECK MODULE ================

	/*	CheckCommand<RepMODM, AdsMODM> check(false);
	 check.add(grC);
	 check.add(eval);
	 //check.add(nsseq_swap);
	 //check.add(nsseq_swapInter);
	 check.add(nsseq_invert);
	 //check.add(nsseq_arProduct);

	 check.run(1, 1);
	 getchar();*/

	// ================ END OF CHECK MODULE ================
	FirstImprovement<RepMODM, AdsMODM> fiSwap(eval, nsseq_swap);
	FirstImprovement<RepMODM, AdsMODM> fiSwapInter(eval, nsseq_swapInter);
	FirstImprovement<RepMODM, AdsMODM> fiInvert(eval, nsseq_invert);

	int nMovesRDM = 500000;
	RandomDescentMethod<RepMODM, AdsMODM> rdmSwap(eval, nsseq_swap, nMovesRDM);
	RandomDescentMethod<RepMODM, AdsMODM> rdmSwapInter(eval, nsseq_swapInter, nMovesRDM);
	RandomDescentMethod<RepMODM, AdsMODM> rdmInvert(eval, nsseq_invert, nMovesRDM);
	RandomDescentMethod<RepMODM, AdsMODM> rdmARProduct(eval, nsseq_arProduct, nMovesRDM);
	RandomDescentMethod<RepMODM, AdsMODM> rdmADD(eval, nsseq_add, 1);

	vector<LocalSearch<RepMODM, AdsMODM>*> vLS;
	//vLS.push_back(&fiSwap);
	// vLS.push_back(&fiSwapInter);
	//vLS.push_back(&fiInvert);

	vLS.push_back(&rdmSwap);
	vLS.push_back(&rdmSwapInter);
	//vLS.push_back(&rdmInvert);
	vLS.push_back(&rdmADD);

	//vLS.push_back(&rdmARProduct);

	VariableNeighborhoodDescent<RepMODM, AdsMODM> vnd(eval, vLS);

	//ILSLPerturbationLPlus2<RepMODM, AdsMODM> ilsl_pert(eval, 100000, nsseq_invert, rg);
	ILSLPerturbationLPlus2<RepMODM, AdsMODM> ilsl_pert(eval, 100000, nsseq_arProduct, rg);
	//ILSLPerturbationLPlus2<RepMODM, AdsMODM> ilsl_pert(eval, 100000, nsseq_add, rg);
	ilsl_pert.add_ns(nsseq_add);
	ilsl_pert.add_ns(nsseq_swapInter);
	ilsl_pert.add_ns(nsseq_swap);
	ilsl_pert.add_ns(nsseq_invert);

	IteratedLocalSearchLevels<RepMODM, AdsMODM> ils(eval, grC, vnd, ilsl_pert, 50, 15);
	ils.setMessageLevel(3);

	pair<Solution<RepMODM, AdsMODM>&, Evaluation&>* finalSol;

	EmptyLocalSearch<RepMODM, AdsMODM> emptyLS;
	BasicGRASP<RepMODM, AdsMODM> g(eval, grC, emptyLS, alphaBuilder, 100000);

	g.setMessageLevel(3);
	int timeGRASP = 100;
	double target = 9999999;
	//MODMProblemCommand problemCommand(rg);
	//finalSol = g.search(timeGRASP,target);

	//===========================================
	//MO
	vector<Evaluator<RepMODM, AdsMODM>*> v_e;
	v_e.push_back(&eval);
	v_e.push_back(&evalRobustness);

//	NSSeqSWAP nsseq_swap(rg, &p);
//	NSSeqSWAPInter nsseq_swapInter(rg, &p);
//	NSSeqInvert nsseq_invert(rg, &p);
//	NSSeqARProduct nsseq_arProduct(rg, &p, alphaNeighARProduct);
//	NSSeqADD nsseq_add(rg, &p);
//
	vector<NSSeq<RepMODM, AdsMODM>*> neighboors;
	neighboors.push_back(&nsseq_arProduct);
	neighboors.push_back(&nsseq_add);
	//neighboors.push_back(&nsseq_swapInter);
	//neighboors.push_back(&nsseq_swap);

	//alphaBuilder as the limit

	GRInitialPopulation<RepMODM, AdsMODM> bip(grC, rg, 0.2);
	int initial_population_size = pop;
	initial_population_size = 10;
	MultiEvaluator<RepMODM, AdsMODM> mev(v_e);

	MOVNSLevels<RepMODM, AdsMODM> multiobjectvns(v_e, bip, initial_population_size, neighboors, rg, 10, 10);
	TwoPhaseParetoLocalSearch<RepMODM, AdsMODM> paretoSearch(mev, bip, initial_population_size, neighboors);

	UnionNDSets<RepMODM, AdsMODM> US(v_e);
	vector<vector<double> > PF1 = US.unionSets("./paretoCorsTesteS3-1", 291);

	vector<vector<double> > PF2 = US.unionSets("./paretoCorsTesteS3-2", 262);
	vector<vector<double> > ref = US.unionSets(PF1, PF2);
	vector<vector<double> > refMin = ref;

	cout << PF1.size() << endl;
	cout << PF2.size() << endl;
	cout << ref.size() << endl;
//	getchar();
	cout << "Reference set" << endl;
	for (int p = 0; p < ref.size(); p++)
	{
		cout << ref[p][0] << "\t" << ref[p][1] << endl;
		refMin[p][0]*=-1;
		refMin[p][1]*=-1;
	}

//
//	getchar();
	Pareto<RepMODM, AdsMODM>* pf;
	int time2PPLS = 120;
	for (int exec = 0; exec < 1; exec++)
	{
		//pf = multiobjectvns.search(300, 0);

		pf = paretoSearch.search(time2PPLS, 0);
	}

	vector<vector<Evaluation*> > vEval = pf->getParetoFront();
	vector<Solution<RepMODM, AdsMODM>*> vSolPf = pf->getParetoSet();

	int nObtainedParetoSol = vEval.size();
	vector<vector<double> > paretoDoubleEval;
	vector<vector<double> > paretoDoubleEvalMin;

	cout << "MO optimization finished! Printing Pareto Front!" << endl;
	for (int i = 0; i < nObtainedParetoSol; i++)
	{

		Solution<RepMODM, AdsMODM>* sol = vSolPf[i];

		const RepMODM& rep = sol->getR();
		const AdsMODM& ads = sol->getADS();
		vector<double> solEvaluations;
		double foProfit = vEval[i][0]->getObjFunction();
		double foVolatility = vEval[i][1]->getObjFunction();
		solEvaluations.push_back(foProfit);
		solEvaluations.push_back(foVolatility);
		paretoDoubleEval.push_back(solEvaluations);
		solEvaluations[0] *= -1;
		solEvaluations[1] *= -1;
		paretoDoubleEvalMin.push_back(solEvaluations);

		vector<int> nPerCat = evalRobustness.checkNClientsPerCategory(rep, ads);
		cout << foProfit << "\t" << foVolatility << "\t";

		int nTotalClients = nPerCat[nPerCat.size() - 1];

		for (int cat = 0; cat < 6; cat++)
			cout << nPerCat[cat] << "\t";
		cout << endl;
	}

	int card = US.cardinalite(paretoDoubleEval, ref);
	double sCToRef = US.setCoverage(paretoDoubleEval, ref);
	double sCFromRef = US.setCoverage(ref, paretoDoubleEval);
	double hv = hipervolume(paretoDoubleEvalMin);

	vector<double> utopicSol;
	utopicSol.push_back(-5226);
	utopicSol.push_back(-10);
	double delta = US.deltaMetric(paretoDoubleEvalMin, utopicSol);

	//Delta Metric and Hipervolume need to verify min
	cout << "Cardinalite = " << card << endl;
	cout << "Set Coverage to ref = " << sCToRef << endl;
	cout << "Set Coverage from ref  = " << sCFromRef << endl;
	cout << "delta  = " << delta << endl;
	cout << "deltaRef  = " << US.deltaMetric(refMin, utopicSol) << endl;
	cout << "hv  = " << hv << endl;
	cout << "ref  = " << hipervolume(refMin) << endl;

	FILE* fGeral = fopen(outputGeral.c_str(), "a");

	size_t pos = filename.find("Instances/");
	string instName = filename.substr(pos);

	fprintf(fGeral, "%s \t %d \t %.7f \t %.7f \t %d \t %d \t %.7f \t %.7f \t %.7f \t %.7f \t %ld \n", instName.c_str(), pop, alphaBuilder, alphaNeighARProduct, nObtainedParetoSol, card, sCToRef, sCFromRef, hv, delta, seed);

	fclose(fGeral);

	//getchar();
	//===========================================

	/*
	 //timeILS = 6;

	 finalSol = ils.search(timeILS, target);

	 cout << "ILS HAS ENDED!" << endl;
	 finalSol->second.print();
	 //finalSol->first.print();

	 RepMODM repFinal = finalSol->first.getR();
	 //finalSol = g.search(time,target);

	 //cout << eval.getAverageTime() << endl;
	 //cout << eval.getAverageTimeEvalComplete() << endl;

	 double fo = finalSol->second.evaluation();
	 int isFeasible = finalSol->second.isFeasible();

	 FILE* fResults = fopen(output.c_str(), "w");

	 fprintf(fResults, "%.7f \t %d \t%f\t%f\t %ld", fo, isFeasible, alphaBuilder, alphaNeighARProduct, seed);
	 fprintf(fResults, "\n Solution");
	 for (int c = 0; c < p.getNumberOfClients(); c++)
	 {
	 fprintf(fResults, "\n", fo, isFeasible);
	 for (int product = 0; product < p.getNumberOfProducts(); product++)
	 {
	 fprintf(fResults, "%d\t", repFinal[c][product]);
	 }
	 }

	 fprintf(fResults, "\n");

	 fclose(fResults);

	 FILE* fGeral = fopen(outputGeral.c_str(), "a");

	 size_t pos = filename.find("Instances/");
	 string instName = filename.substr(pos);

	 fprintf(fGeral, "%s\t%.7f\t%d \t %f\t%f\t%ld\n", instName.c_str(), fo, isFeasible, alphaBuilder, alphaNeighARProduct, seed);

	 fclose(fGeral);

	 */
	cout << "Programa terminado com sucesso!" << endl;
	return 0;
}

