#include "Representation.h"

ostream & operator<<(ostream & s, const RepEFP& rep)
{
	s << " ======================== \n Solution printing \n" << endl;
	s << "Single Inputs" << endl;
	vector<int> nFuzzyFunction(NFUZZYFUNCTIONS, 0);
	double counter = 0;

	for (int i = 0; i < rep.singleIndex.size(); i++)
	{
		//s << "(" << rep.singleIndex[i].first << "," << rep.singleIndex[i].second << ")" << endl;
		//s << "\t (" << rep.singleFuzzyRS[i][GREATER] << "->" << rep.singleFuzzyRS[i][GREATER_WEIGHT] << ")";
		//s << "\t (" << rep.singleFuzzyRS[i][LOWER] << "->" << rep.singleFuzzyRS[i][LOWER_WEIGHT] << ")";
		//s << "\t Epsilon:" << rep.singleFuzzyRS[i][EPSILON];
		//s << "\t FuzzyFunction:" << rep.singleFuzzyRS[i][PERTINENCEFUNC] << endl;

		if (rep.singleFuzzyRS[i][PERTINENCEFUNC] == Heavisde)
			nFuzzyFunction[Heavisde]++;
		if (rep.singleFuzzyRS[i][PERTINENCEFUNC] == Trapezoid)
			nFuzzyFunction[Trapezoid]++;
		counter++;
	}

	s << "Averaged Inputs" << endl;
	for (int i = 0; i < rep.averageIndex.size(); i++)
	{
		//s << "([" << rep.averageIndex[i][0].first << "," << rep.averageIndex[i][0].second << "]";
		for (int j = 1; j < rep.averageIndex[i].size(); j++)
		{
			//s << "\t [" << rep.averageIndex[i][j].first << "," << rep.averageIndex[i][j].second << "]";
		}
		//s << ")" << endl;
		//s << "\t (" << rep.averageFuzzyRS[i][GREATER] << "->" << rep.averageFuzzyRS[i][GREATER_WEIGHT] << ")";
		//s << "\t (" << rep.averageFuzzyRS[i][LOWER] << "->" << rep.averageFuzzyRS[i][LOWER_WEIGHT] << ")";
		//s << "\t Epsilon:" << rep.averageFuzzyRS[i][EPSILON];
		//s << "\t FuzzyFunction:" << rep.averageFuzzyRS[i][PERTINENCEFUNC] << endl;

		if (rep.averageFuzzyRS[i][PERTINENCEFUNC] == Heavisde)
			nFuzzyFunction[Heavisde]++;
		if (rep.averageFuzzyRS[i][PERTINENCEFUNC] == Trapezoid)
			nFuzzyFunction[Trapezoid]++;
		counter++;
	}

	s << "Derivative Inputs" << endl;
	for (int i = 0; i < rep.derivativeIndex.size(); i++)
	{
		//s << "([" << rep.derivativeIndex[i][0].first << "," << rep.derivativeIndex[i][0].second << "]";
		for (int j = 1; j < rep.derivativeIndex[i].size(); j++)
		{
			//s << "\t [" << rep.derivativeIndex[i][j].first << "," << rep.derivativeIndex[i][j].second << "]";
		}
		//s << ")" << endl;
		//s << "\t (" << rep.derivativeFuzzyRS[i][GREATER] << "->" << rep.derivativeFuzzyRS[i][GREATER_WEIGHT] << ")";
		//s << "\t (" << rep.derivativeFuzzyRS[i][LOWER] << "->" << rep.derivativeFuzzyRS[i][LOWER_WEIGHT] << ")";
		//s << "\t Epsilon:" << rep.derivativeFuzzyRS[i][EPSILON];
		//s << "\t FuzzyFunction:" << rep.derivativeFuzzyRS[i][PERTINENCEFUNC] << endl;

		if (rep.derivativeFuzzyRS[i][PERTINENCEFUNC] == Heavisde)
			nFuzzyFunction[Heavisde]++;
		if (rep.derivativeFuzzyRS[i][PERTINENCEFUNC] == Trapezoid)
			nFuzzyFunction[Trapezoid]++;
		counter++;
	}

	s << "earliestInput: " << rep.earliestInput << endl;
	s << "counter: " << counter << endl;
	s << "Heaviside functions on rules: " << nFuzzyFunction[Heavisde] / counter * 100 << endl;
	s << "Trapezoid functions on rules: " << nFuzzyFunction[Trapezoid] / counter * 100 << endl;

	if ((nFuzzyFunction[Heavisde] + nFuzzyFunction[Trapezoid]) != counter)
	{
		s << "BUGOU!" << endl;
	}

	s << " Solution printed \n ======================== \n";

	return s;
}


