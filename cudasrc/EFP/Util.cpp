#include "Util.h"

float sigmoid(float x)
{
	float exp_value;
	float return_value;

	/*** Exponential calculation ***/
	exp_value = exp((double) -x);

	/*** Final sigmoid value ***/
	return_value = 1 / (1 + exp_value);

	return return_value;
}

static bool compara(double d1, double d2)
{
	return d1 < d2;
}
