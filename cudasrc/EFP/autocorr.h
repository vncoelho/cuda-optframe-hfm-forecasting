#ifndef AUTOCORR_H
#define AUTOCORR_H

#include <vector>
#include <stdlib.h>

#include "lregress.h"

class acorrInfo
{
private:
	acorrInfo(const acorrInfo &rhs);
	std::vector<double> points_;
	/** slope of the autocorrelation line on a log/log plot */
	double slope_;
	double slopeErr_;
public:
	acorrInfo()
	{
		slope_ = 0;
	}
	~acorrInfo()
	{
	}

	double slope()
	{
		return slope_;
	}
	void slope(double s)
	{
		slope_ = s;
	}

	double slopeErr()
	{
		return slopeErr_;
	}
	void slopeErr(double sE)
	{
		slopeErr_ = sE;
	}

	std::vector<double> &points()
	{
		return points_;
	}
};
// class acorrInfo

/**
 Calculate the autocorrelation function for a vector


 <h4>
 Copyright and Use
 </h4>

 You may use this source code without limitation and without
 fee as long as you include:

 <p>
 <i>
 This software was written and is copyrighted by Ian Kaplan, Bear
 Products International, www.bearcave.com, 2001.
 </i>
 </p>

 This software is provided "as is", without any warranty or
 claim as to its usefulness.  Anyone who uses this source code
 uses it at their own risk.  Nor is any support provided by
 Ian Kaplan and Bear Products International.

 Please send any bug fixes or suggested source changes to:

 <pre>
 iank@bearcave.com
 </pre>

 */
class autocorr
{
private:
	autocorr(const autocorr &rhs);
	// Minimum correlation value
	const double limit_;
	// Total number of points to calculate
	const size_t numPoints_;

public:

	/**
	 A container for the autocorrelation function result.
	 */

private:
	//void acorrSlope(acorrInfo &info);

	void acorrSlope(acorrInfo &info)
	{

		const size_t len = info.points().size();
		if (len > 0)
		{
			lregress::lineInfo regressInfo;
			lregress lr;
			vector<lregress::point> regressPoints;
			for (size_t i = 0; i < len; i++)
			{
				double x = log(i + 1);
				double y = log((info.points())[i]);
				regressPoints.push_back(lregress::point(x, y));
			}
			lr.lr(regressPoints, regressInfo);
			info.slope(regressInfo.slope());
			info.slopeErr(regressInfo.slopeErr());
		}
	} // acorrSlope


public:
	autocorr(double lim = 0.01, size_t numPts = 32) :
			limit_(lim), numPoints_(numPts)
	{
	}
	~autocorr()
	{
	}

	// Autocorrelation function
	//void ACF(const double *v, const size_t N, acorrInfo &info);

	void ACF(const double *v, const size_t N, acorrInfo &info)
	{
		if (!info.points().empty())
		{
			info.points().clear();
		}

		// The devMean array will contain the deviation from the mean.
		// That is, v[i] - mean(v).
		double *devMean = new double[N];

		double sum;
		size_t i;

		// Calculate the mean and copy the vector v, into devMean
		sum = 0;
		for (i = 0; i < N; i++)
		{
			sum = sum + v[i];
			devMean[i] = v[i];
		}
		double mean = sum / static_cast<double>(N);

		// Compute the values for the devMean array.  Also calculate the
		// denominator for the autocorrelation function.
		sum = 0;
		for (i = 0; i < N; i++)
		{
			devMean[i] = devMean[i] - mean;
			sum = sum + (devMean[i] * devMean[i]);
		}
		double denom = sum / static_cast<double>(N);

		// Calculate a std::vector of values which will make up
		// the autocorrelation function.
		double cor = 1.0;
		for (size_t shift = 1; shift <= numPoints_ && cor > limit_; shift++)
		{
			info.points().push_back(cor);
			size_t n = N - shift;
			sum = 0.0;
			for (i = 0; i < n; i++)
			{
				sum = sum + (devMean[i] * devMean[i + shift]);
			}
			double numerator = sum / static_cast<double>(n);
			cor = numerator / denom;
		}

		// calculate the log/log slope of the autocorrelation
		acorrSlope(info);

		//delete [] m;
	} // ACF

};
// autocorr

#endif
