#ifndef LREGRESS_H_
#define LREGRESS_H_

#include <vector>

using namespace std;

/**
 Linear regression and related functions

 The lregress class packages the lr function, which calculates a
 linear regression line given paired vectors of X and Y points.


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
class lregress
{
private:
	lregress(const lregress &rhs);

public:
	lregress()
	{
	}
	~lregress()
	{
	}

	/**
	 A container for an {x,y} point of type double.
	 */
	class point
	{
	private:
		double x_, y_;
	public:
		/** constructor */
		point(double xVal, double yVal)
		{
			x_ = xVal;
			y_ = yVal;
		}
		/** copy constructor */
		point(const point &rhs)
		{
			x_ = rhs.x_;
			y_ = rhs.y_;
		}
		/** destructor does nothing */
		~point()
		{
		}
		;

		double x()
		{
			return x_;
		}
		void x(const double v)
		{
			x_ = v;
		}
		double y()
		{
			return y_;
		}
		void y(const double v)
		{
			y_ = v;
		}
	}; // class point

	/**
	 Line information.

	 A container for linear regression information.

	 A regression line, defined by the equation y' = a + bx, where a and
	 b are constants.  The constant a is the y intercept when x == 0.
	 The constant b is the slope of the line.

	 */
	class lineInfo
	{
	private:
		/** the slope of the line (e.g., b in y' = a + bx) */
		double slope_;
		/** the y-intercept (e.g., a, when x == 0 in y' = a + bx) */
		double intercept_;
		/** standard deviation of the points around the line */
		double stddev_;
		/** Regression error */
		double slopeError_;
		/** correlation between the x points and the y points */
		double cor_;
	public:
		/** constructor */
		lineInfo()
		{
			slope_ = 0;
			intercept_ = 0;
			cor_ = 0;
			stddev_;
			slopeError_ = 0;
		}
		~lineInfo()
		{
		}
		/** copy constructor */
		lineInfo(const lineInfo &rhs)
		{
			slope_ = rhs.slope_;
			intercept_ = rhs.intercept_;
			cor_ = rhs.cor_;
			stddev_ = rhs.stddev_;
			slopeError_ = rhs.slopeError_;
		}

		double slope()
		{
			return slope_;
		}
		void slope(double s)
		{
			slope_ = s;
		}

		double intercept()
		{
			return intercept_;
		}
		void intercept(double a)
		{
			intercept_ = a;
		}

		double stddev()
		{
			return stddev_;
		}
		void stddev(double s)
		{
			stddev_ = s;
		}

		double corr()
		{
			return cor_;
		}
		void corr(double c)
		{
			cor_ = c;
		}

		double slopeErr()
		{
			return slopeError_;
		}
		void slopeErr(double e)
		{
			slopeError_ = e;
		}
	}; // class lineInfo


	//double meanX(std::vector<point> &data) const;

	double meanX( vector<lregress::point> &data ) const
	{
	  double mean = 0.0;
	  double sum = 0.0;

	  const size_t N = data.size();
	  size_t i;
	  for (i = 0; i < N; i++) {
	    sum = sum + data[i].x();
	  }
	  mean = sum / N;
	  return mean;
	} // meanX


	//double meanY(std::vector<point> &data) const;

	double meanY( vector<lregress::point> &data ) const
	{
	  double mean = 0.0;
	  double sum = 0.0;

	  const size_t N = data.size();
	  size_t i;
	  for (i = 0; i < N; i++) {
	    sum = sum + data[i].y();
	  }
	  mean = sum / N;
	  return mean;
	} // meanY


	/**
	 Calculate the linear regression on a set of N points,
	 {Xi, Yi}.
	 */
	//void lr(std::vector<point> &data, lineInfo &info) const;


	void lr( vector<lregress::point> &data,
			   lregress::lineInfo &info ) const
	{
	  const size_t N = data.size();
	  if (N > 0) {

	    double muX = meanX( data );

	    double muY = meanY( data );

	    //     N-1
	    //     ---
	    //     \   (Xi - meanX)(Yi - meanY)
	    //     /__
	    //     i=0
	    // b = -----------------------------
	    //     N-1
	    //     ---             2
	    //     \   (Xi - meanX)
	    //     /__
	    //     i=0
	    //

	    double SSxy = 0;
	    double SSxx = 0;
	    double SSyy = 0;
	    double Sx = 0;
	    double Sy = 0;
	    double Sxy = 0;
	    double SSy = 0;
	    double SSx = 0;
	    for (size_t i = 0; i < N; i++) {
	      Sx = Sx + data[i].x();
	      Sy = Sy + data[i].y();
	      Sxy = Sxy + (data[i].x() * data[i].y());
	      SSx = SSx + (data[i].x() * data[i].x());
	      SSy = SSy + (data[i].y() * data[i].y());
	      double subX = (data[i].x() - muX);
	      double subY = (data[i].y() - muY);
	      SSyy = SSyy + subY * subY;
	      SSxy = SSxy + subX * subY;
	      SSxx = SSxx + subX * subX;
	    }

	    // slope
	    double b = SSxy / SSxx;
	    // intercept
	    double a = muY - b * muX;

	    // standard deviation of the points
	    double stddevPoints = sqrt( (SSyy - b * SSxy)/(N-2) );

	    // Error of the slope
	    double bError = stddevPoints / sqrt( SSxx );

	    double r2Numerator = (N * Sxy) - (Sx * Sy);
	    double r2Denominator = ((N*SSx) - (Sx * Sx))*((N*SSy) - (Sy * Sy));
	    double r2 = (r2Numerator * r2Numerator) / r2Denominator;

	    double signB = (b < 0) ? -1.0 : 1.0;

	    double r = signB * sqrt( r2 );

	    info.corr( r );
	    info.stddev( stddevPoints );
	    info.slopeErr( bError );
	    info.slope( b );
	    info.intercept( a );
	  } // if N > 0

	} // lineInfo

};
// lregress

#endif
