/*
 * fuzzylogic.cpp
 *
 *  Created on: Jan 9, 2012
 *      Author: dcapra
 */

#include "fuzzylogic.h"
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <stdio.h>

using namespace cv;

struct Quality_index_pair {
  double quality;
  int index;
};

struct ByQuality {
    bool operator ()(Quality_index_pair const& a, Quality_index_pair const& b) {
        return a.quality < b.quality;
    }
};

vector<int> fuzzy(double distance[], double angle[], int points) {

        vector<double> distance_vector(distance, &distance[points]);
        vector<double> angle_vector(angle, &angle[points]);

	// sort errors
	std::sort(distance_vector.begin(), distance_vector.end());
	std::sort(angle_vector.begin(), angle_vector.end());
	int mid = points / 2;
	double Md = 0;
	double Ma= 0;

	// calculate median
	// if odd number of elements, take middle one
	if (points &  1) {
		Md = distance_vector[mid];
		Ma = angle_vector[mid];
	}
	// if even number of elements, take the average of the two middle ones
	else {
		Md = (distance_vector[mid] + distance_vector[mid+1]) / 2;
		Ma = (angle_vector[mid] + angle_vector[mid + 1]) / 2;
	}

	vector<Quality_index_pair> ranking;

	// calculate quality index for each error pair
	for (int i = 0; i < points; i++) {
		double d = distance[i]/Md; // distance error scaled by median
		double a = angle[i]/Ma; // angle error scaled by median
		double high_d = 0;
		double low_d = 0;
		double medium_d = 0;
		double high_a = 0;
		double low_a = 0;
		double medium_a = 0;	       

		// find how much of distance error is in each set (high, medium, low)
		if (d < .3) {
		  high_d = 1;
		}
		else if (d < 1) {
		  high_d = 1 - (d-.3)*10/7;
		  medium_d = (d-.3)*10/7;
		}
		else if (d < 1.7) {
		  medium_d = 1 - (d-1)*10/7;
		  low_d = (d-1)*10/7;
		}
		else {
		  low_d = 1;
		}

		// find how much of angle error is in each set (high, medium, low)
		if (a < .3) {
		  high_a = 1;
		}
		else if (a < 1) {
		  high_a = 1 - (a-.3)*10/7;
		  medium_a = (a-.3)*10/7;
		}
		else if (a < 1.7) {
		  medium_a = 1 - (a-1)*10/7;
		  low_a = (a-1)*10/7;
		}
		else {
		  low_a = 1;
		}
		
		
		// calculate how much of the pair is in each set (excellent, good, medium, bad)
		double excellence_score = high_d*high_a; // if both are high, then in excellent
		double good_score = high_d*medium_a + medium_d*high_d; // if one is high and one is medium, then in good
		double medium_score = medium_d*medium_a; // if both are medium then in medium
		double bad_score = low_d + low_a*medium_d+low_a*high_a; // if one is low, then both in bad
		
		// find weighted quality index
		/*
		double excellence_score = min(high_d, high_a);
		double good_score = max(min(high_d, medium_a), min(medium_d, high_a));
		double medium_score = min(medium_d, medium_a);
		double bad_score = max(low_d, low_a);
		*/
		Quality_index_pair pair;
		pair.quality = excellence_score + 0.75*good_score + 0.5*medium_score;
		pair.index = i;
		ranking.push_back(pair);
	}

	// sort quality indices and find the best 40%
	std::sort(ranking.begin(), ranking.end(), ByQuality());

	vector<int> indices;

	// find the indices of the errors that are most significant
	for (int i = (points*3)/5; i < points; i++) {
		indices.push_back(ranking[i].index);
	}

	return indices;

}
