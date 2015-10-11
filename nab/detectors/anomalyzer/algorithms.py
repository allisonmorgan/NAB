import pandas
import math
import numpy as np
import scipy.stats as ss
import py.test
from statsmodels.distributions.empirical_distribution import ECDF

def extract_windows(timeseries, referenceSize, activeSize):
	n = len(timeseries)
	activeSize = min(activeSize, n)
	referenceSize = min(referenceSize, n-activeSize)

	reference = []
	active = []

	reference = timeseries[n-activeSize-referenceSize:n-activeSize]
	active = timeseries[n-activeSize:]

	return reference, active

def test_extract_windows():
	x, y = extract_windows([0,1,2,3,4,5,6,7], 5, 3)
	assert x == [0,1,2,3,4]
	assert y == [5,6,7]


def cap(x, minimum, maximum):
	return max(min(x, maximum), minimum)

def test_cap():
	assert cap(1.2, 0, 1) == 1
	assert cap(-0.1, 0, 1) == 0
	assert cap(0.5, 0, 1) == 0.5

#"fence" <- https://github.com/lytics/anomalyzer/blob/master/algorithms.go#L73
def fence(timeseries, referenceSize, activeSize, upperBound, lowerBound):
	_, active = extract_windows(timeseries, referenceSize, activeSize)
	x = np.mean(active)

	distance = 0
	bound = 0 
	mid = 0
	if np.isnan(lowerBound):
		distance = x/upperBound
	else:
		bound = (upperBound-lowerBound)/2.0
		mid = lowerBound+bound
		distance = np.abs(x-mid)/bound

	return weight_exp(cap(distance, 0 , 1), 10) 

def weight_exp(x, base):
	return (math.pow(base, x) - 1)/(math.pow(base,1) -1)

def test_fence():
	assert fence([0.5,0.5,0.5,0.75,0.85,0.95,1], 3, 4, 1, 0) > 0
	assert fence([0.5,0.5,0.5,1,1,1,1], 3, 4, 1, 0) == 1
	assert fence([0.5,0.5,0.5,0,0,0,0], 3, 4, 1, 0) == 1
	assert fence([0.5,0.5,0.5,0.5,0.5,0.5,0.5], 3, 4, 1, 0) == 0
	assert fence([50,60,75,90,100], 1, 4, 100, np.nan) > 0

def rank_vector(timeseries):
	return ss.rankdata(np.array(timeseries))

def test_rank_vector():
	assert (rank_vector([1,2,3,4])-np.array([1,2,3,4])).all() == np.array([0,0,0,0]).all()
	assert (rank_vector([14,13,12,11])-np.array([4,3,2,1])).all() == np.array([0,0,0,0]).all()

#comparison is a function
def rank_test(timeseries, referenceSize, activeSize, comparison, permCount):
	ranks = rank_vector(timeseries)
	_, active = extract_windows(ranks, referenceSize, activeSize)

	activeSum = vector_sum(active)
	
	i = 0 
	significant = 0
	while i < permCount:
		np.random.shuffle(timeseries)
		permRanks = rank_vector(timeseries)
		_, permActive = extract_windows(permRanks, referenceSize, activeSize)

		permSum = vector_sum(permActive)
		if comparison(permSum, activeSum):
			significant = significant + 1

		i = i + 1

	return significant/permCount

def vector_sum(timeseries):
	total = 0
	for point in timeseries:
		total = total + point
	return total

def test_vector_sum():
	assert vector_sum([1,2,3,4]) == 10
	assert vector_sum([0,0,0,0]) == 0

def lessThan(x, y):
	if x < y:
		return True
	return False

def greaterThan(x, y):
	if x > y:
		return True
	return False

#"high rank" <- https://github.com/lytics/anomalyzer/blob/master/algorithms.go#L142
def regular_rank_test(timeseries, referenceSize, activeSize, permCount):
	return rank_test(timeseries, referenceSize, activeSize, lessThan, permCount)

def test_regular_rank_test():
	assert regular_rank_test([1,2,3,4,5,6,7,8,9], 6, 3, 10) == 1.0 
	assert regular_rank_test([7,8,9,4,5,6,1,2,3], 6, 3, 10) == 0.0 

#"low rank" <- https://github.com/lytics/anomalyzer/blob/master/algorithms.go#L146
def reverse_rank_test(timeseries, referenceSize, activeSize, permCount):
	return rank_test(timeseries, referenceSize, activeSize, greaterThan, permCount)

def test_reverse_rank_test():
	assert reverse_rank_test([1,2,3,4,5,6,7,8,9], 6, 3, 10) == 0.0 
	assert reverse_rank_test([7,8,9,4,5,6,1,2,3], 6, 3, 10) == 1.0 

#"cdf" <- https://github.com/lytics/anomalyzer/blob/master/algorithms.go#L209
def cdf_test(timeseries, referenceSize, activeSize):
	differences = np.abs(np.diff(timeseries))
	reference, active = extract_windows(differences, referenceSize - 1, activeSize)

	referenceEcdf = ECDF(reference)
	activeDiff = np.mean(active) - np.mean(reference)

	percentile = referenceEcdf(activeDiff)

	return 2*np.abs(0.5*percentile)

def test_cdf_test():
	assert cdf_test([1,2,3,4,5,6], 3, 3) == 0.0
	assert cdf_test([1,2,3,9,20,50], 3, 3) == 1.0

#"magnitude" <- https://github.com/lytics/anomalyzer/blob/master/algorithms.go#L231
def magnitude_test(timeseries, referenceSize, activeSize):
	reference, active = extract_windows(timeseries, referenceSize, activeSize)
	
	refMean = np.mean(reference)
	activeMean = np.mean(active)

	"if baseline is zero, then the magnitude should be infinite, but we'll round to one"
	if refMean == 0:
		if activeMean == 0:
			return 0
		else:
			return 1

	percentDiff = np.abs(activeMean-refMean) / refMean
	return percentDiff

def test_magnitude_test():
	assert magnitude_test([0,0,0,1,1,1], 3, 3) == 1.0
	assert magnitude_test([0,0,0,0,0,0], 3, 3) == 0.0
	assert magnitude_test([1,1,1,1.5,1.5,1.5], 3, 3) == 0.5

def ks_test(timeseries, referenceSize, activeSize):
	reference, active = extract_windows(timeseries, referenceSize, activeSize)

	n1 = len(reference)
	n2 = len(active)
	if n1%n2 != 0:
		return np.nan

	activeEcdf = ECDF(active)
	referenceEcdf = ECDF(reference)

	minimum = min(min(reference), min(active))
	maximum = max(max(reference), max(active))
	
	interpolated = interpolate(minimum, maximum, n1+n2)
	activeDist = activeEcdf(interpolated)
	referenceDist = referenceEcdf(interpolated)
	
	d = 0.0
	i = 0
	while i < (n1+n2):
		d = max(d, np.abs(activeDist[i]-referenceDist[i]))
		i = i+1

	return d

def test_ks_test():
	assert ks_test([1,2,1,2,1,2,1,2,1,2,1,2], 8, 4) == 0.0
	assert ks_test([1,2,1,2,1,2,1,2,1,10,20,50], 8, 4) > 0.0

def interpolate(minimum, maximum, npoints):
	interp = [0.0]*npoints
	step = (maximum-minimum) / float(npoints-1)

	interp[0] = minimum
	i = 1
	while i < npoints:
		interp[i] = interp[i-1]+step
		i = i + 1

	return interp

#"ks" <- https://github.com/lytics/anomalyzer/blob/master/algorithms.go#L255
def bootstrap_ks_test(timeseries, referenceSize, activeSize, permCount):
	dist = ks_test(timeseries, referenceSize, activeSize)
	if np.isnan(dist):
		return np.nan

	i = 0
	significant = 0

	while i < permCount:
		np.random.shuffle(timeseries)
		permutedDist = ks_test(timeseries, referenceSize, activeSize)

		if permutedDist < dist:
			significant = significant + 1
		i = i + 1

	return significant/permCount

def test_bootstrap_ks_test():
	assert bootstrap_ks_test([1,2,1,2,1,2,1,2,1,2,1,2], 8, 4, 10) == 0.0
	#assert bootstrap_ks_test([1,2,1,2,1,2,1,2,1,10,20,50], 8, 4, 10) > 0.0

#"diff" <- https://github.com/lytics/anomalyzer/blob/master/algorithms.go#L105
def diff_test(timeseries, referenceSize, activeSize, permCount):
	ranks = rank_vector(np.abs(np.diff(timeseries)))
	_, active = extract_windows(ranks, referenceSize - 1, activeSize)

	activeSum = vector_sum(active)
	i = 0
	significant = 0

	while i < permCount:
		np.random.shuffle(timeseries)
		permRanks = rank_vector(np.abs(np.diff(timeseries)))
		_, permActive = extract_windows(permRanks, referenceSize, activeSize)

		permSum = vector_sum(permActive)
		if permSum < activeSum:
			significant = significant + 1

		i = i + 1

	return float(significant)/float(permCount)

def test_diff_test():
	assert diff_test([1,1,1,1,1,1,1,1,1,1], 7, 3, 10) == 0.0
	assert diff_test([1,1,2,1,2,3,7,9,10,6], 7, 3, 10) > 0.0