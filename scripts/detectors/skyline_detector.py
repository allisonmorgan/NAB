from anomaly_detector import AnomalyDetector

from skyline.algorithms import (median_absolute_deviation,
                                first_hour_average,
                                stddev_from_average,
                                stddev_from_moving_average,
                                mean_subtraction_cumulation,
                                least_squares,
                                histogram_bins)

class SkylineDetector(AnomalyDetector):

  def __init__(self, *args, **kwargs):

    # Initialize the parent
    super(SkylineDetector, self).__init__(*args, **kwargs)

    # Store our running history
    self.timeseries = []
    self.recordCount = 0
    self.algorithms =   [median_absolute_deviation,
                         first_hour_average,
                         stddev_from_average,
                         stddev_from_moving_average,
                         mean_subtraction_cumulation,
                         least_squares,
                         histogram_bins]

  def getOutputPrefix(self):
    return "skyline"

  def getThreshold(self):
    """
    Returns a known good threshold for the dataset. Discovered by using the
    optimize_threshold.py script.
    """

    return 0.9

  def handleRecord(self, inputData):
    """
    Returns a list [anomalyScore].
    """

    score = 0.0
    inputRow = [inputData["timestamp"], inputData["value"]]
    self.timeseries.append(inputRow)
    if self.recordCount < self.probationaryPeriod:
      self.recordCount += 1
    else:
      for algo in self.algorithms:
        score += algo(self.timeseries)

    averageScore = score / len(self.algorithms)
    return [averageScore]