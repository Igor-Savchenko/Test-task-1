import statistics
import pickle

import matplotlib.pyplot as plt

from task import generate_percentile_distribution

# load initial distribution
FILE_NAME='Test-task-1\\set_of_10-day_returns_1000000_trials.data'
with open(FILE_NAME, 'rb') as filehandle:
    timeseries_of_10_day_returns=pickle.load(filehandle)

# Generate distributions with a given numbers of trials
NUMBER_OF_TRIALS=10000
distributions=[]
for i in range(100):
    dist=generate_percentile_distribution(timeseries_of_10_day_returns, number_of_trials=NUMBER_OF_TRIALS)
    distributions.append(dist)
arr_mean = []
arr_median = []
arr_std_dev = []
arr_variation = []

# Calculate statistical metrics for all distributions
for dist in distributions:
    arr_mean.append(statistics.mean(dist))
    arr_median.append(statistics.median(dist))
    arr_variation.append(statistics.variance(dist))
    arr_std_dev.append(statistics.stdev(dist))

fig1, p1=plt.subplots()
plt.title('The histogram of the mean value (number of trials = '+str(NUMBER_OF_TRIALS)+')')
plt.xlabel('Mean')
plt.ylabel('Amount')
p1.annotate(s='Mean = ' + str(round(statistics.mean(arr_mean), 2)) 
            + '\n SD = ' + str(round(statistics.stdev(arr_mean), 2)), xy=(-12500, 0.0012))
p1.hist(arr_mean, 
            bins=100, 
            density=True, 
            histtype='stepfilled', 
            alpha=0.5, 
            color='blue', 
            label='Mean = ' + str(statistics.mean(arr_mean)) + '\n Standard deviation'+str(statistics.stdev(arr_mean)))
# Plot histograms of metrics 
fig2, p2=plt.subplots()
plt.title('The histogram of the standard deviation (number of trials = '+str(NUMBER_OF_TRIALS)+')')
plt.xlabel('Standard deviation')
plt.ylabel('Amount')
p2.annotate(s='Mean = ' + str(round(statistics.mean(arr_std_dev), 2)) 
            + '\n SD = ' + str(round(statistics.stdev(arr_std_dev), 2)), xy=(6000, 0.001))
p2.hist(arr_std_dev, 
            bins=100, 
            density=True, 
            histtype='stepfilled', 
            alpha=0.5, 
            color='blue', 
            label='Mean = ' + str(statistics.mean(arr_std_dev)) + '\n Standard deviation'+str(statistics.stdev(arr_std_dev)))
plt.show()