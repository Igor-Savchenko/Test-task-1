import statistics
import pickle

import matplotlib.pyplot as plt

from task import generate_percentile_distribution

# load initial distribution
FILE_NAME='Test-task-1\\set_of_10-day_returns_1000000_trials.data'
with open(FILE_NAME, 'rb') as filehandle:
    timeseries_of_10_day_returns=pickle.load(filehandle)

# Generate distributions with a difference numbers of trials
distributions=[]
for i in range(100, 1001, 100):
    dist=generate_percentile_distribution(timeseries_of_10_day_returns, number_of_trials=i)
    distributions.append(dist)
for i in range(1000, 10001, 1000):
    dist=generate_percentile_distribution(timeseries_of_10_day_returns, number_of_trials=i)
    distributions.append(dist)
for i in range(10000, 100001, 10000):
    dist=generate_percentile_distribution(timeseries_of_10_day_returns, number_of_trials=i)
    distributions.append(dist)
# Calculate statistical metrics for all distributions
arr_Number_of_trials = []
arr_mean = []
arr_median = []
arr_std_dev = []
arr_variation = []
for dist in distributions:
    arr_Number_of_trials.append(len(dist))
    arr_mean.append(statistics.mean(dist))
    arr_median.append(statistics.median(dist))
    arr_variation.append(statistics.variance(dist))
    arr_std_dev.append(statistics.stdev(dist))
# Plot dependences of metrics from the number of trial
fig1, p1=plt.subplots()
plt.title('The dependence of the mean value from the number of trials')
plt.xlabel('Number of trials')
plt.ylabel('Mean')
p1.plot(arr_Number_of_trials, arr_mean, 'k-', lw=1, 
            label='The dependence of the mean value from the number of trials')

fig2, p2=plt.subplots()
plt.title('The dependence of the median value from the number of trials')
plt.xlabel('Number of trials')
plt.ylabel('Median')
p2.plot(arr_Number_of_trials, arr_median, 'k-', lw=1, 
            label='The dependence of the median value from the number of trials')

fig3, p3=plt.subplots()
plt.title('The dependence of the variance value from the number of trials')
plt.xlabel('Number of trials')
plt.ylabel('Varianec')
p3.plot(arr_Number_of_trials, arr_variation, 'k-', lw=1, 
            label='The dependence of the variance value from the number of trials')

fig4, p4=plt.subplots()
plt.title('The dependence of the standard deviation value from the number of trials')
plt.xlabel('Number of trials')
plt.ylabel('Standard deviation')
p4.plot(arr_Number_of_trials, arr_std_dev, 'k-', lw=1, 
            label='The dependence of the standard deviation value from the number of trials')
plt.show()
