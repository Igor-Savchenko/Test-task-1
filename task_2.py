""" Base module of test task 

Task: Monte-Carlo Simulation

Task part: Generate distribution of 0.01 quantile (1% percentile) of 10-days 
overlapping proportional returns obtained from the 3-years timeseries (750 
observations) of 1-day returns. Original timeseries is generated using stable 
distribution with the following parameters: alpha = 1.7; beta = 0.0; gamma = 
1.0; delta = 1.0.
"""

# Import the required libraries, numpy for calculations, scipy.stats.levi_stable 
# for stable distribution generation, matplotlib for result demonstration, 
# scipy.stats.ks_2samp for compute the Kolmogorov-Smirnov statistic on 2 samples

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import levy_stable
from scipy.stats import ks_2samp

def generate_10_days_returns(n):
    """ The function generate_10_days_returns - used to generate a set of 
    10-day returns. 
    Parameters:
    n - number of 1-day returns 
    """
    # Generate timeseries of 1-day returns. 
    ts_r_1 = levy_stable.rvs(1.7, 0, 1, 1, n)
    ts_r_10 = []
    for i in range(9,n,1):
        r_10 = ((ts_r_1[i]+1)*
            (ts_r_1[i-1]+1)*
            (ts_r_1[i-2]+1)*
            (ts_r_1[i-3]+1)*
            (ts_r_1[i-4]+1)*
            (ts_r_1[i-5]+1)*
            (ts_r_1[i-6]+1)*
            (ts_r_1[i-7]+1)*
            (ts_r_1[i-8]+1)*
            (ts_r_1[i-9]+1)-1)
        ts_r_10.append(r_10)           
    return np.array(ts_r_10)

def generate_percentile_distribution(n_samples=750, percentile=1, number_of_trials=100):
    """ The function generate_percentile_distribution - generate the distribution 
    required by the condition of the task with a given number of samples. 
    Parameters:
    n_samples - number of exemplars in sets of 1 days returns
    percentile - percentile 
    number_of_trials - number of Monte-Carlo trials
    """
    result=[]
    for _ in range(number_of_trials):
        result.append(np.percentile(generate_10_days_returns(n_samples), percentile))
    return np.array(result)

def remove_outliers_double_MAD(y,thresh=3.5):
    """ The function generate_percentile_distribution - removes the outliers 
    using double absolute deviation method 
    https://stackoverflow.com/questions/22354094/pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data  
    Parameters:
    y - distribution
    thresh - thresh
    """
    m = np.median(y)
    abs_dev = np.abs(y - m)
    left_mad = np.median(abs_dev[y <= m])
    right_mad = np.median(abs_dev[y >= m])
    y_mad = left_mad * np.ones(len(y))
    y_mad[y > m] = right_mad
    modified_z_score = 0.6745 * abs_dev / y_mad
    modified_z_score[y == m] = 0 
    return y[modified_z_score <= thresh]

# num - number of Monte-Carlo trials
num=3000
percentile_d1=generate_percentile_distribution(number_of_trials=num)
# generate another distribution with the same number of trials
percentile_d2=generate_percentile_distribution(number_of_trials=num)
# remove outliers from distributions
n_percentile_d1=remove_outliers_double_MAD(percentile_d1, 3.5)
n_percentile_d2=remove_outliers_double_MAD(percentile_d2, 3.5)
print(len(n_percentile_d1))
print(len(n_percentile_d2))
# Use KS test
ks_res=ks_2samp(n_percentile_d1, n_percentile_d2)
print(ks_res)
print(ks_2samp(percentile_d1, percentile_d2))
# create histogram of 1-percentiles
fig1, p1=plt.subplots()
plt.title('''Histogram of 0.01 quantile (1% percentile) of 10-days returns 
obtained from the 3-years timeseries of 1-day returns.''')
plt.xlabel('1% percentile')
plt.ylabel('Probability')
p1.hist(percentile_d1, 
        bins=100, 
        density=True, 
        histtype='stepfilled', 
        alpha=0.3, 
        color='blue')
# create histograms of two percentile distributions without outliers on 1 graph
fig2, p2=plt.subplots()
plt.title('''Comparison of histograms of two percentile distributions without outlers''')
plt.xlabel('1% percentile')
plt.ylabel('Probability')
p2.annotate(s='number of trials = '+
            str(num)+
            '\nks_2 \n'+
            'statistic = '+
            str(ks_res[0])+
            '\np-value = '+
            str(ks_res[1]), xy=(-80000, 0.00003))
p2.hist(n_percentile_d1, 
        bins=100, 
        density=True, 
        histtype='stepfilled', 
        alpha=0.3, 
        color='blue')
p2.hist(n_percentile_d2, 
        bins=100, 
        density=True, 
        histtype='stepfilled', 
        alpha=0.3, 
        color='red')
# Display diagrams
plt.show()

