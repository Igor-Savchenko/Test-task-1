""" Base module of test task 

Task: Monte-Carlo Simulation

Task part: Generate distribution of 0.01 quantile (1% percentile) of 10-days 
overlapping proportional returns obtained from the 3-years timeseries (750 
observations) of 1-day returns. Original timeseries is generated using stable 
distribution with the following parameters: alpha = 1.7; beta = 0.0; gamma = 
1.0; delta = 1.0.
"""
# As I understand it is necessary to form the desired distribution using some 
# method of obtaining curves 10 daily returns. Since the method is not 
# explicitly specified in the task, I chose the most popular bootstrap method
# with the sample size coinciding with the volume of the original sample.


# Import the required libraries, numpy for calculations, scipy.stats.levi_stable 
# for stable distribution generation, scipy.stats.normal for normal distribution,
# matplotlib for result demonstration, pickle for data serealization, statistics 
# for statistic calculations 
import pickle
import statistics

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import levy_stable
from scipy.stats import norm


# Some code fragments are designed as a function for the convenience of subsequent
# use and export. It is supposed to use them in other modules.
def calculate_n_days_returns(timeseries_of_1_day_returns, n):
    """ The function calculate_n_days_returns - used to calculate a set of n-day returns
    from a set of 1-day returns. 
    Parameters:
    timeseries_of_1_day_returns - a set of 1-day returns
    n - number of days in period for return calculation
    """
    # To calculate the prices, we need to set the initial value of the price. It can be
    # arbitrary, because it will reduсed when calculating the n-day returns. Verified.
    p=[1,]

    for item in timeseries_of_1_day_returns:
        p.append(p[-1]*(item+1))
    n_days_returns=[]
    i=0

    while (i+n)<len(p):
        n_days_returns.append(p[i+n]/p[i]-1)
        i+=1
        
    return np.array(n_days_returns)


def generate_percentile_distribution(initial_collection, n_samples=None, percentile=1, number_of_trials=100):
    """ The function generate_percentile_distribution - generate the distribution 
    required by the condition of the task with a given number of samples. 
    Parameters:
    initial_collection - initial distribution (set, sample)
    n_samples - number of exemplars in botstrap sets
    percentile - percentile 
    number_of_trials - number of botstrap sets
    """
    # the size of bootstrap samle coinciding with the volume of the original sample by default
    if not n_samples:
        n_samples=len(initial_collection)

    result_distribution=[]

    for i in range(number_of_trials):
        result_distribution.append(np.percentile(np.random.choice(
            initial_collection, size=n_samples, replace=True), 1))

    return result_distribution


# The code inside the block will not be executed during the execution of the import
# statement
if __name__ == '__main__':
    # Set constants of stable distribution
    ALPHA = 1.7
    BETA = 0
    GAMMA = 1
    DELTA = 1
    # Generate timeseries of 1-day returns. 
    timeseries_of_1_day_returns = levy_stable.rvs(ALPHA, BETA, GAMMA, DELTA, 750)
    # Calculation of 10-day returns
    timeseries_of_10_day_returns=calculate_n_days_returns(timeseries_of_1_day_returns,10)
    # Calculation of the required distribution of 1 percentile
    NUMBER_OF_TRIALS=1000001
    distribution_of_1_percentile=generate_percentile_distribution(timeseries_of_10_day_returns, 
                                                                    number_of_trials=NUMBER_OF_TRIALS)
    # Save the necessary data
    file_name_for_10_day_returns_set='Test-task-1\\set_of_10-day_returns_'+str(NUMBER_OF_TRIALS)+'_trials.data'
    file_name_for_distribution_of_1_percentile='Test-task-1\\distribution_of_1_percentile_'+str(NUMBER_OF_TRIALS)+'_trials.data'
    with open(file_name_for_10_day_returns_set, 'wb') as filehandle:
        pickle.dump(timeseries_of_10_day_returns, filehandle)
    with open(file_name_for_distribution_of_1_percentile, 'wb') as filehandle:
        pickle.dump(distribution_of_1_percentile, filehandle)
    # Create histogram of 1-day returns
    fig1, p1=plt.subplots()
    plt.title('Histogram of 1-day returns')
    plt.xlabel('1-day return')
    plt.ylabel('Probability')
    p1.hist(timeseries_of_1_day_returns, 
            bins=100, 
            density=True, 
            histtype='stepfilled', 
            alpha=0.5, 
            color='blue', 
            label='1-day returns')
    # print a probability density function
    x = np.linspace(levy_stable.ppf(0.01, ALPHA, BETA, GAMMA, DELTA), 
                    levy_stable.ppf(0.99, ALPHA, BETA, GAMMA, DELTA), 
                    100)
    rv = levy_stable(ALPHA, BETA, GAMMA, DELTA)
    p1.plot(x, rv.pdf(x), 'k-', lw=1, 
            label='probability density function of stable distribution')
    # Create histogram of 10-day returns
    fig2, p2=plt.subplots()
    plt.title('Histogram of 10-day returns')
    plt.xlabel('10-day return')
    plt.ylabel('Probability')
    p2.hist(timeseries_of_10_day_returns, 
            bins=100, 
            density=True, 
            histtype='stepfilled', 
            alpha=0.5, 
            color='blue',
            label='10-day returns')
    # Create histogram of 1-percentiles
    fig3, p3=plt.subplots()
    plt.title('''Histogram of 0.01 quantile (1% percentile) of 10-days returns
            obtained from the 3-years timeseries of 1-day returns.''')
    plt.xlabel('1% percentile')
    plt.ylabel('Probability')
    p3.hist(distribution_of_1_percentile, 
            bins=100, 
            density=True, 
            histtype='stepfilled', 
            alpha=0.5, 
            color='blue')
    # print a probability density function
    MEAN = statistics.mean(distribution_of_1_percentile)
    SIGMA = statistics.stdev(distribution_of_1_percentile)
    x = np.linspace(norm.ppf(0.01, MEAN, SIGMA), 
                    norm.ppf(0.99, MEAN, SIGMA), 
                    100)
    rv = norm(loc=MEAN, scale=SIGMA)
    p3.plot(x, rv.pdf(x), 'k-', lw=1, 
            label='probability density function of stable distribution')
    # Display diagrams
    plt.show()