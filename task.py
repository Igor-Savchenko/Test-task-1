from scipy.stats import levy_stable
import matplotlib.pyplot as plt
import numpy as np
import statistics


ALPHA = 1.7
BETA = 0
GAMMA = 1
DELTA = 1


x = np.linspace(levy_stable.ppf(0.01, ALPHA, BETA, GAMMA, DELTA), levy_stable.ppf(0.99, ALPHA, BETA, GAMMA, DELTA), 100)
timeseries_of_1_day_returns = levy_stable.rvs(ALPHA, BETA, GAMMA, DELTA, 750)
rv = levy_stable(ALPHA, BETA, GAMMA, DELTA)


def calculate_n_days_returns(timeseries_of_1_day_returns, n):
    p=[1,]
    for item in timeseries_of_1_day_returns:
        p.append(p[-1]*(item+1))
    n_days_returns=[]
    i=0
    while (i+n)<len(p):
        n_days_returns.append(p[i+n]/p[i]-1)
        i+=1
    return np.array(n_days_returns)


def get_bootstrap_samples(collection, n_samples):
    indices=np.random.randint(0, len(collection), n_samples)
    samples=collection[indices]
    return samples
    

def generate_percentile_distribution(initial_collection, n_samples=None, percentile=1, number_of_trials=100):
    if not n_samples:
        n_samples=len(initial_collection)
    result_distribution=[]
    for i in range(number_of_trials):
        result_distribution.append(np.percentile(get_bootstrap_samples(initial_collection, n_samples), 1))
    return result_distribution


timeseries_of_10_day_returns=calculate_n_days_returns(timeseries_of_1_day_returns,10)
distribution_of_1_percentile=generate_percentile_distribution(timeseries_of_10_day_returns, number_of_trials=1000) 
counter=[('median','mode','mean')]
distributions=[]
percentile_1=[]
percentile_99=[]
for i in range(100, 1001, 100):
    dist=generate_percentile_distribution(timeseries_of_10_day_returns, number_of_trials=i)
    distributions.append(dist)
for i in range(1000, 10001, 1000):
    dist=generate_percentile_distribution(timeseries_of_10_day_returns, number_of_trials=i)
    distributions.append(dist)
for i in range(10000, 100001, 10000):
    dist=generate_percentile_distribution(timeseries_of_10_day_returns, number_of_trials=i)
    distributions.append(dist)
#for i in range(1000, 20000, 1000):
    #dist=generate_percentile_distribution(timeseries_of_10_day_returns, number_of_trials=i)
    #counter.append((statistics.median(dist), statistics.mode(dist), statistics.mean(dist)))
    #percentile_1.append(np.percentile(dist, 1))
    #percentile_99.append(np.percentile(dist, 99))
    #distributions.append(dist)


fig1, p1=plt.subplots()
fig2, p2=plt.subplots()
fig3, p3=plt.subplots()
fig4, p4=plt.subplots()
p1.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
p1.hist(timeseries_of_1_day_returns, bins=50, density=True, histtype='stepfilled', alpha=0.3, color='green')
p2.hist(timeseries_of_10_day_returns, bins=50, density=True, histtype='stepfilled', alpha=0.3, color='green')
p3.hist(distribution_of_1_percentile, bins=200, density=True, histtype='stepfilled', alpha=0.3, color='green')
for dist in distributions:
    print(len(dist), statistics.mean(dist), statistics.median(dist), statistics.stdev(dist), statistics.variance(dist))
plt.show()