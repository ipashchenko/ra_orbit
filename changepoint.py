import os
import pymc3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_dir = '/home/ilya/Dropbox/petya'
data_file = 'Total_rate_vs_Years.txt'
df = pd.read_table(os.path.join(data_dir, data_file), delim_whitespace=True,
                   names=['exper', 'band', 'date', 'time', 'st1', 'st2',
                          'rate_off1', 'rate_off2', 'total_rate', 'snr'])
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

# Plot ``total_rate`` for some antenna [cm/s]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(df[df['st2'] == 'ARECIBO']['datetime'],
        df[df['st2'] == 'ARECIBO']['total_rate'] * 3. * 10 ** 10, '.k')
fig.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(df['datetime'], df['total_rate'] * 3. * 10 ** 10, '.k')
fig.show()


# plot one plot for all stations
ground_stations = set(df['st2'])
colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for i, ground_station in enumerate(ground_stations):
    ax.plot(df[df['st2'] == ground_station]['datetime'],
            df[df['st2'] == ground_station]['total_rate'] * 3. * 10 ** 10,
            '.{}'.format(colors[i]), label=ground_station)
plt.legend()
fig.show()


# Plot several subplots for each ground stations
fig, axes = plt.subplots(nrows=len(ground_stations), ncols=1, sharex=True,
                         sharey=True)
fig.set_size_inches(18.5, 18.5)
plt.rcParams.update({'axes.titlesize': 'medium'})

for i, ground_station in enumerate(ground_stations):
    print i
    axes[i].plot(df[df['st2'] == ground_station]['datetime'],
                df[df['st2'] == ground_station]['total_rate'] * 3. * 10 ** 10,
                    '.{}'.format(colors[i]), label=ground_station)
    axes[i].legend(prop={'size': 11}, loc='best', fancybox=True,
                      framealpha=0.5)
fig.show()
#
# Fit changepoint problem
# Convert datetime to timedelta
df['timedelta'] = df['datetime'] - sorted(df['datetime'])[0]
df['timedelta'] = [int(dt.days) for dt in df['timedelta']]

# FIXME: Allow each station can have nonzero mean. Check it for both parts. Use
# hierarchical model.
# FIXME: Use t-distribution for likelihood
with pymc3.Model() as model:
    # Prior for distribution of switchpoint location
    switchpoint = pymc3.DiscreteUniform('switchpoint', lower=0,
                                        upper=max(df['timedelta']))
    # Priors for pre- and post-switch std
    early_std = pymc3.Exponential('early_std', lam=1.)
    late_std = pymc3.Exponential('late_std', lam=1.)

    # Allocate appropriate gaussian std to years before and after current
    # switchpoint location
    idx = np.arange(len(df['timedelta']))
    std = pymc3.switch(switchpoint >= idx, early_std, late_std)

    # Data likelihood
    total_rates = pymc3.Normal('total_rates', mu=0., sd=std,
                               observed=3. * 10 ** 10 * df['total_rate'])

with model:
    # Initial values for stochastic nodes
    start = {'early_std': 3., 'late_std': 0.5}
    # Use slice sampler for means
    step1 = pymc3.Slice([early_std, late_std])
    # Use Metropolis for switchpoint, since it accomodates discrete variables
    step2 = pymc3.Metropolis([switchpoint])

    tr = pymc3.sample(10000, tune=500, start=start, step=[step1, step2])