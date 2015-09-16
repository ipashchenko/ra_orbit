import os
import george
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import gaussian_process


data_dir = '/home/ilya/Dropbox/petya'
data_file = 'Total_rate_vs_Years_v2.txt'
df = pd.read_table(os.path.join(data_dir, data_file), delim_whitespace=True,
                   names=['exper', 'band', 'date', 'time', 'st1', 'st2',
                          'rate_off1', 'rate_off2', 'total_rate', 'snr'])
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
# Convert datetime to timedelta
df['timedelta'] = df['datetime'] - sorted(df['datetime'])[0]
df['timedelta'] = [int(dt.days) for dt in df['timedelta']]
df['total_rate'] = 3. * 10 ** 10 * df['total_rate']

ground_stations = set(df['st2'])
colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for i, ground_station in enumerate(ground_stations):
    ax.plot(df[df['st2'] == ground_station]['datetime'],
            df[df['st2'] == ground_station]['total_rate'],
            '.{}'.format(colors[i]), label=ground_station)
plt.legend(loc=2)
plt.gcf().autofmt_xdate()
fig.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for i, ground_station in enumerate(ground_stations):
    ax.plot(df[df['st2'] == ground_station]['datetime'],
            df[df['st2'] == ground_station]['total_rate'],
            '.{}'.format(colors[i]), label=ground_station)
plt.legend(loc=2)
plt.gcf().autofmt_xdate()
fig.show()

# # Fit data with GP regression
from george.kernels import CosineKernel, ConstantKernel
kernel = CosineKernel(365) + ConstantKernel(-1.)
gp = george.GP(kernel)
x = np.array(df[df['st2'] == 'ARECIBO']['timedelta']) +\
    np.random.normal(0., scale=0.5,
                     size=len(df[df['st2'] == 'ARECIBO']['timedelta']))
x = (x - np.mean(x)) / np.std(x)
gp.compute(x, yerr=10**(-3))

t = np.linspace(0, 900, 900)
mu, cov = gp.predict(df['total_rate'], t)
std = np.sqrt(np.diag(cov))

X = np.atleast_2d(np.array(df['timedelta'], dtype=float) +
                  np.random.normal(0., scale=0.1, size=len(df['timedelta']))).T

y = df['total_rate']
x = np.atleast_2d(np.linspace(0, 900, 1000)).T
gp = gaussian_process.GaussianProcess(regr='constant', theta0=1e-1, thetaL=1e-4,
                                      thetaU=1e-1, nugget=0.1)
gp.fit(X, y)
y_pred, sigma2_pred = gp.predict(x, eval_MSE=True)

plt.fill_between(x[:, 0], y_pred+sigma2_pred, y_pred-sigma2_pred, color="k",
                 alpha=0.1)
plt.plot(x[:, 0], y_pred+sigma2_pred, color="k", alpha=1, lw=0.25)
plt.plot(x[:, 0], y_pred-sigma2_pred, color="k", alpha=1, lw=0.25)
plt.plot(x[:, 0], y_pred, color="k", alpha=1, lw=0.5)
plt.plot(df['timedelta'], df['total_rate'], ".k")

