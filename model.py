import os
import emcee
import triangle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# See http://dan.iel.fm/posts/mixture-models/
data_dir = '/home/ilya/Dropbox/petya'
data_file = 'Total_rate_vs_Years_v2.txt'
df = pd.read_table(os.path.join(data_dir, data_file), delim_whitespace=True,
                   names=['exper', 'band', 'date', 'time', 'st1', 'st2',
                          'rate_off1', 'rate_off2', 'total_rate', 'snr'])
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
# Convert datetime to timedelta
df['timedelta'] = df['datetime'] - sorted(df['datetime'])[0]
df['timedelta_sec'] = [dt.total_seconds() for dt in df['timedelta']]
df['total_rate'] *= 3. * 10 ** 10

# convenience variables
t = np.array(df['timedelta_sec'])
y = np.array(df['total_rate'])

# Define the probabilistic model...
# Parameters: a, w, fi_0, ab, b, yerr, Q, M, lnV
# a - amplitude of periodic component
# w - frequency of periodic component
# fi_0 - initial phase of periodic component
# ab - slope of linear trend
# b - intercept of linear trend
# yerr - std of noise distribution
# Q - probability that point comes from foreground distribution (not outlier)
# M - mean of gaussian outlier distribution
# lnV - ln of dispersion of outlier distribution

# A simple uniform prior:
bounds = [(0.0, 5.0), (0.0, 10 ** (-5)), (0, 2 * np.pi),
          (-10. ** (-7), 10. ** (-7)), (-3.0, 3.0), (0.0, 5.0), (0.0, 1.0),
          (-5.0, 5.0), (-3., 3.)]


def lnprior(p):
    if not all([b[0] < v < b[1] for v, b in zip(p, bounds)]):
        return -np.inf
    return 0


# The "foreground" linear likelihood:
def lnlike_fg(p):
    a, w, fi_0, ab, b, yerr, _, _, _ = p
    model = a * np.cos(w * t + fi_0) + ab * t + b
    return -0.5 * (((model - y) / yerr) ** 2 + 2 * np.log(yerr))


# The "background" outlier likelihood:
def lnlike_bg(p):
    _, _, _, _, _, yerr, Q, M, lnV = p
    var = np.exp(lnV) + yerr ** 2
    return -0.5 * ((M - y) ** 2 / var + np.log(var))


# Full probabilistic model.
def lnprob(p):
    a, w, fi_0, ab, b, yerr, Q, M, lnV = p

    # First check the prior.
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf, None

    # Compute the vector of foreground likelihoods and include the q prior.
    ll_fg = lnlike_fg(p)
    arg1 = ll_fg + np.log(Q)

    # Compute the vector of background likelihoods and include the q prior.
    ll_bg = lnlike_bg(p)
    arg2 = ll_bg + np.log(1.0 - Q)

    # Combine these using log-add-exp for numerical stability.
    ll = np.sum(np.logaddexp(arg1, arg2))

    # We're using emcee's "blobs" feature in order to keep track of the
    # foreground and background likelihoods for reasons that will become
    # clear soon.
    return lp + ll, (arg1, arg2)


# Initialize the walkers at a reasonable location.
ndim, nwalkers = 9, 100
p0 = np.array([0.5, 10. ** (-6.7), 0.1, 10. ** (-8), -1.25, 0.2, 0.3, 0.0, 0.1])
p0 = emcee.utils.sample_ball(p0, [0.01, 10 ** (-8), 0.01, -10. ** (-9), 0.01,
                                  0.01, 0.01, 0.01, 0.01], size=nwalkers)

# Set up the sampler.
print "Setting up sampler..."
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

# Run a burn-in chain and save the final location.
print "Burning in..."
pos, _, _, _ = sampler.run_mcmc(p0, 500)

# Run the production chain.
sampler.reset()
print "Sampling..."
sampler.run_mcmc(pos, 1500)

# Plot marginalized posterior of period in years
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.hist(2 * np.pi / sampler.flatchain[::100, 1] / (365. * 24 * 60 * 60),
        bins=40, range=[1., 1.10], normed=True)
plt.xlabel('Period, [years]')
plt.ylabel('Posterior Probability')
fig.savefig("period_mcmc.png", bbox_inches="tight", dpi=300)

# Calculate class (bg/fg) membership probability for data points
norm = 0.0
post_prob = np.zeros(len(t))
for i in range(sampler.chain.shape[1]):
    for j in range(sampler.chain.shape[0]):
        ll_fg, ll_bg = sampler.blobs[i][j]
        post_prob += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
        norm += 1
post_prob /= norm


def model_(p, timestamps):
    a, w, fi_0, ab, b, yerr, _, _, _ = p
    timedelta = timestamps - sorted(df['datetime'])[0]
    t = np.array([dt.total_seconds() for dt in timedelta])
    return a * np.cos(w * t + fi_0) + ab * t + b


def model_tr(p, timestamps):
    a, w, fi_0, ab, b, yerr, _, _, _ = p
    timedelta = timestamps - sorted(df['datetime'])[0]
    t = np.array([dt.total_seconds() for dt in timedelta])
    return ab * t + b


t1 = sorted(df['datetime'])[0]
t2 = sorted(df['datetime'])[-1]
tt = pd.date_range(start=t1, end=t2, freq='D')
# It works for symmetric posteriors. Actually, we don't need single value.
p = np.mean(sampler.flatchain[::100, :], axis=0)
# p = [0.51, 1.885*10**(-7), 0.50, 0.771*10**(-8), -1.13, 0.153, 0.775, -0.21,
#      -2.04]

# Plot 100 samples from posterior for full model and linear trend
samples = sampler.flatchain
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for s in samples[np.random.randint(len(samples), size=99)]:
    ax.plot(tt, model_tr(s, tt), color="red", lw=1, alpha=0.1)
for s in samples[np.random.randint(len(samples), size=1)]:
    ax.plot(tt, model_tr(s, tt), color="red", lw=1, alpha=0.1,
            label='linear trend')
plt.ylabel("Total rate, [cm/s]")
ax.scatter(list(df['datetime']), y, marker="o", s=22, c=post_prob,
           cmap="gray_r", vmin=0, vmax=1, zorder=1000)
plt.gcf().autofmt_xdate()

for s in samples[np.random.randint(len(samples), size=99)]:
    ax.plot(tt, model_(s, tt), color="#4682b4", lw=1, alpha=0.1)
for s in samples[np.random.randint(len(samples), size=1)]:
    ax.plot(tt, model_(s, tt), color="#4682b4", lw=1, alpha=0.1,
            label='full model')
plt.legend(loc=2)
fig.savefig("mcmc_fit.png", bbox_inches="tight", dpi=300)

# Plot corner-plot of samples
fig, axes = plt.subplots(nrows=ndim, ncols=ndim)
fig.set_size_inches(25.5, 25.5)
# plt.rcParams.update({'axes.titlesize': 'small'})
triangle.corner(sampler.flatchain[::10, :], labels=['A', 'w', 'fi', 'a', 'b',
                                                    'err', 'Prob', 'M', 'lnV'],
                fig=fig)
fig.savefig('corner_plot.png', bbox_inches='tight', dpi=300)
