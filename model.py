import os
import emcee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_dir = '/home/ilya/Dropbox/petya'
data_file = 'Total_rate_vs_Years_v2.txt'
df = pd.read_table(os.path.join(data_dir, data_file), delim_whitespace=True,
                   names=['exper', 'band', 'date', 'time', 'st1', 'st2',
                          'rate_off1', 'rate_off2', 'total_rate', 'snr'])
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
# Convert datetime to timedelta
df['timedelta'] = df['datetime'] - sorted(df['datetime'])[0]
df['timedelta'] = [dt.total_seconds() for dt in df['timedelta']]
df['total_rate'] = 3. * 10 ** 10 * df['total_rate']

t = np.array(df['timedelta'])
y = df['total_rate']


# Define the probabilistic model...
# A simple prior:
bounds = [(0.0, 5.0), (0.0, 10 ** (-5)), (0, 2 * np.pi), (-2.0, 2.0),
          (-5.0, 5.0), (0.0, 1.0), (-5.0, 5.0), (-2., 2.)]
def lnprior(p):
    a, w, fi_0, b, yerr, Q, M, lnV = p
    if not all(b[0] < v < b[1] for v, b in zip(p, bounds)):
        return -np.inf
    return 0

# The "foreground" linear likelihood:
def lnlike_fg(p):
    a, w, fi_0, b, yerr, _, _, _ = p
    model = a * np.cos(w * t + fi_0) + b
    return -0.5 * (((model - y) / yerr) ** 2 + 2 * np.log(yerr))

# The "background" outlier likelihood:
def lnlike_bg(p):
    _, _, _, _, yerr, Q, M, lnV = p
    var = np.exp(lnV) + yerr ** 2
    return -0.5 * ((M - y) ** 2 / var + np.log(var))

# Full probabilistic model.
def lnprob(p):
    a, w, fi_0, b, yerr, Q, M, lnV = p

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
print "Initialization ..."
ndim, nwalkers = 8, 100
p0 = np.array([0.5, 10. ** (-6.7), 0.1, -0.75, 0.2, 0.3, 0.0, 0.1])
p0 = emcee.utils.sample_ball(p0, [0.01, 10 ** (-8), 0.01, 0.01, 0.01, 0.01,
                                  0.01, 0.01], size=nwalkers)

# Set up the sampler.
print "Setting up sampler..."
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

# Run a burn-in chain and save the final location.
print "Burn in..."
pos, _, _, _ = sampler.run_mcmc(p0, 500)

# Run the production chain.
sampler.reset()
print "Sampling..."
sampler.run_mcmc(pos, 1500)

plt.hist(sampler.flatchain[::100, 1], bins=40, range=[1.65*10**(-7),
                                                      1.9*10**(-7)])

norm = 0.0
post_prob = np.zeros(len(t))
for i in range(sampler.chain.shape[1]):
    for j in range(sampler.chain.shape[0]):
        ll_fg, ll_bg = sampler.blobs[i][j]
        post_prob += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
        norm += 1
post_prob /= norm

tt = np.linspace(0, 8 * 10**7, 10000)
p = [0.5, 1.775*10**(-7), 1.2, -0.72, 0.175, 0.8, -0.2, -2.]
a, w, fi_0, b, yerr, _, _, _ = p

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(tt, a * np.cos(w * tt + fi_0) + b, 'r', lw=2, label="T = 1.12 years")
plt.xlabel("time, [s]")
plt.ylabel("Total rate, [cm/s]")
ax.scatter(t, y, marker="o", s=22, c=post_prob, cmap="gray_r", vmin=0, vmax=1,
            zorder=1000)
plt.legend(loc=2)
# samples = sampler.flatchain
# for s in samples[np.random.randint(len(samples), size=14)]:
#     a, w, fi_0, b, yerr, _, _, _ = s
#     ax.plot(tt, a * np.cos(w * tt + fi_0) + b, 'r', lw=1, alpha=0.3)
fig.show()
fig.savefig("mcmc_fit.png", bbox_inches="tight")



