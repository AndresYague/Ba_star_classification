import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

df_cross = pd.read_csv('cross-sections/lastZr90.dat', header=None, names=['temp', 's_zr90'], delim_whitespace=True)
df_cross.set_index('temp')
files = ['cross-sections/lastZr91.dat', 'cross-sections/lastZr92.dat', 'cross-sections/lastZr93.dat', 'cross-sections/lastZr94.dat', 'cross-sections/lastZr93_30lower.dat']
cols = ['s_zr91', 's_zr92', 's_zr93', 's_zr94', 's_zr93_30']
for ii in range(len(files)):
    df_now = pd.read_csv(files[ii], header=None, names=['temp', 'xx'], delim_whitespace=True)
    df_cross = df_cross.join(df_now.set_index('temp'), on='temp', how='outer')
    df_cross.rename(columns = {'xx': cols[ii]}, inplace=True)


df_cross['temp'] = df_cross['temp']/8.62e-5
alpha = (1/df_cross['s_zr90']+1/df_cross['s_zr91']+1/df_cross['s_zr92']+1/df_cross['s_zr94'])
df_cross['om1'] = df_cross['s_zr93']*alpha
df_cross['om2'] = df_cross['s_zr93_30']*alpha
relerr = 0.05
#df_cross['om1_relerr'] = relerr * (1 + alpha*(df_cross['s_zr90']+df_cross['s_zr91']+df_cross['s_zr92']+df_cross['s_zr94']))
#df_cross['om1_err0'] = df_cross['om1_relerr']*df_cross['om1']
df_cross['om1_err'] = (relerr*df_cross['s_zr93']*np.sqrt(alpha**2 + 1/df_cross['s_zr90']**2 + 1/df_cross['s_zr91']**2) +
                       1/df_cross['s_zr92']**2 + 1/df_cross['s_zr94']**2)
df_cross['om2_err'] = (relerr*df_cross['s_zr93_30']*np.sqrt(alpha**2 + 1/df_cross['s_zr90']**2 + 1/df_cross['s_zr91']**2) +
                       1/df_cross['s_zr92']**2 + 1/df_cross['s_zr94']**2)

def conv_r_to_sigma(r,temp):
    return r/(43817805*np.sqrt(temp)*6.022e-4)*1e6
s_from_r93 = conv_r_to_sigma(np.array([20.0, 17.9, 16.5, 15.5, 14.5, 14.1, 13.5, 11.9, 12.2, 11.6, 11.3]),
                                         np.array([5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100])).T
print(df_cross['om1'])

df_oms_Ba = pd.read_csv('intercept_om.dat', na_values='""')


# PLOTTING --------------------------------------------------------------------------
fig = plt.figure(figsize=(12,8.7))
font = {'family': 'sans-serif', 'size': 25}
matplotlib.rc('font', **font)
tt = df_cross['temp'].div(1e8)
kwargplot = {'capsize': 5, 'elinewidth': 2, 'markeredgewidth': 2}
plt.errorbar(tt, df_cross['om1'], yerr=df_cross['om1_err'], marker='o', color='teal', label="Tagliente original", **kwargplot)
plt.errorbar(tt, df_cross['om2'], yerr=df_cross['om2_err'],  marker='s', color='crimson', label="Tagliente modified", **kwargplot)
#plt.scatter(tt, df_cross['s_zr94'])
for elem in df_oms_Ba['om']:
    #plt.axhline(elem, ls='-', alpha=1, color='grey', lw=.4)
    print()

plt.xlim(min(tt)-0.2, max(tt)+0.2)
plt.ylabel(r'$\omega^*$')
plt.xlabel('T [$10^8\,$K]')
plt.legend()

ax = plt.gca()
ax.locator_params(axis='x', nbins=20)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.xaxis.set_tick_params(direction="in", length=6)
ax.yaxis.set_tick_params(direction="in", length=6)
plt.tight_layout()
plt.savefig('omega-T', bbox_inches="tight")
plt.show()