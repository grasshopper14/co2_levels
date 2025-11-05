import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as ss

dpts = pd.read_csv('~/Downloads/data_2025-10-20.csv',
                   skip_blank_lines=True,
                   skiprows=1)
dpts['CO2 Year']=dpts['CO2 Date'].apply(lambda x:x[:4]).astype('int')
dfyear = dpts['CO2 PPM'].groupby(dpts['CO2 Year']).mean()
xyear = dfyear.index.to_numpy()
yyear = dfyear.to_numpy()

plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.45
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize']='small'
plt.rcParams['figure.autolayout']=True

xy = dict(zip(xyear,yyear))
xall = np.arange(xyear[0],xyear[-1]+1)
# Linear interpolation
j=0
for i in xall:
    if i not in xy.keys():
        xy[i]=xy[i-1]+(yyear[j]-yyear[j-1])/(xyear[j]-xyear[j-1])
    else:
        j+=1
xy=dict(sorted(xy.items()))
t = np.array(list(xy.keys()))
N = len(t);L = t[-1]-t[0]
yvals = np.array(list(xy.values()))
lin = ((yvals[-1]-yvals[0])/(t[-1]-t[0]))*(t-t[0])
dynamics = yvals - lin
ak = np.fft.rfft(dynamics)
freqs = np.fft.rfftfreq(len(dynamics))*N
amps = np.abs(ak)
phases = np.arctan2(-ak.imag, ak.real)
times = t[:, None]-t[0]
freqs = np.arange(N // 2 + 1) / L
out = (amps / N) * np.cos((2 * np.pi * freqs) * times - phases)
out[:, 1:(-1 if N%2 == 0 else None)] *= 2  
out = out.T  
topk=N//2
fig, ax = plt.subplots()
cumout = np.cumsum(out, axis=0)
ax.plot(xyear[142:],yyear[142:],'ok',fillstyle='none',alpha=0.3,markersize=3)
ax.plot(t,cumout[topk-1]+lin,'k',lw=0.5)
ax.legend(['Data','Fourier'+'\n'+'Representation'])
ax.set_xlabel('Year');ax.set_ylabel('CO$_2$ concentration in ppm')
plt.show()

#Get Fourier components
fig, ax = plt.subplots()
ax.plot(t,lin,'k',alpha=0.5,label=r'$\bar{A}t$')
for i in range(topk//5):
        ax.plot(t,out[i],'--k',alpha=0.5,
            label="Sinusoids" if i == 0 else "",)
ax.legend()
ax.set_xlabel('Year');ax.set_ylabel('CO$_2$ concentration in ppm')
plt.show()

#Spectral amplitude distribution
coeffs=np.abs(ak)/N
coeffs[1:(-1 if N%2 == 0 else None)] *= 2
freqs = np.arange(len(coeffs)) / L
fig, ax = plt.subplots()
ax.plot(freqs[:topk],np.abs(coeffs[:topk]),'k',lw=0.3)
ax.stem(freqs[:topk],np.abs(coeffs[:topk]),linefmt="k",markerfmt="ko",
        basefmt=" ",label="samples",use_line_collection=True)
ax.set_ylabel(r"$|A_k|$");ax.set_xlabel('frequency')
plt.show()
#obtain strength of persistence
dt=1
plt.plot(np.log(freq)[:6],np.log(S)[:6],'ok');
plt.xlabel(r'$\log$ frequency');plt.ylabel(r'$\log$ power spectral density');

m,b=np.polyfit(np.log(freqs[1:topk])[:6],
               np.log(np.abs(coeffs[1:topk])**2*dt**2/L)[:6],1)
print(m,b)
plt.show()
