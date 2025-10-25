import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gen_curve_fitting import sup_fit,fn_c,fnder
from lmfit import minimize, Parameters, fit_report, Model

def fn_sum(params,xax,ams):
    supmod = 0
    a = params['a']
    for i in range(ams+1):
        supmod+=params['p'+str(i)]*(fn_c(xax-xc[i],a,
                params['m'+str(i)])+yc[i])
    return supmod
def fn_cparts(x,ap,m):
    xdep = (-27*x*m/2/ap+np.sqrt(729*(x*m/ap)**2+108/ap**3)/2)
    return np.array([-xdep**(1./3)/3,xdep**(-1./3)/ap])

def fn_sumparts(params,xax,ams):
    supmod = 0
    for i in range(ams+1):
        supmod+=params['p'+str(i)]*(fn_cparts(xax-xc[i],params['a'],params['m'+str(i)]
                     )+yc[i])
    return supmod

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
#---------------------------------------------------------------
fig,ax = plt.subplots(1,1)
fig.tight_layout()
ams=3
x = xyear[:-1]; y = yyear[:-1]
##x = xyear[200:]; y = yyear[200:]
yeval,params = sup_fit(x,y,ams)
xc=params['xc'];yc=params['yc']
ax.plot(x,y,'.k',alpha=0.3,fillstyle='none');
ax.plot(x,yeval,'k',lw=0.5)
ax.legend(['Data','Fit'])
ax.set_xlabel('Year')
ax.set_ylabel(r'$CO_2$ levels in ppm')
##ax.plot(xc,yc,'dk',fillstyle='none')
##xx=np.concatenate([x,np.linspace(x[-1],x[-1]+100,101)[1:]])
##ax2.plot(xx,fn_sum(params,xx,ams))
fig.tight_layout();plt.show()
fig,ax = plt.subplots(1,1)
s1s2=fn_sumparts(params,x,ams)
ax.plot(x,s1s2[0],'--k',label=r'$S_I$')
ax.plot(x,s1s2[1],'k',label=r'$S_{II}$')
ax.set_xlabel('Year')
ax.legend()
ax.set_yticklabels([])
ax.annotate("1948", xy=(1948, 622), xytext=(1948, 590),
        arrowprops=dict(arrowstyle="->"))
fig.tight_layout()
plt.show()

x=xyear[281:]-xyear[-1]
y=yyear[281:]-yyear[-1]
model = Model(fn_c)
pars_bounded = model.make_params(ap=dict(value=0.1, min=1e-9),
                               m=dict(value=1))
out = model.fit(y,pars_bounded,x=x)
m = out.values['m']
a = out.values['ap']
xax = np.linspace(x[0],np.max(x)+50,500)
fig,ax = plt.subplots(1,1)
ax.plot(x+xyear[-1],y+yyear[-1],'ok',fillstyle='none',
         markersize=5,alpha=0.5)
ax.plot(xax+xyear[-1],fn_c(xax,a,m)+yyear[-1],'--k');
ax.set_xlabel('Year')
ax.set_ylabel(r'$CO_2$ levels in ppm')
ax.annotate("450 ppm at 2033", xy=(2033, 450), xytext=(2033, 400),
        arrowprops=dict(arrowstyle="->"))
ax.annotate("point of"+'\n'+ "inflection at 2025", xy=(2025, 427),
            xytext=(2030, 350),ha='center',arrowprops=dict(arrowstyle="->"))

fig.tight_layout()
plt.show()
