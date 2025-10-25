import numpy as np
from lmfit import minimize, Parameters

def fn_c(x,ap,m):
    xdep = (-27*x*m/2/ap+np.sqrt(729*(x*m/ap)**2+108/ap**3)/2)
    return xdep**(-1./3)/ap-xdep**(1./3)/3
def fnder(x,params):
    derval=0;a=params['a']
    xc = params['xc'];yc = params['yc']
    for i in range(len(xc)):
        derval+=params['p'+str(i)]*params['m'+
            str(i)]/(1+3*a*fn_c(x-xc[i],a,params['m'+str(i)])**2)
    return derval
    
def origins(x,y,ams):
    grads=(y[1:]-y[:-1])/(x[1:]-x[:-1])
    mxpts = np.argsort(np.abs(grads))[-ams-1:][::-1]
    xl = np.empty(ams+1)
    yl = np.empty(ams+1)
    for i in np.arange(len(mxpts)):
        xl[i] = (x[mxpts[i]]+x[mxpts[i]+1])/2
        yl[i] = (y[mxpts[i]]+y[mxpts[i]+1])/2
    return xl,yl,grads[mxpts]
def sup_fit(x,y,ams):
    xax = (x - np.mean(x))/np.std(x)
    yax = (y - np.mean(y))/np.std(y)
    xc,yc,ics = origins(xax,yax,ams)
    params = Parameters()
    params.add('a',value=1,min=1e-6)
    for i in range(ams+1):
        params.add('p'+str(i),value=0)
        params.add('m'+str(i),value=1)
    def fn_sum(params,xax,yax):
        supmod = 0
        a = params['a']
        for i in range(ams+1):
            supmod+=params['p'+str(i)]*(fn_c(xax-xc[i],a,
                    params['m'+str(i)])+yc[i])
        return yax-supmod
    out = minimize(fn_sum, params, args=(xax,yax,),)
    outdict = out.params.valuesdict()
    outdict['a'] = outdict['a']/np.std(y)**2
    for i in range(ams+1):
        outdict['m'+str(i)] = outdict['m'+str(i)]*np.std(y)/np.std(x)
    xc=xc*np.std(x)+np.mean(x);yc=yc*np.std(y)+np.mean(y)
    offset = np.mean(y)-np.sum([outdict['p'+str(i)]
                        for i in range(ams+1)])*np.mean(y)
    outdict['xc']=xc;outdict['yc']=yc
    return y-fn_sum(outdict,x,y)+offset,outdict
