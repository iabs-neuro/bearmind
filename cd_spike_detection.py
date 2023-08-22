from cd_inspector_callbacks import colornum_Metro
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import median_abs_deviation as mad
from scipy.optimize import curve_fit
from scipy import signal
import bokeh.plotting as bpl
import pandas as pd
import numpy as np
import pickle

def EventForm(xdata, a, b, t0, ton, toff):
    return [b if t < t0 else a*(1 - np.exp((t0 - t)/ton))*np.exp((t0 - t)/toff) + b for t in xdata]

def FitEvents(fname, opts):
    #file reading
    traces = np.genfromtxt(fname, delimiter = ',', skip_header = 1)[:,1:].T #note the transposition for better iterability
    time = np.genfromtxt(fname, delimiter = ',', skip_header = 1)[:,0]
    spikes = np.zeros(traces.shape)   #this in for non-sparse storage like in traces.csv, as it used to be, just for back compartibility
    events = []                       #this is for sparse storage in pickle 
    for cell_num, trace in enumerate(traces):
        #smoothing
        sm_trace = gaussian_filter1d(trace, sigma=opts['sigma'])
        thr = opts['thr']*mad(sm_trace)
        x_peaks = signal.argrelmax(sm_trace)[0]
        x_pits = signal.argrelmin(sm_trace)[0]
        loc_events = []
               
        for x_peak in x_peaks:
            x_left = 0 if all (x_pits > x_peak) else x_pits[x_pits < x_peak][-1]
            x_right = len(trace)-1 if all (x_pits < x_peak) else x_pits[x_pits > x_peak][0]
            if sm_trace[x_peak] - sm_trace[x_left] >= thr:
                try:
                    popt, pcov = curve_fit(EventForm, time[x_left:x_right], trace[x_left:x_right], p0 = [thr, trace[x_left], time[x_left], opts['est_ton'], opts['est_toff']])  
                    loc_events.append(dict(zip(['cell_num','a','b','t0','ton','toff','x_left','x_right'],[cell_num, *popt, x_left, x_right])))
                    idx = len(time[time < popt[2]]) #position of t0 in time array
                    spikes[cell_num, idx] = sm_trace[x_peak] - sm_trace[x_left]
                    print(f'Spike detected: {loc_events[-1]}')
                except:
                    continue

        events.append(loc_events)
        
    pd.DataFrame(np.concatenate(time[None,:],spikes).T).to_csv(fname.partition('traces.csv')[0] + 'spikes.csv', index=False, header = ['time_s', *np.arange(len(traces))])
    with open(fname.partition('traces.csv')[0] + 'events.pickle', "wb") as f:
        pickle.dump(events, f)
      

      
def DrawSpEvents(tr_fname, sp_fname):
	#file reading
	bpl.output_notebook()
	bpl.output_file()
    traces = np.genfromtxt(tr_fname, delimiter = ',', skip_header = 1)[:,1:].T #note the transposition for better iterability
	sp_events = np.genfromtxt(sp_fname, delimiter = ',', skip_header = 1)[:,1:].T
    time = np.genfromtxt(tr_fname, delimiter = ',', skip_header = 1)[:,0]

	p = bpl.figure(title = tr_fname.split('\\')[-1])
	for cell_num, trace, spikes in enumerate(zip(traces, sp_events)):
		p.line(time, trace/np.max(trace) + cell_num, line_color = colornum_Metro(cell_num + 1))
		p.scatter(time[spikes>0], cell_num +0.9, fill_color = colornum_Metro(cell_num + 1))
	bpl.show(p)
