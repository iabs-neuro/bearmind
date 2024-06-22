from bm_examinator import colornum_Metro as clnm
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
    bpl.output_file(fname.replace('.csv','_events.html'))
    p = bpl.figure(title = fname.split('\\')[-1].split('traces')[0], height = 1000, width = 1800)#, width_policy = 'fit')
    for cell_num, trace in enumerate(traces):
        #smoothing
        sm_trace = gaussian_filter1d(trace, sigma=opts['sigma'])
        thr = opts['thr']*mad(sm_trace)
        x_peaks = signal.argrelmax(sm_trace)[0]
        x_pits = signal.argrelmin(sm_trace)[0]
        evs = []  #list of this cell's events
        #plotting
        p.line(time, trace/np.max(trace) + cell_num, line_color = clnm(cell_num))
        if opts['draw_details']:
            p.line(time, sm_trace/np.max(trace) + cell_num, line_color = clnm(cell_num), line_dash = 'dashed', line_alpha = 0.7)
            p.scatter(time[x_peaks], trace[x_peaks]/np.max(trace) + cell_num, line_color = None, fill_color = clnm(cell_num), fill_alpha = 0.5, marker = 'inverted_triangle', size = 8)
            p.scatter(time[x_pits], trace[x_pits]/np.max(trace) + cell_num, line_color = None, fill_color = clnm(cell_num), fill_alpha = 0.5, marker = 'triangle', size = 8)
        for x_peak in x_peaks:
            x_left = 0 if all (x_pits > x_peak) else x_pits[x_pits < x_peak][-1]
            x_right = len(trace)-1 if all (x_pits < x_peak) else x_pits[x_pits > x_peak][0]
            if trace[x_peak] - trace[x_left] >= thr:
                #estimated values and bounds of fitting params; a,b,t0,ton,toff
                p0 = (thr, trace[x_left], (time[x_left] + time[x_peak])/2, opts['est_ton'], opts['est_toff'])
                bounds=((thr, -np.inf, time[x_left], 0, 0), (np.inf, np.inf, time[x_peak], np.inf, np.inf)) #((l,o,w,e,r,_),(h,i,g,h,e,r))
                try:
                    popt,_ = curve_fit(EventForm, time[x_left:x_right], trace[x_left:x_right], p0 = p0, bounds = bounds)  
                except:
                    try:
                        p0 = (thr, trace[x_left], time[x_left], opts['est_ton'], opts['est_toff'])
                        popt,_ = curve_fit(EventForm, time[x_left:x_right], trace[x_left:x_right], p0 = p0, bounds = bounds)
                    except:
                        print(f'FAILED to detect event at cell {cell_num} time {time[x_peak]} s')
                        if opts['draw_details']:
                            p.scatter((time[x_left] + time[x_peak])/2, (trace[x_left] + trace[x_peak])/2/np.max(trace) + cell_num, marker = 'circle_x', line_color = clnm(cell_num), fill_color = None, size = 15)
                        continue
                fit = EventForm(time[x_left:x_right], *popt)
                ampl = np.max(fit) - popt[1]  #relative amplitude of the event
                idx = len(time[time < popt[2]]) #position of t0 in time array
                #save, print and plot
                evs.append(dict(zip(['cell_num','ampl','a','b','t0','ton','toff','x_left','x_right'],[cell_num,ampl,*popt,x_left,x_right])))
                spikes[cell_num, idx] = ampl  
                print(f'Event detected: {evs[-1]}')
                p.scatter(popt[2], np.max(fit)/np.max(trace) + cell_num, line_color = None, fill_color = clnm(cell_num), size = 5)
                if opts['draw_details']:
                    p.line(time[x_left:x_right], fit/np.max(trace)+cell_num, line_color = clnm(cell_num), line_width =3.0, line_alpha = 0.5)
                    
        events.append(evs)
        
    pd.DataFrame(np.concatenate(([time],spikes)).T).to_csv(fname.replace('traces','spikes'), index=False, header = ['time_s', *np.arange(len(traces))])
    with open(fname.partition('traces.csv')[0] + 'events.pickle', "wb") as f:
        pickle.dump(events, f)
    bpl.show(p)

      
def DrawSpEvents(tr_fname, sp_fname):
    #file reading
    bpl.output_file(tr_fname.replace('.csv','_events.html'))
    traces = np.genfromtxt(tr_fname, delimiter = ',', skip_header = 1)[:,1:].T #note the transposition for better iterability
    sp_events = np.genfromtxt(sp_fname, delimiter = ',', skip_header = 1)[:,1:].T
    time = np.genfromtxt(tr_fname, delimiter = ',', skip_header = 1)[:,0]

    p = bpl.figure(title = tr_fname.split('\\')[-1], width = 1800)#, width_policy = 'fit')
    for cell_num, (trace, spikes) in enumerate(zip(traces, sp_events)):
        p.line(time, trace/np.max(trace) + cell_num, line_color = clnm(cell_num), line_width = 2.0)
        p.scatter(time[spikes>0], cell_num - 0.1, line_color = None, fill_color = clnm(cell_num), size = 5)
    bpl.show(p)


def DrawSpEvents2(tr_fname, sp_fname1, sp_fname2):
    #file reading
    bpl.output_file(tr_fname.replace('.csv','_events.html'))
    traces = np.genfromtxt(tr_fname, delimiter = ',', skip_header = 1)[:,1:].T #note the transposition for better iterability
    sp_events1 = np.genfromtxt(sp_fname1, delimiter = ',', skip_header = 1)[:,1:].T
    sp_events2 = np.genfromtxt(sp_fname2, delimiter = ',', skip_header = 1)[:,1:].T
    time = np.genfromtxt(tr_fname, delimiter = ',', skip_header = 1)[:,0]

    p = bpl.figure(title = tr_fname.split('\\')[-1], width = 1800)#, width_policy = 'fit')
    for cell_num, (trace, spikes1, spikes2) in enumerate(zip(traces, sp_events1, sp_events2)):
        p.line(time, trace/np.max(trace) + cell_num, line_color = clnm(cell_num), line_width = 2.0)
        p.scatter(time[spikes1>0], cell_num - 0.1, line_color = None, fill_color = clnm(cell_num), size = 5)
        p.scatter(time[spikes2>0], cell_num - 0.2, line_color = None, fill_color = clnm(cell_num), marker = 'star', size = 8)
    bpl.show(p)
