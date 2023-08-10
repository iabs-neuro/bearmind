import numpy as np
import pandas as pd
import os

def DetectSpikes(fanme, opts):
    #Fits calcium activity data in given .csv file with function single_spike_model
    #i.e., y(i) = ampl*(1 - exp((t - x(i))/t_on))*exp((t - x(i))/t_off) + backgr, x(i) >=t
    #      y(i) = backgr, x(i) < t.
    #Fitting begins whenever the signal cross threshold level, and is
    #restricted to definite time window, designed to catch fast spike rise,
    #peak and beginning of slow decay.
    #
    #arguments:
    #path, filename of .csv file with data
    #thr - threshold level for spike detection, MADs above background
    #t_before - window interval before threshold crossing, s
    #t_after - window interval after nearest peak, s
    #max_t_on - upper limit of e-fold rise time, s 
    #min_t_off - lower limit of e-fold decay time, s 
    #toler - fitting tolerance, 0..1
    #j_wind  - peak smoothing window half-interval, frames (1 = skip peak smoothing)'
    #bckg_med_wind - window for median filter for background calculation, frames (1 = flat background, old version)
    #
    # Vladimir Sotskov, 2017-2023
'''
    prompt = ['Threshold level, MADs', 'Window interval before threshold crossing, s', 'Window interval after nearest peak, s', 'Maximal e-fold rise time, s', 'Minimal e-fold decay time, s', 'Fitting tolerance, 0..1', 'Peak smoothing window half-time, frames (1 = skip peak smoothing)', 'Background median filter window, frames (1 = flat background)','Write spike amplitudes (vs ones) - y/n', 'Write auxiliary tables (FITS, THRES) - y/n']
    variables = ['thr', 't_before', 't_after', 'max_t_on', 'min_t_off', 'toler', 'j_wind', 'bckg_med_wind', 'sp_ampl', 'aux']
    default_data = ['4','1','1','1','0.5','0.8','5', '500', 'y', 'y']

'''
#reading data
T = csvread(sprintf('%s%s',path,filename), 1);
dim = size(T);
X = T(1:dim(1),1);
fps = round((dim(1) - 1)/(X(dim(1))-X(1)));

#main spikes array
SPIKES = zeros(dim(1)+1,dim(2)); 
SPIKES(2:dim(1)+1,1) = X;

#array of fits
FITS = zeros(dim(1)+1,dim(2)); 
FITS(2:dim(1)+1,1) = X;

#array of background level
BACKG = zeros(dim(1)+1,dim(2)); 
BACKG(2:dim(1)+1,1) = X;

#array of threshold level
THRES = zeros(dim(1)+1,dim(2)); 
THRES(2:dim(1)+1,1) = X;

#array of scam (unaccounted events)
SCAM = zeros(dim(1)+1,dim(2)); 
SCAM(2:dim(1)+1,1) = X;

#array of true values (unaccounted events)
CORR = zeros(dim(1)+1,dim(2)); 
CORR(2:dim(1)+1,1) = X;

h = waitbar(0, sprintf('Processing trace %d of %d', 0,  dim(2)-1)); 

for i = 2:dim(2)
    waitbar((i-1)/(dim(2)-1), h, sprintf('Processing trace %d of %d', i-1,  dim(2)-1));
    m_dev = mad(T(1:dim(1),i),1);
    #background calculation
    if bckg_med_wind == 1
        BACKG(2:dim(1)+1,i) = median(T(1:dim(1),i));
    else
        BACKG(2:dim(1)+1,i) = medfilt1(T(1:dim(1),i), bckg_med_wind);
    end
    #background subtraction: from now, all bckg is set to 0    
    T(1:dim(1),i) = T(1:dim(1),i) - BACKG(2:dim(1)+1,i);
    CORR(1:dim(1),i) = T(1:dim(1),i);
    
    #threshold levels
    THRES(2:dim(1)+1,i) = m_dev*thr;
    
    for j = j_wind+1:dim(1)
        #condition of threshold crossing
        if T(j,i) > m_dev*thr && T(j-1,i) <= m_dev*thr
            
            #nearest peak searching
            ampl =  mean(T(j-j_wind:min(dim(1),j+j_wind),i));
            j_peak = j+1;
            while j_peak + j_wind <= dim(1) && mean(T(j_peak-j_wind:j_peak+j_wind,i)) > ampl
                ampl = mean(T(j_peak-j_wind:j_peak+j_wind,i));
                BACKG(j_peak,i) = ampl;
                j_peak = j_peak + 1;
            end

            #fitting
            j_start = round(max(1, j - t_before*fps));
            j_end = round(min(dim(1), j_peak + t_after*fps));
            
            BACKG(j_start:j, i) = m_dev*2;
            BACKG(j_peak:j_end, i) = m_dev*2;
            
            [fitresult, gof] = sd_spike_fit_zero(X(j_start:j_end), T(j_start:j_end, i), m_dev*thr, max_t_on, min_t_off);
              
         
            if gof.rsquare >= toler # && gof.sse <= 0.000025*(j_end-j_start)
                t = fitresult.t;
                t_on = fitresult.t_on;
                t_off = fitresult.t_off;
                ampl = fitresult.ampl;
                Y = sd_spike_model_zero(X, t, t_on, t_off, ampl);
                #subtracting fit from original data - this allows next
                #potential spikes to be scored
                T(1:dim(1),i) = T(1:dim(1),i) - Y;
                #FITS saves cumulative spike fits
                FITS(2:dim(1)+1,i) = FITS(2:dim(1)+1,i) + Y;
                #spike scoring
                for jt = j_start:j_end
                    if t < X(jt)
                        break
                    end
                end
                if sp_ampl == 'y'
                    SPIKES(jt, i) = max(FITS(j_start:j_end, i));
                else
                    SPIKES(jt, i) = 1;
                end
                fprintf('Spike detected in trace %d\tt = %.2f s\t ampl = %.3f\tt_on = %.2f s\tt_off = %.2f s\tgof = %.2f\n',i-1,t,ampl,t_on,t_off,gof.rsquare); 
            else
                #one isolated point will not alter fit significantly, but
                #it can allow next potential over-thresholded spike to be scored
                T(j_end,i) = m_dev*thr;
                fprintf('Spike NOT detected in trace %d\tt = %.2f s\tgof = %.2f\n',i-1,fitresult.t,gof.rsquare);
                
                #just for information, what this spike could be like
                #Y = sd_spike_model_zero(X, fitresult.t, fitresult.t_on, fitresult.t_off, fitresult.ampl); 
                #SCAM(j_start:j_end, i) = Y(j_start:j_end)+ m_dev;
            end    
        end  
    end
end
csvwrite(sprintf('%sspikes_%s',path,filename), SPIKES);
if aux == 'y'
    csvwrite(sprintf('%sfits_%s',path,filename), FITS);
    csvwrite(sprintf('%sbackg_%s',path,filename), BACKG);
    csvwrite(sprintf('%sthres_%s',path,filename), THRES);
    csvwrite(sprintf('%sscam_%s',path,filename), SCAM);
    csvwrite(sprintf('%scorr_%s',path,filename), CORR);
end
delete(h);

end