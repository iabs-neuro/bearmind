% A pipeline for calcium spike detection
% csv traces -> spike traces & plots
%
% Vladimir Sotskov, 2020

%% Parameters

path = 'C:\SCRIPTS\SpikeDetector\';  %Your path to data folder
fname = 'sample_traces2.csv';


thr_mad = 10;            %Threshold (MADs)
t_before = 1;           %Time (s) taken into account for the spike fitting before the threshold crossing 
t_after = 1;            %Time (s) taken into account for the spike fitting after the first peak after the threshold crossing
max_t_on = 1.0;         %Maximal e-fold spike rise time (s)
min_t_off = 0.5;        %Minimal e-fold spike decay time (s)
toler = 0.8;            %Minimal goodness_of_fit (r-square) 
j_wind = 20;             %Smooth parameter (half-width of the window for mean filter, frames) for the-first-peak-searching
bckg_med_wind = 500;    %Window for median filter for background calculation, frames (1 = flat background, old version)
sp_ampl = 'y';          %To write or not (y/n) spike amplitudes (if no, all ampls = 1)
aux = 'n';              %To write or not (y/n) auxilary tables

%% Spike detection
   
warning('off','MATLAB:table:RowsAddedNewVars') %it's really annoying
sd_spike_detector(path, fname, thr_mad, t_before, t_after, max_t_on, min_t_off, toler, j_wind, bckg_med_wind, sp_ampl, aux)
warning('on','MATLAB:table:RowsAddedNewVars')
    
%% Draw traces and spikes

sd_draw_traces(path, fname, strcat('spikes_',fname))