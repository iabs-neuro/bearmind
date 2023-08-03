MATLAB toolbox for event detection in calcium imaging data (1P or 2P).

INPUTS

1) CSV tables (delimiter = ',') with df/f traces of the following type:
time(s) cell#1  cell#2  cell#3 ...
0.05    8.8734  6.7786  6.7864
0.10    5.6453  4.7567  6.6478    
...


OUTPUTS

1) tables with spikes (like (1) but with discrete spike values instead of traces)
2) plot with detected spikes and traces


Vladimir P. Sotskov, Viktor V. Plusnin 2017-2020