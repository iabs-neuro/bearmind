function [fitresult, gof] = sd_spike_fit_zero(x, y, thr, max_t_on, min_t_off)

%Fitting data [x, y] (must be columns!) with function single_spike_model
%i.e., y(i) = ampl*(1 - exp((t - x(i))/t_on))*exp((t - x(i))/t_off) , x(i) >=t
%      y(i) = 0, x(i) < t.
%arguments:
%
%thr - threshold level for spike detection, a.u. above background (usually n MADs)
%max_t_on - upper limit of e-fold rise time, s 
%min_t_off - lower limit of e-fold decay time, s 
%
%returns the same values as std fit function does
%
%Vladimir Sotskov, 2017-2020

[xData, yData] = prepareCurveData(x, y);
% Setting up missing parameters
len = length(x);
max_ampl = max(y)*2;
min_t_on = 0.01;
max_t_off = 50;

if max_ampl - thr < 0.1
    max_ampl = max_ampl + 0.1;
end

% Setting up fittype and options.
ft = fittype( 'sd_spike_model_zero(x, t, t_on, t_off, ampl)', 'independent', 'x', 'dependent', 'y' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
%Alphabet!! [ampl, backg, t, t_off, t_on]

if thr >= 0
    opts.Lower = [thr, x(1),  min_t_off, min_t_on];
    opts.Upper = [max_ampl, x(len), max_t_off, max_t_on];
else
    opts.Lower = [max_ampl, x(1), min_t_off, min_t_on];
    opts.Upper = [thr, x(len), max_t_off, max_t_on];
end
opts.StartPoint = [thr, x(1), min_t_off, max_t_on];    
% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, opts );

end