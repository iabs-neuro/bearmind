function y = sd_spike_model_zero(x, t, t_on, t_off, ampl)
%3 Aug Vova: Modified from single_spike_model, bckg  now is strictly zero

%01 Nov 2017 Vitya: added changes for negative virus, 15str
%Function which models single calcium spike
%t - time of spike start
%t_on - rise parameter
%t_off - decay parameter
%ampl - amplitude of spike
%
%Vladimir Sotskov, 2017-2020

y = zeros(size(x));

for i = 1:length(x)
    if x(i) < t
        y(i) = 0;
    else
        y(i) = ampl*(1 - exp((t - x(i))/t_on))*exp((t - x(i))/t_off);
    end
end
end