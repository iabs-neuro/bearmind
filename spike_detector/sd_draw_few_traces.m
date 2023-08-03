function sd_draw_few_traces(p, path, fname, sp_fname)
%9 Jan 2019 Vova: Thickness increased for traces with spikes
%
%Reads traces from csv files and plot them on the same figure, coloring them
%using "colornum" routine; user can define offset between traces at the
%Y-axis (in %DF/F)
%
% Vladimir Sotskov, 2017-2020

TRACES = readtable(strcat(path, fname));
SPIKES = readtable(strcat(path, sp_fname));

dim = size(TRACES);
X = TRACES{1:dim(1),1};
maxim = max(max(TRACES{1:dim(1),2:dim(2)}));
minim = min(min(TRACES{1:dim(1),2:dim(2)}));
absmax = max(max(abs(maxim), abs(minim)));
offset = 1.2;

%% traces drawing
w = waitbar(0, sprintf('Plotting trace %d of %d', 1,  dim(2)-1));    
hold on
for i = 1:length(p)
    waitbar((i-1)/(dim(2)-1), w, sprintf('Processing cell %d of %d', i-1,  dim(2)-1));
    if nnz(SPIKES{1:dim(1),p(i)})
        line_width = 2;
    else
        line_width = 1;
    end
    plot(X, TRACES{1:dim(1),p(i)}/max(TRACES{1:dim(1),p(i)}) + offset*(i-2), 'Color', sd_colornum_metro(i), 'LineWidth', line_width);
end
delete(w);

%% spikes drawing
w = waitbar(0, sprintf('Drawing spikes: trace %d of %d', 1,  dim(2)-1));

for i = 1:length(p)
    waitbar((i-1)/(dim(2)-1), w, sprintf('Processing cell %d of %d', i-1,  dim(2)-1));
    for j = 1:dim(1)
        if SPIKES {j,p(i)}
            sp_ampl = SPIKES {j,p(i)}/max(TRACES{1:dim(1),p(i)});
            patch([X(j)-0.3, X(j), X(j)+0.3, X(j)], [offset*(i-1.8) + sp_ampl, offset*(i-1.65) + sp_ampl, offset*(i-1.8) + sp_ampl, offset*(i-1.95) + sp_ampl], sd_colornum_metro(i), 'EdgeColor', 'none');
        end
    end
end

delete(w);

