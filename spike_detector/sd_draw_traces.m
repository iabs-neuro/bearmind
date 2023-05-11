function sd_draw_traces(path, fname, sp_fname)
%9 Jan 2019 Vova: Thickness increased for traces with spikes
%
%Reads traces from csv files and plot them on the same figure, coloring them
%using "colornum" routine; user can define offset between traces at the
%Y-axis (in %DF/F)
%
% Vladimir Sotskov, 2017-2020

TRACES = csvread(strcat(path, fname), 1);
SPIKES = csvread(strcat(path, sp_fname), 1);

dim = size(TRACES);
X = TRACES(1:dim(1),1);
maxim = max(max(TRACES(1:dim(1),2:dim(2))));
minim = min(min(TRACES(1:dim(1),2:dim(2))));
absmax = max(max(abs(maxim), abs(minim)));
offset = 1;

%% traces drawing
w = waitbar(0, sprintf('Plotting trace %d of %d', 1,  dim(2)-1));    
hold on
for i = 2:dim(2)
    waitbar((i-1)/(dim(2)-1), w, sprintf('Processing cell %d of %d', i-1,  dim(2)-1));
    if nnz(SPIKES(1:dim(1),i))
        line_width = 2;
    else
        line_width = 1;
    end
    plot(X, TRACES(1:dim(1),i)/max(TRACES(1:dim(1),i)) + offset*(i-2), 'Color', sd_colornum_metro(i-2), 'LineWidth', line_width);
end
delete(w);

%% spikes drawing
w = waitbar(0, sprintf('Drawing spikes: trace %d of %d', 1,  dim(2)-1));

for i = 2:dim(2)
    waitbar((i-1)/(dim(2)-1), w, sprintf('Processing cell %d of %d', i-1,  dim(2)-1));
    for j = 1:dim(1)
        if SPIKES (j,i)
            sp_ampl = SPIKES (j,i)/max(TRACES(1:dim(1),i));
            patch([X(j)-0.3, X(j), X(j)+0.3, X(j)], [offset*(i-1.8) + sp_ampl, offset*(i-1.65) + sp_ampl, offset*(i-1.8) + sp_ampl, offset*(i-1.95) + sp_ampl], sd_colornum_metro(i-2), 'EdgeColor', 'none');
        end
    end
end

delete(w);

