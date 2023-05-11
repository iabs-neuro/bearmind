function cn = sd_colornum_metro(num)
%Returns color vector [R, G, B], where R, G, B ranges from 0 to 1, which
%number in palette is 'num'
%
% Vladimir Sotskov, 2017-2020

    switch mod(num,10)
        case 0
            cn = [0.6, 1, 0.3];   %LIME
            return;
        case 1
            cn = [1, 0, 0]; %RED         
            return;
        case 2
            cn = [0, 0.6, 0]; %GREEN 
            return;            
        case 3
            cn = [0, 0, 1]; %BLUE 
            return;
        case 4
            cn = [0, 1, 1]; %CYAN 
            return;
        case 5
            cn = [0.5, 0.25, 0.2];  %BROWN 
            return;
        case 6
            cn = [1, 0.5, 0.1]; %ORANGE 
            return;
        case 7
            cn = [0.7, 0, 1];   %VIOLET 
            return;
        case 8
            cn = [0.8, 0.8, 0]; %YELLOW 
            return;
        case 9
            cn = [0.5, 0.5, 0.5];   %GRAY 
            return;
        otherwise
%-----------JUST IN CASE--------------
            return;
    end
end
