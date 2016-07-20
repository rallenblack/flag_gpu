% Script to display total power in pseduo real time
clearvars;
close all;

data_dir = '/tmp/JUNK';
mcnt_step = 1600;
Ninstances = 1;
Nant = 40;
timeout = 1e4;
window_size = 100;

i = 0;
files = zeros(Ninstances, 1);
data = NaN(Nant, window_size);
while 1
    for inst = 0:Ninstances-1
        filename = [data_dir,...
            '/power_', num2str(inst), '_mcnt_', num2str(i), '.out'];
        
        t = 0;
        file_exists = exist(filename, 'file');
        while (~file_exists)
            file_exists = exist(filename, 'file');
            if t > timeout
                disp(['Waiting for ', filename]);
                t = 0;
            end
            t = t + 1;
        end
        [files(inst+1), MESSAGE] = fopen(filename, 'r');
    end
    
    tmp_data = zeros(Nant, 1);
    for inst = 0:Ninstances-1
        if files(inst+1) ~= -1
            good_data = fread(files(inst+1), 1, 'int32');
            if (good_data ~= 1)
                fclose(files(inst+1));
                tmp_data = NaN(Nant, 1);
                break;
            end
            tmp_data = tmp_data + fread(files(inst+1), Nant, 'single');
            fclose(files(inst+1));
        end
    end
    data = [tmp_data, data(:,1:end-1)];
    
    figure(1);
    h = plot(data.');
    set(gca, 'NextPlot', 'replacechildren');
    title(['MCNT = ', num2str(i)]);
    drawnow;
    
    i = i + mcnt_step;
end

