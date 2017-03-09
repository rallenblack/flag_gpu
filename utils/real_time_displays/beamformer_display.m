% Script to display total power in pseduo real time
% clearvars;
% close all; % Comment these out when loading file in beamformer_disp_gui.m

data_dir = '/tmp/JUNK';
mcnt_step = 1600;
Ninstances = 1;
Nant = 40;
timeout = 1e4;
window_size = 100;

N_beam = 7;
N_pol = 4;
N_bin = 25;
start_freq = 1;
end_freq = 25;
N_time = 100;
N_beamformer_output = N_beam*N_pol*N_bin*N_time;

i = 0;
files = zeros(Ninstances, 1);
data = NaN(N_beamformer_output,1);
while 1
    for inst = 0:Ninstances-1
        filename = [data_dir,...
            '/beamformer_', num2str(inst), '_mcnt_', num2str(i), '.out'];
        
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
    
    tmp_data = zeros(N_beamformer_output, 1);
    for inst = 0:Ninstances-1
        if files(inst+1) ~= -1
            good_data = fread(files(inst+1), 1, 'int32');
            if (good_data ~= 1)
                fclose(files(inst+1));
                break;
            end
            tmp_data = tmp_data + fread(files(inst+1), N_beamformer_output, 'single');
            fclose(files(inst+1));
        end
    end
    data = [tmp_data, data(:,1:end-1)];
%     size(data)
    data = reshape(data, N_beam, N_pol, N_bin, N_time);
    
    power_acc_x = reshape(squeeze(data(:,1,:,:)), N_beam, N_bin, N_time);
    power_acc_x = permute(power_acc_x, [3, 2, 1]);
    power_acc_x = sum(power_acc_x, 1)./N_time;          % Integrate time samples for GUI.
    power_acc_x = reshape(power_acc_x, N_bin, N_beam1);
    
    power_acc_y = reshape(squeeze(data(:,2,:,:)), N_beam, N_bin, N_time);
    power_acc_y = permute(power_acc_y, [3, 2, 1]);
    power_acc_y = sum(power_acc_y, 1)./N_time;          % Integrate time samples for GUI.
    power_acc_y = reshape(power_acc_y, N_bin, N_beam1);
    
    power_acc_xyr = reshape(squeeze(data(:,3,:,:)), N_beam, N_bin, N_time);
    power_acc_xyr = permute(power_acc_xyr, [3, 2, 1]);
    power_acc_xyr = sum(power_acc_xyr, 1)./N_time;      % Integrate time samples for GUI.
    power_acc_xyr = reshape(power_acc_xyr, N_bin, N_beam1);
    
    power_acc_xyi = reshape(squeeze(data(:,4,:,:)), N_beam, N_bin, N_time);
    power_acc_xyi = permute(power_acc_xyi, [3, 2, 1]);
    power_acc_xyi = sum(power_acc_xyi, 1)./N_time;      % Integrate time samples for GUI.
    power_acc_xyi = reshape(power_acc_xyi, N_bin, N_beam1);
    
%     X - Polarization
    tmp1(t+1, start_freq:end_freq) = 10*log10(abs(power_acc_x(start_freq:end_freq,1)));
    axes(handles.axes1);
    plot(0:t, tmp1(1:t+1, start_freq:end_freq));
    set(gca, 'ydir', 'normal');
    xlabel('Time');
    ylabel('Power (dB)');
    title('X pol Beam 1');
    drawnow;
   
    tmp2(t+1, start_freq:end_freq) = 10*log10(abs(power_acc_x(start_freq:end_freq,2)));
    axes(handles.axes2);
    plot(0:t, tmp2(1:t+1, start_freq:end_freq));
    set(gca, 'ydir', 'normal');
    xlabel('Time');
    ylabel('Power (dB)');
    title('X pol Beam 2');
    drawnow;
   
    tmp3(t+1, start_freq:end_freq) = 10*log10(abs(power_acc_x(start_freq:end_freq,3)));
    axes(handles.axes3);
    plot(0:t, tmp3(1:t+1, start_freq:end_freq));
    set(gca, 'ydir', 'normal');
    xlabel('Time');
    ylabel('Power (dB)');
    title('X pol Beam 3');
    drawnow;
   
    tmp4(t+1, start_freq:end_freq) = 10*log10(abs(power_acc_x(start_freq:end_freq,4)));
    axes(handles.axes4);
    plot(0:t, tmp4(1:t+1, start_freq:end_freq));
    set(gca, 'ydir', 'normal');
    xlabel('Time');
    ylabel('Power (dB)');
    title('X pol Beam 4');
    drawnow;
   
    tmp5(t+1, start_freq:end_freq) = 10*log10(abs(power_acc_x(start_freq:end_freq,5)));
    axes(handles.axes5);
    plot(0:t, tmp5(1:t+1, start_freq:end_freq));
    set(gca, 'ydir', 'normal');
    xlabel('Time');
    ylabel('Power (dB)');
    title('X pol Beam 5');
    drawnow;
   
    tmp6(t+1, start_freq:end_freq) = 10*log10(abs(power_acc_x(start_freq:end_freq,6)));
    axes(handles.axes6);
    plot(0:t, tmp6(1:t+1, start_freq:end_freq));
    set(gca, 'ydir', 'normal');
    xlabel('Time');
    ylabel('Power (dB)');
    title('X pol Beam 6');
    drawnow;
   
    tmp7(t+1, start_freq:end_freq) = 10*log10(abs(power_acc_x(start_freq:end_freq,7)));
    axes(handles.axes7);
    plot(0:t, tmp7(1:t+1, start_freq:end_freq));
    set(gca, 'ydir', 'normal');
    xlabel('Time');
    ylabel('Power (dB)');
    title('X pol Beam 7');
    drawnow;
    %     figure(1);
    %     h = plot(data.');
    %     set(gca, 'NextPlot', 'replacechildren');
    %     title(['MCNT = ', num2str(i)]);
    %     drawnow;
    
    i = i + mcnt_step;
end

