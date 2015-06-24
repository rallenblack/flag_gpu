% xgpu ordering check...
NSTATION = 32;
NPOL = 2;
NFREQUENCY = 1;

tmp = 1;
for i = 0:NSTATION-1
    for j = 0:i
        for pol1 = 0:NPOL-1
            for pol2 = 0:NPOL-1
                for f = 0:NFREQUENCY-1
                    k = f*(NSTATION+1)*(NSTATION/2) + i*(i+1)/2 + j;
                    idx(tmp) = (k*NPOL + pol1)*NPOL+pol2;
                    tmp = tmp + 1;
                end
            end
        end
    end
end
