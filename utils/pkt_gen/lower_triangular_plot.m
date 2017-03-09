idx = [];
Nrows = 40;
Ncols = 40;
for i = 1:Nrows
    idx = [idx, (((i-1)*Ncols) + 1):(((i-1)*Ncols) + i)];
end
figure();
plot(abs(kw_Rhat(idx)));