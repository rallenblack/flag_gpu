% Read correlations

FILE = fopen('floatArray.out', 'r');
[R, count] = fscanf(FILE, '%g\n');
fclose(FILE);

Nele = 40;
Nbin = 50;

Nbaselines = (Nele + 1)*Nele/2;
plot(abs(R));