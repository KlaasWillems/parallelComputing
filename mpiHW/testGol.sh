rm data/*
make gol_parallel
mpirun -np 4 ./gol_parallel 40 40 1 80