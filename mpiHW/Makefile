# VSC commands:
# 	module load intel
# 	module load Boost/1.70.0-intel-2018a

# To run on VSC:
# 	mpirun -np 4 ./gol_parallel

FCSerial = icpc
FCParallel = mpiicpc 
FFLAGS = -Wall -Werror -pedantic -std=c++17
# FFLAGS = -std=c++17 -DNDEBUG -DBOOST_UBLAS_NDEBUG -O3

gol_serial: gol_serial.o
	$(FCSerial) -o gol_serial gol_serial.o 

gol_serial.o: gol_serial.cpp
	$(FCSerial) $(FFLAGS) -c gol_serial.cpp

gol_parallel: gol_parallel.cpp
	$(FCParallel) -o gol_parallel $(FFLAGS) gol_parallel.cpp

clean:
	@ rm -f *.o *.mod $(r)