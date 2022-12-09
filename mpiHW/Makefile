FCSerial=g++
FCParallel = mpiicpc # load module intel!
FFLAGS = -std=c++1z -Wall -Werror -pedantic -DNDEBUG -DBOOST_UBLAS_NDEBUG -O3


# g++ -o gol -O3 -DNDEBUG -DBOOST UBLAS NDEBUG -std=c++1z gol.cpp
gol_serial: gol_serial.cpp
	$(FCSerial) -o gol_serial $(FFLAGS) gol_serial.cpp

gol_parallel: gol_parallel.cpp
	$(FCParallel) -o gol_parallel $(FFLAGS) gol_parallel.cpp

clean:
	@ rm -f *.o *.mod $(r)