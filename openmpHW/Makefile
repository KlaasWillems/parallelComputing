FC=g++
FFLAGS = -std=c++17 -DNDEBUG -DBOOST_UBLAS_NDEBUG -O3 -fopenmp # For performance
# FFLAGS = -std=c++17 -Wall -Werror -pedantic -fopenmp # For debugging 

cpp_main.o: cpp_main.cpp
	$(FC) $(FFLAGS) -c cpp_main.cpp

cpp_main_serial: cpp_main.o cpp_matrixops_serial.o
	$(FC) -o cpp_main_serial cpp_main.o cpp_matrixops_serial.o -fopenmp

cpp_main_parallel: cpp_main.o cpp_matrixops_parallel.o
	$(FC) -o cpp_main_parallel cpp_main.o cpp_matrixops_parallel.o -fopenmp

cpp_matrixops_parallel.o: cpp_matrixops_parallel.cpp
	$(FC) $(FFLAGS) -c cpp_matrixops_parallel.cpp 

cpp_matrixops_serial.o: cpp_matrixops_serial.cpp
	$(FC) $(FFLAGS) -c cpp_matrixops_serial.cpp

clean:
	@ rm -f *.o *.mod $(r)