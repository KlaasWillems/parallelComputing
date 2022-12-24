/**
 * Serial implementatin of the game of life for the course Parallel Computing.
 * Emil Loevbak (emil.loevbak@cs.kuleuven.be)
 * First implementation: November 2019
 * Updated to ublas datastructures in December 2022
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <boost/numeric/ublas/matrix.hpp>
#include <ctime>
#include <tuple>
#include <mpi.h>


namespace ublas = boost::numeric::ublas;

int const globalBufferLength = 50;

int sub2ind(int p, int row, int col) 
{
    // Convert subscripts to linear indices. 
    int li = p*row + col;
    return li;
}

std::tuple<int, int> ind2sub(int p, int li)
{
    // Convert linear indices to row and column tuple. p^2 grid of processors.
    int c = li % p;
    int r = li / p;
    std::tuple<int, int> sub(r, c);
    return sub;
}

void wipeBoard(ublas::matrix<bool> &board)
{
    for (auto row = board.begin1(); row != board.end1(); ++row)
        {
            for (auto element = row.begin(); element != row.end(); ++element)
            {
                *element = 0;
            }
        }
}

void initializeBoard(ublas::matrix<bool> &board, int init = 0)
    // Initializes board to a pattern set by init. This routine can overwrite the ghost cells!
{
    if (init == 0) {
        int deadCellMultiplyer = 2;
        srand(time(0));
        for (auto row = board.begin1(); row != board.end1(); ++row)
        {
            for (auto element = row.begin(); element != row.end(); ++element)
            {
                *element = (rand() % (deadCellMultiplyer + 1) == 0);
            }
        }
    } else if (init == 1) { // Blinker oscillator
        const size_t rows = board.size1();
        const size_t cols = board.size2();
        size_t middleRow = rows / 2;  
        size_t middleCol = cols / 2; 
        wipeBoard(board);
        board(middleRow, middleCol) = 1;
        board(middleRow-1, middleCol) = 1;
        board(middleRow+1, middleCol) = 1;
    } else {
        assert(0==1);
    }
    
}

void updateBoard(ublas::matrix<bool> &board)
{
    const size_t rows = board.size1();
    const size_t cols = board.size2();
    ublas::matrix<int> liveNeighbors(rows, cols, 0);

    //Count live neigbors
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            if (board(i, j))
            {
                for (int di = -1; di <= 1; ++di)
                {
                    for (int dj = -1; dj <= 1; ++dj)
                    {
                        //Periodic boundary conditions
                        liveNeighbors((i + di + rows) % rows, (j + dj + cols) % cols)++;
                    }
                }
                liveNeighbors(i, j)--; //Correction so that a cell does not concider itself as a live neighbor
            }
        }
    }

    //Update board
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            board(i, j) = ((liveNeighbors(i, j) == 3) || (board(i, j) && liveNeighbors(i, j) == 2));
        }
    }
}

void writeBoardToFile(ublas::matrix<bool> &board, size_t firstRow, size_t lastRow, size_t firstCol, size_t lastCol, std::string fileName, int iteration, unsigned int processID)
{
    // Open file
    std::ofstream outputFile("data/" + fileName + "_" + std::to_string(iteration) + "_" + std::to_string(processID) + ".gol");
    // Write metadata
    outputFile << std::to_string(firstRow) << " " << std::to_string(lastRow) << std::endl;
    outputFile << std::to_string(firstCol) << " " << std::to_string(lastCol) << std::endl;
    // Write data
    std::ostream_iterator<bool> outputIterator(outputFile, "\t");
    for (auto row = board.begin1()+1; row != board.end1()-1; ++row)
    {
        copy(row.begin()+1, row.end()-1, outputIterator);
        outputFile << std::endl;
    }

    // for (size_t row = 1; row < board.size1()-1; ++row) {
    //     for (size_t col = 1; col < board.size2()-1; ++col) {
    //         outputFile << "\t" << board(row, col);
    //     }
    //     outputFile << std::endl;
    // }
    outputFile.close();
}

std::string setUpProgram(size_t rows, size_t cols, int iteration_gap, int iterations, int processes)
{
    //Generate progam name based on current time, all threads should use the same name!
    time_t rawtime;
    struct tm *timeInfo;
    char buffer[globalBufferLength];
    time(&rawtime);
    timeInfo = localtime(&rawtime);
    strftime(buffer, sizeof(buffer), "%Y-%m-%d-%H-%M-%S", timeInfo);
    std::string programName(buffer);

    //Generate main file
    std::ofstream outputFile("data/" + programName + ".gol");
    outputFile << std::to_string(rows) << " " << std::to_string(cols) << " " << std::to_string(iteration_gap) << " " << std::to_string(iterations) << " " << std::to_string(processes) << std::endl;
    outputFile.close();

    return programName;
}

int main(int argc, char *argv[])
{
    // MPI initialization
    int rank; int num_proc;
    int const root = 0; // main proc
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Check MPI arguments
    long long sr = sqrt(num_proc);
    if (sr * sr != num_proc || num_proc % 2 != 0){
        std::cout << "Amount of processors should be a perfect square and even." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int const p = sqrt(num_proc);

    // Check GOL arguments
    if (argc != 5)
    {
        std::cout << "This program should be called with four arguments! \nThese should be, the total number of rows; the total number of columns; the gap between saved iterations and the total number of iterations, in that order." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t rows, cols;
    int iteration_gap, iterations;
    try
    {
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
        iteration_gap = atoi(argv[3]);
        iterations = atoi(argv[4]);
    }
    catch (std::exception const &exc)
    {
        std::cout << "One or more program arguments are invalid!" << std::endl;
        return 1;
    }

    if (rows != cols || rows % num_proc != 0) {
        std::cout << "Domain should be square. Domain size should be divisible by num_proc" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // setUpProgram -> Broadcast program to all threads
    std::string programName; 
    if (rank == root){
        programName = setUpProgram(rows, cols, iteration_gap, iterations, num_proc);
    }
    int programNameSize = programName.size();
    MPI_Bcast(&programNameSize, 1, MPI_INT, root, MPI_COMM_WORLD); // Broadcast size of string
    if (rank != root){programName.resize(programNameSize);}
    MPI_Bcast(const_cast<char*>(programName.data()), programNameSize, MPI_CHAR, root, MPI_COMM_WORLD);

    // Grid of processors
    std::tuple<int, int> coord = ind2sub(p, rank);
    int pcol = std::get<1>(coord);
    int prow = std::get<0>(coord);

    // Build board
    int localBoardSize = cols/num_proc;
    size_t firstRow = prow*localBoardSize, lastRow = (prow+1)*localBoardSize - 1;
    size_t firstCol = pcol*localBoardSize, lastCol = (pcol+1)*localBoardSize - 1;
    size_t ghostDepth = 1; // Ghost cells depth on either side of localBoard
    
    ublas::matrix<bool> localBoard(localBoardSize+ghostDepth*2, localBoardSize+ghostDepth*2);
    int init = 1;
    initializeBoard(localBoard, init);

    // Do iteration
    writeBoardToFile(localBoard, firstRow, lastRow, firstCol, lastCol, programName, 0, rank);

    std::cout << rank << " aborting" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 1); // always abort after this for testing

    for (int i = 1; i <= iterations; ++i)
    {   
        // Communicate ghost cells here
        updateBoard(localBoard);
        if (i % iteration_gap == 0)
        {
            writeBoardToFile(localBoard, firstRow, lastRow, firstCol, lastCol, programName, i, rank);
        }
    }

    MPI_Finalize();
}
