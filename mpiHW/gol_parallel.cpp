/**
 * Serial implementatin of the game of life for the course Parallel Computing.
 * Emil Loevbak (emil.loevbak@cs.kuleuven.be)
 * First implementation: November 2019
 * Updated to ublas datastructures in December 2022
 */

#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>
#include <iterator>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
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

void initializeBoard(ublas::matrix<bool> &board, int rank, int init = 0)
    // Initializes board to a pattern set by init. This routine can overwrite the ghost cells!
{
    if (init == 0) { // Random init
        int deadCellMultiplyer = 2;
        srand(time(0));
        for (auto row = board.begin1(); row != board.end1(); ++row)
        {
            for (auto element = row.begin(); element != row.end(); ++element)
            {
                *element = (rand() % (deadCellMultiplyer + 1) == 0);
            }
        }
    } else if (init == 1) { // Blinker oscillator in the middle of the board
        const size_t rows = board.size1();
        const size_t cols = board.size2();
        size_t middleRow = rows / 2;  
        size_t middleCol = cols / 2; 
        wipeBoard(board);
        board(middleRow, middleCol) = 1;
        board(middleRow-1, middleCol) = 1;
        board(middleRow+1, middleCol) = 1;
    } else if (init == 2){ // Blinker on all four boundaries 
        wipeBoard(board);
        if (rank == 0) {
            int localBoardSize = board.size1()-2;
            size_t middlePos = (localBoardSize+2) / 2;  
            board(middlePos, 1) = 1; // blinker on left boundary
            board(middlePos+1, 1) = 1;
            board(middlePos-1, 1) = 1;
            board(middlePos, localBoardSize) = 1; // blinker on right boundary
            board(middlePos+1, localBoardSize) = 1;
            board(middlePos-1, localBoardSize) = 1;
            board(1, middlePos) = 1; // blinker on top boundary
            board(1, middlePos+1) = 1;
            board(1, middlePos-1) = 1;
            board(localBoardSize, middlePos) = 1; // blinker on bottom boundary
            board(localBoardSize, middlePos+1) = 1;
            board(localBoardSize, middlePos-1) = 1;
        } 
    } else if (init == 3) { // Glider at the center of thread
        wipeBoard(board);
        const size_t rows = board.size1();
        const size_t cols = board.size2();
        size_t middleRow = rows / 2;  
        size_t col = cols-2; 
        board(middleRow, col) = 1;
        board(middleRow+1, col) = 1;
        board(middleRow-1, col) = 1;
        board(middleRow+1, col-1) = 1;
        board(middleRow, col-2) = 1;
    } else {
        assert(0==1);
    }
    
}

void updateBoard(ublas::matrix<bool> &board)
    // Same as serial implementation. Ghost cells are updated assuming periodic boundary conditions. This is wrong but they are not used after this anyway.
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

void writeGhosts(ublas::matrix<bool> &localBoard, bool leftGhosts[], bool rightGhosts[], bool topGhosts[], bool bottomGhosts[]) 
{
    int localBoardSize = localBoard.size1()-2;
    for (size_t i = 0; i < localBoardSize; ++i) { // write left ghost cells
        localBoard(i+1, 0) = leftGhosts[i];
    }
    for (size_t i = 0; i < localBoardSize; ++i) { // write right ghost cells
        localBoard(i+1, localBoardSize+1) = rightGhosts[i];
    }
    for (size_t i = 0; i < localBoardSize+2; ++i) { // write top ghost cells
        localBoard(0, i) = topGhosts[i];
    }
    for (size_t i = 0; i < localBoardSize+2; ++i) { // write bottom ghost cells
        localBoard(localBoardSize+1, i) = bottomGhosts[i];
    }
}

void getBorders(ublas::matrix<bool> &localBoard, bool leftBorder[], bool rightBorder[], bool topBorder[], bool bottomBorder[])
{
    int localBoardSize = localBoard.size1()-2;
    for (size_t i = 0; i < localBoardSize; ++i){ // copy left border to array
        leftBorder[i] = localBoard(i+1, 1);
    }
    for (size_t i = 0; i < localBoardSize; ++i){ // copy right border to array
        rightBorder[i] = localBoard(i+1, localBoardSize);
    }
    for (size_t i = 0; i < localBoardSize+2; ++i){ // copy top border to array. Include two ghostcells
        topBorder[i] = localBoard(1, i);
    }
    for (size_t i = 0; i < localBoardSize+2; ++i){ // copy bottom border to array. Include two ghostcells
        bottomBorder[i] = localBoard(localBoardSize, i);
    }
}

void printArray(bool ar[], int arSize, int rank){
    for (size_t i = 0; i < arSize; ++i){
        std::cout << "Rank: " << rank << " i: " << i << " el: " << ar[i] << std::endl;
    }
}


void communicateGhostCells(ublas::matrix<bool> &localBoard, int &rank, int &prow, int &pcol, int const p)
{
    // Initialize arrays
    MPI_Status status;
    int localBoardSize = localBoard.size1()-2;
    bool leftBorder[localBoardSize], rightBorder[localBoardSize], topBorder[localBoardSize+2], bottomBorder[localBoardSize+2];
    bool leftGhosts[localBoardSize], rightGhosts[localBoardSize], topGhosts[localBoardSize+2], bottomGhosts[localBoardSize+2];
    getBorders(localBoard, leftBorder, rightBorder, topBorder, bottomBorder);

    // Get ranks for neighbours
    int temp1 = (p + (pcol+1 % p)) % p, rightNeighbour = sub2ind(p, prow, temp1);
    int temp2 = (p + (pcol-1 % p)) % p, leftNeighbour = sub2ind(p, prow, temp2);
    temp1 = (p + (prow-1 % p)) % p; int topNeighbour = sub2ind(p, temp1, pcol);
    temp2 = (p + (prow+1 % p)) % p; int bottomNeighbour = sub2ind(p, temp2, pcol);

    // std::cout << p << std::endl;
    // std::cout << -1 % 2 << std::endl;
    // std::cout << "Proc: " << rank << " Left neighbour col: " << temp2 << " Right neighbour col: " << temp1 << std::endl;    
    // std::cout << "Proc: " << rank << " Left neighbour: " << leftNeighbour << " Right neighbour: " << rightNeighbour << " Top neighbour: " << topNeighbour << " Bottom neighbour: " << bottomNeighbour << std::endl;

    if (pcol % 2 == 0) {
        // Send & receive with right neighbour
        MPI_Sendrecv(rightBorder, localBoardSize, MPI_CXX_BOOL, rightNeighbour, 0, rightGhosts, localBoardSize, MPI_CXX_BOOL, rightNeighbour, 1, MPI_COMM_WORLD, &status);
        // Send & receive with left neighbour
        MPI_Sendrecv(leftBorder, localBoardSize, MPI_CXX_BOOL, leftNeighbour, 2, leftGhosts, localBoardSize, MPI_CXX_BOOL, leftNeighbour, 3, MPI_COMM_WORLD, &status);
    } else {
        // Send & receive with left neighbour
        MPI_Sendrecv(leftBorder, localBoardSize, MPI_CXX_BOOL, leftNeighbour, 1, leftGhosts, localBoardSize, MPI_CXX_BOOL, leftNeighbour, 0, MPI_COMM_WORLD, &status);
        // Send & receive with right neighbour
        MPI_Sendrecv(rightBorder, localBoardSize, MPI_CXX_BOOL, rightNeighbour, 3, rightGhosts, localBoardSize, MPI_CXX_BOOL, rightNeighbour, 2, MPI_COMM_WORLD, &status);
    }

    // Include diagonal elements in top border and bottom border
    bottomBorder[0] = leftBorder[localBoardSize-1];
    bottomBorder[localBoardSize+1] = rightBorder[localBoardSize-1];
    topBorder[0] = leftBorder[0];
    topBorder[localBoardSize+1] = rightBorder[0];

    if (prow % 2) {
        // Send & receive with top neighbour
        MPI_Sendrecv(topBorder, localBoardSize+2, MPI_CXX_BOOL, topNeighbour, 4, topGhosts, localBoardSize+2, MPI_CXX_BOOL, topNeighbour, 5, MPI_COMM_WORLD, &status);
        // Send & receive with with bottom neighbour
        MPI_Sendrecv(bottomBorder, localBoardSize+2, MPI_CXX_BOOL, bottomNeighbour, 6, bottomGhosts, localBoardSize+2, MPI_CXX_BOOL, bottomNeighbour, 7, MPI_COMM_WORLD, &status);
    } else {
        // Send & receive with bottom neighbour
        MPI_Sendrecv(bottomBorder, localBoardSize+2, MPI_CXX_BOOL, bottomNeighbour, 5, bottomGhosts, localBoardSize+2, MPI_CXX_BOOL, bottomNeighbour, 4, MPI_COMM_WORLD, &status);
        // Send & receive with top neighbour
        MPI_Sendrecv(topBorder, localBoardSize+2, MPI_CXX_BOOL, topNeighbour, 7, topGhosts, localBoardSize+2, MPI_CXX_BOOL, topNeighbour, 6, MPI_COMM_WORLD, &status);
    }

    writeGhosts(localBoard, leftGhosts, rightGhosts, topGhosts, bottomGhosts);
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
    int localBoardSize = cols/p;
    size_t firstRow = prow*localBoardSize, lastRow = (prow+1)*localBoardSize - 1;
    size_t firstCol = pcol*localBoardSize, lastCol = (pcol+1)*localBoardSize - 1;
    
    ublas::matrix<bool> localBoard(localBoardSize+2, localBoardSize+2);
    int init = 0;
    initializeBoard(localBoard, rank, init);

    // Do iteration
    writeBoardToFile(localBoard, firstRow, lastRow, firstCol, lastCol, programName, 0, rank);

    for (int i = 1; i <= iterations; ++i)
    {   
        // Communicate ghost cells here
        communicateGhostCells(localBoard, rank, prow, pcol, p);
        updateBoard(localBoard);
        if (i % iteration_gap == 0)
        {
            writeBoardToFile(localBoard, firstRow, lastRow, firstCol, lastCol, programName, i, rank);
        }
    }

    // std::cout << rank << " aborting" << std::endl;
    // MPI_Barrier(MPI_COMM_WORLD);
    // MPI_Abort(MPI_COMM_WORLD, 1); // always abort after this for testing


    MPI_Finalize();
}
