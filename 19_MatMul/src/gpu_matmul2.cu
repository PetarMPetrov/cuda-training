#include <cuda_runtime_api.h>
#include <iostream>
#include <cstdlib>
#include <chrono>

int A[3*4] = {1,1,1,1, 1,1,1,1, 2,2,2,2};

int B[4*2] = {5, 10, 5, 10, 5, 10, 5, 10};

int C[3*2]; // = { 0, 0, 0 ,0, 0, 0};


//template<std::size_t NROWS, std::size_t NCOLS, std::size_t VLEN>
__global__ void matMulGPU(int * mA, int * mB, int * mC, int NROWS, int NCOLS, int VLEN)
{
    int op_id = blockIdx.x * blockDim.x + threadIdx.x;
    int row = op_id / NCOLS;
    int col = op_id % NCOLS;
    if (op_id < NCOLS * NROWS)
    {
        for(int k = 0; k < VLEN; k++)
        {
            //mC[row][col] += mA[row][k] * mB[k][col];
            int idxA = row * VLEN + k;
            int idxB = k * NCOLS + col;
            mC[op_id] += mA[idxA] * mB[idxB];
        }
    }
    printf("index = %d; mC[index] = %d", op_id, mC[op_id]);
}

int main(int argc, char *argv[])
{
    unsigned int _nrows, _ncols, _vlen;
    if (argc == 1)
    {
        _nrows = 3000;
        _ncols = 200;
        _vlen  = 4;
    }
    else if (argc == 2)
    {
        _nrows = 3 * atoi(argv[1]);
        _ncols = 2 * atoi(argv[1]);
        _vlen  = 4 * atoi(argv[1]);
    }
    else
    {
        _nrows = atoi(argv[1]);
        _ncols = atoi(argv[2]);
        _vlen  = atoi(argv[3]);
    }

    const unsigned int nrows = _nrows;
    const unsigned int ncols = _ncols;
    const unsigned int vlen = _vlen;

    int _A[nrows][vlen];
    int _B[vlen][ncols];
    int _C[nrows][ncols];

    // Initialize _A
    for (int i = 0; i < nrows; i++)
    {
        for(int j = 0; j < vlen; j++)
        {
            _A[i][j] = std::rand()%10;
        }
    }
    // Initialize _B
    for (int i = 0; i < vlen; i++)
    {
        for(int j = 0; j < ncols; j++)
        {
            _B[i][j] = std::rand()%10;
        }
    }
    // Initialize _C
//    for (int i = 0; i < nrows; i++)
//    {
//        for(int j = 0; j < ncols; j++)
//        {
//            _C[i][j] = 0;
//        }
//    }

    // Matrix Multiplication
    cudaEvent_t start, malloc_done, ABcopy_done, matmul_done, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&malloc_done);
    cudaEventCreate(&ABcopy_done);
    cudaEventCreate(&matmul_done);
    cudaEventCreate(&stop);
    // Make sure nrows * ncols is exactly divisible by nBlocks
    const std::size_t nBlocks = 1;
    //int mA[nrows][vlen], mB[vlen][ncols], mC[nrows][ncols];
    int * mA;
    int * mB;
    int * mC;
    cudaEventRecord(start);
    std::size_t sizeA = sizeof(int) * nrows * vlen;
    std::size_t sizeB = sizeof(int) * ncols * vlen;
    std::size_t sizeC = sizeof(int) * nrows * ncols;
    cudaMalloc((void**)&mA, sizeA);
    cudaMalloc((void**)&mB, sizeB);
    cudaMalloc((void**)&mC, sizeC);
    cudaEventRecord(malloc_done);
    cudaMemcpy(mA, _A, sizeA, cudaMemcpyHostToDevice); 
    cudaMemcpy(mB, _B, sizeB, cudaMemcpyHostToDevice); 
    cudaEventRecord(ABcopy_done);
    //matMulGPU<nrows,ncols,vlen><<<nBlocks, nrows * ncols / nBlocks>>>(mA, mB, mC);
    matMulGPU<<<nBlocks, nrows * ncols / nBlocks>>>(mA, mB, mC, _nrows, _ncols, _vlen);
    cudaEventRecord(matmul_done);
    cudaMemcpy(_C, mC, sizeC, cudaMemcpyDeviceToHost); 
    cudaEventRecord(stop);
    cudaFree(mA);
    cudaFree(mB);
    cudaFree(mC);
    
    cudaEventSynchronize(stop);
    float malloc_time = 0, ABcopy_time = 0, matmul_time = 0, Ccopy_time = 0, total_time = 0;
    cudaEventElapsedTime(&malloc_time, start, malloc_done);
    cudaEventElapsedTime(&ABcopy_time, malloc_done, ABcopy_done);
    cudaEventElapsedTime(&matmul_time, ABcopy_done, matmul_done);
    cudaEventElapsedTime(&Ccopy_time, matmul_done, stop);
    cudaEventElapsedTime(&total_time, start, stop);
    std::cout << "Total MatMul (ms): " << total_time << std::endl;
    std::cout << "Malloc (ms): " << malloc_time << std::endl;
    std::cout << "AB MemCpy (ms): " << ABcopy_time << std::endl;
    std::cout << "MatMul (ms): " << matmul_time << std::endl;
    std::cout << "C MemCpy (ms): " << Ccopy_time << std::endl;
    //matMulCPU(_A, _B, _C, nrows, ncols, vlen);
/*    for(int i = 0; i < nrows; i++)
    {
        for(int j = 0; j < ncols; j++)
        {
            for(int k = 0; k < vlen; k++)
            {
                _C[i][j] += _A[i][k] * _B[k][j];
            }
        }
    }
*/
//    for(int i = 0; i < 6; i++){
//        const int row = i/2;
//        const int col = i%2;
//        std::cout << "C[" << row << "][" << col << "] = " << C[i] <<std::endl;
//    }

    // Print small Matrices
    if(nrows * ncols < 20 && false)
    {
        for(int i = 0; i < nrows * vlen; i++){
            const int row = i/vlen;
            const int col = i%vlen;
            std::cout << "A[" << row << "][" << col << "] = " << _A[row][col] <<std::endl;
        }
        for(int i = 0; i < vlen * ncols; i++){
            const int row = i/ncols;
            const int col = i%ncols;
            std::cout << "B[" << row << "][" << col << "] = " << _B[row][col] <<std::endl;
        }
        for(int i = 0; i < nrows * ncols; i++){
            const int row = i/ncols;
            const int col = i%ncols;
            std::cout << "C[" << row << "][" << col << "] = " << _C[row][col] <<std::endl;
        }
    }
    return 0;
}
