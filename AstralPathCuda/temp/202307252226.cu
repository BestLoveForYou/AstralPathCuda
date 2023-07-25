#include<stdio.h>
#include <cuda_runtime.h>
using namespace std;
//现在开始全局变量书写
     int N = 1;//global
//现在开始全局内存书写
__device__  int d_z[10] =  {};//length:10
__global__ void sum(int *d_x,int N) {
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
__shared__ int __shared__s_x[128];
        const int n = bid * blockDim.x + tid;
        int x = 0;
        if(n < N) {
            __shared__s_x[n] = d_x[n];
        }
        __syncthreads();
        for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
        {
            if (tid < offset)
            {
                __shared__s_x[tid] += __shared__s_x[tid + offset];
            }
            __syncthreads();
        }
        x = __shared__s_x[tid];
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            x += __shfl_down_sync(0xffffffff,x, offset);
        }


        if (tid == 0)
        {

            atomicAdd(&d_z[0], x);
        }
    }
    @Override
    __device__ void main() {
        N = argc - 1;
        int *a = (int*) malloc(sizeof(int)*N);
        int *b = (int *) malloc(sizeof(int)*10);
        for (int x = 0 ;x < N;x ++) {
            a[x] = atoi(argv[x + 1]);
        }
        int *d_x;
        cudaMalloc((void **)&d_x,sizeof(int)*N);
        cudaMemcpy(d_x,a,sizeof(int)*N,cudaMemcpyHostToDevice);
        const int block_size = 128;
        const int grid_size =  (N + block_size - 1) / block_size;
        cudaEvent_t start ;
        cudaEvent_t stop ;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        sum<<<grid_size,block_size>>>(d_x,N);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_time = 0;
        cudaEventElapsedTime(&elapsed_time,start,stop);
        printf("Time:%g ms\n",elapsed_time);

        cudaDeviceSynchronize();
        cudaMemcpyFromSymbol(b,d_z,sizeof(int)*10);
        printf("x:%d \n",b[0]);

        free(a);
        free(b);
        cudaFree(d_x);
}
int main(int argc, char* argv[]) {
}
