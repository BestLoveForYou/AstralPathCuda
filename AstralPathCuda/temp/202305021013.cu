#include<stdio.h>
using namespace std;
//现在开始全局变量书写
     const int N = 1000;//global
//现在开始全局内存书写
__device__  int d[2] =  {};//length:2
__device__  int d_x[N] =  {};//length:N
__global__ void reduce() {
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int n = tid;
extern __shared__ int __shared__s_y[];
        if (n < N) {
            __shared__s_y[tid] = __ldg(&d_x[n]);
        }
        __syncthreads();

        for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
        {
            if (tid < offset)
            {
                __shared__s_y[tid] += __shared__s_y[tid + offset];
            }
            __syncthreads();
        }

        int y = __shared__s_y[tid];
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            y += __shfl_down_sync(0xffffffff, y, offset);
        }

        if (tid == 0)
        {
            atomicAdd(&d[0], y);
        }
    }
int main(int argc, char* argv[]) {
        int ha[N] = {};
        for(int x = 0; x < N ; ++x) {
            ha[x] = 1;
        }
        ha[0] = atoi(argv[1]);

        cudaMemcpyToSymbol(d_x,ha,sizeof(int)*N);

        const int block_size = 128;
        const int grid_size =  (N + block_size - 1) / block_size;

        reduce<<<block_size,grid_size,128>>>();//tags:<<<block_size,grid_size,128>>>


        int device_id = 0;
        cudaMemcpyFromSymbol(ha,d,sizeof(int)*2);
        printf("c=%d\n",ha[0]);
}
