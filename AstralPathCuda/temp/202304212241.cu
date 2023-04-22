#include<stdio.h>
using namespace std;
//现在开始全局变量书写
     const int N = 10;//global
//现在开始全局内存书写
__device__  int d_x[N] =  {};//length:N
__device__  int d[2] =  {};//length:2
__global__ void reduce() {
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int n = bid * blockDim.x + tid;
__shared__ int __shared__s_y[128];
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
int ha[N] =  {};//length:N
        for(int x = 0; x < N ; ++x) {
            ha[x] = 1;
        }
        ha[0] = atoi(argv[1]);

        cudaMemcpyToSymbol(d_x,ha,sizeof(int)*N);

        const int block_size = 128;
        const int grid_size =  (N + block_size - 1) / block_size;

        reduce<<<grid_size,block_size>>>();


        int device_id = 0;

        cudaSetDevice(device_id);
        cudaDeviceProp prop ;
        cudaGetDeviceProperties(&prop,device_id);
        printf("Device id:                                 %d\n",
                device_id);
        printf("Device name:                               %s\n",
                prop.name);
        printf("Compute capability:                        %d.%d\n",
                prop.major, prop.minor);
        printf("Amount of global memory:                   %g GB\n",
                prop.totalGlobalMem / (1024.0 * 1024 * 1024));
        printf("Amount of constant memory:                 %g KB\n",
                prop.totalConstMem  / 1024.0);
        printf("Maximum grid size:                         %d %d %d\n",
                prop.maxGridSize[0],
                prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("Maximum block size:                        %d %d %d\n",
                prop.maxThreadsDim[0], prop.maxThreadsDim[1],
                prop.maxThreadsDim[2]);
        printf("Number of SMs:                             %d\n",
                prop.multiProcessorCount);
        printf("Maximum amount of shared memory per block: %g KB\n",
                prop.sharedMemPerBlock / 1024.0);
        printf("Maximum amount of shared memory per SM:    %g KB\n",
                prop.sharedMemPerMultiprocessor / 1024.0);
        printf("Maximum number of registers per block:     %d K\n",
                prop.regsPerBlock / 1024);
        printf("Maximum number of registers per SM:        %d K\n",
                prop.regsPerMultiprocessor / 1024);
        printf("Maximum number of threads per block:       %d\n",
                prop.maxThreadsPerBlock);
        printf("Maximum number of threads per SM:          %d\n",
                prop.maxThreadsPerMultiProcessor);

        cudaMemcpyFromSymbol(ha,d,sizeof(int)*2);
        printf("c=%d\n",ha[0]);
}
