#include<stdio.h>
using namespace std;
//现在开始全局变量书写
     const int N = 128;//global
//现在开始全局内存书写
__device__  int d[128] =  {};//length:128
__device__  int e[128] =  {};//length:128
__global__ void hello() {
        const int x = threadIdx.x;
        const int y = threadIdx.y;
        int n = blockDim.x * blockIdx.x + threadIdx.x;
        int a = d[n];
        int b = e[n];
        d[n] = a*b;
}
int main(int argc, char* argv[]) {
int ha[128] =  {};//length:128
        for(int x = 0; x < 128 ; ++x) {
            ha[x] = 5;
        }
        cudaMemcpyToSymbol(d,ha,sizeof(int)*2);
        for(int x = 0; x < 128 ; ++x) {
            ha[x] = 7;
        }
        cudaMemcpyToSymbol(e,ha,sizeof(int)*2);
        const int block_size = 128;
        const int grid_size = N / block_size;

        hello<<<grid_size,block_size>>>();
        cudaMemcpyFromSymbol(ha,d,sizeof(int)*2);
        printf("c=%d\n",ha[0]);
}
