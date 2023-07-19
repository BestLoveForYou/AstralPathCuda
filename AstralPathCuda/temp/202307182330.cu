#include<stdio.h>
#include <cuda_runtime.h>
using namespace std;
//现在开始全局变量书写
     const int N = 100;//global
//现在开始全局内存书写
__device__ void add(int *a,int *b,int *z) {
        z[threadIdx.x] = a[threadIdx.x]+ b[threadIdx.x];
}
__global__ void hello(int *h,int *b,int *z) {
        int n = threadIdx.x;
        add(h,b,z);
        printf("(%d,%d) = %d \n",threadIdx.x,threadIdx.y,z[n]);
        b[n] = h[n];

}
__global__ void hello2(int *h,int *b,int *z) {
        int n = threadIdx.x;
        add(h,b,z);
        printf("(%d,%d) = %d \n",threadIdx.x,threadIdx.y,z[n]);
        b[n] = h[n];

}
int main(int argc, char* argv[]) {
        int M = sizeof(int)*N;
        int *h_x = (int *) malloc(M);
        for (int x = 0;x < N;x ++) {
            h_x[x] = 1;
        }
        int *d_x1;
        int *d_y1;
        int *d_y2;

        cudaMalloc((void **)&d_x1,M);
        cudaMemcpy(d_x1,h_x,M,cudaMemcpyHostToDevice);
        cudaMalloc((void **)&d_y1,M);
        cudaMemcpy(d_y1,h_x,M,cudaMemcpyHostToDevice);
        cudaMalloc((void **)&d_y2,M);

        const int block_size = 128;
        const int grid_size =  (N + block_size - 1) / block_size;

        hello<<<grid_size,block_size>>>(d_x1,d_y1,d_y2);

        cudaDeviceSynchronize();

        cudaMemcpy(h_x,d_y1,M,cudaMemcpyDeviceToHost);


        free(h_x);
        cudaFree(d_x1);

        cudaFree(d_y1);

}
