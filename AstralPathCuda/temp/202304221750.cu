#include<stdio.h>
using namespace std;
//现在开始全局变量书写
     const int N = 100;//global
//现在开始全局内存书写
__global__ void hello(int *h,int *b) {
        int n = blockDim.x * blockIdx.x + threadIdx.x;
        printf("(%d,%d,%d)\n",blockIdx.x,blockIdx.y,blockIdx.z);
        b[n] = h[n];
}
int main(int argc, char* argv[]) {
        int M = sizeof(int)*N;
        int *h_x = (int*) malloc(M);
        int *h;
        
        for (int x = 0;x < N;x ++) {
            h_x[x] = 1;
        }
        int *d_x1;
        int *d_x2;
        int *d_x3;
        int *d_y1;
        int *d_y2;
        int *d_y3;
        cudaMalloc((void **)&d_x1,M);
        cudaMalloc((void **)&d_x2,M);
        cudaMalloc((void **)&d_x3,M);
        cudaMemcpy(d_x1,h_x,M,cudaMemcpyHostToDevice);
        cudaMemcpy(d_x2,h_x,M,cudaMemcpyHostToDevice);
        cudaMemcpy(d_x3,h_x,M,cudaMemcpyHostToDevice);
        cudaMalloc((void **)&d_y1,M);
        cudaMalloc((void **)&d_y2,M);
        cudaMalloc((void **)&d_y3,M);

        const int block_size = 128;
        const int grid_size =  (N + block_size - 1) / block_size;
        cudaStream_t stream_t1 ;
        cudaStream_t stream_t2 ;
        cudaStream_t stream_t3 ;
        cudaStreamCreate(&stream_t1);
        cudaStreamCreate(&stream_t2);
        cudaStreamCreate(&stream_t3);
        hello<<<grid_size,block_size,0,stream_t1>>>(d_x1,d_y1);//[tags:<<<grid_size,block_size,0,stream_t1>>>]
        hello<<<grid_size,block_size,0,stream_t2>>>(d_x2,d_y2);//[tags:<<<grid_size,block_size,0,stream_t2>>>]
        hello<<<grid_size,block_size,0,stream_t3>>>(d_x3,d_y3);//[tags:<<<grid_size,block_size,0,stream_t3>>>]
        cudaDeviceSynchronize();
        cudaStreamSynchronize(stream_t1);
        cudaStreamSynchronize(stream_t2);
        cudaStreamSynchronize(stream_t3);
        cudaMemcpy(h_x,d_y1,M,cudaMemcpyDeviceToHost);


        free(h_x);
        cudaFree(d_x1);
        cudaFree(d_x2);
        cudaFree(d_x3);
        cudaFree(d_y1);
        cudaFree(d_y2);
        cudaFree(d_y3);
        cudaStreamDestroy(stream_t1);
        cudaStreamDestroy(stream_t2);
        cudaStreamDestroy(stream_t3);
}
