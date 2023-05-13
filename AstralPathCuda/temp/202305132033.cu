#include<stdio.h>
using namespace std;
//现在开始全局变量书写
     const int N = 1000;//global
//现在开始全局内存书写
__global__ void test(float *h,float *b,int x) {
        printf("Stream: %d - (%f,%f)\n",x,threadIdx.x,threadIdx.y);
        b[threadIdx.x] = sqrt(h[threadIdx.x]);
    }
int main(int argc, char* argv[]) {
        cudaStream_t stream_1 ;
        cudaStream_t stream_2 ;

        cudaStreamCreate(&stream_1);
        cudaStreamCreate(&stream_2);

        int M = sizeof(float)*N;
        float *h_x = (float*) malloc(M);
        for (int x = 0;x < N;x ++) {
            h_x[x] = (float) ((x + 1) * 999.0123);
        }
        float *d_x1;
        float *d_y1;
        cudaMalloc((void **)&d_x1,M);
        cudaMemcpy(d_x1,h_x,M,cudaMemcpyHostToDevice);
        cudaMalloc((void **)&d_y1,M);

        const int block_size = 128;
        const int grid_size =  (N + block_size - 1) / block_size;

        test<<<grid_size,block_size,0,stream_1>>>(d_x1,d_y1,0);//tags:<<<grid_size,block_size,0,stream_1>>>
        test<<<grid_size,block_size,0,stream_2>>>(d_x1,d_y1,1);//tags:<<<grid_size,block_size,0,stream_2>>>

        cudaStreamSynchronize(stream_1);
        cudaStreamSynchronize(stream_2);

        cudaStreamDestroy(stream_1);
        cudaStreamDestroy(stream_2);
}
