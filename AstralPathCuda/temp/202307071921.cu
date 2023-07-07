#include<stdio.h>
using namespace std;
//现在开始全局变量书写
     const int N = 5;//global
//现在开始全局内存书写
__device__  double d_z[10] =  {};//length:10
__global__ void sum(double *d_x,double *d_y) {
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
__shared__ double __shared__s_x[128];
__shared__ double __shared__s_x2[128];
__shared__ double __shared__s_y[128];
__shared__ double __shared__s_xy[128];
        const int n = bid * blockDim.x + tid;
        double x = 0.0;
        double y = 0.0;
        double x2 = 0.0;
        double xy = 0.0;
        __shared__s_x[tid] = (n < N) ? d_x[n] : 0.0;
        __shared__s_x2[tid] = (n < N) ? (__shared__s_x[tid] * __shared__s_x[tid]) : 0.0;
        __shared__s_y[tid] = (n < N) ? d_y[n] : 0.0;

        __shared__s_xy[tid] = (n < N) ? __shared__s_x[tid] * __shared__s_y[tid] : 0.0;
        __syncthreads();
        for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
        {
            if (tid < offset)
            {
                __shared__s_x[tid] += __shared__s_x[tid + offset];
                __shared__s_y[tid] += __shared__s_y[tid + offset];
                __shared__s_x2[tid] += __shared__s_x2[tid + offset];
                __shared__s_xy[tid] += __shared__s_xy[tid + offset];
            }
            __syncthreads();
        }

        y = __shared__s_y[tid];
        x = __shared__s_x[tid];
        x2 = __shared__s_x2[tid];
        xy = __shared__s_xy[tid];
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            y += __shfl_down_sync(0xffffffff,y, offset);
            x += __shfl_down_sync(0xffffffff,x, offset);
            x2 += __shfl_down_sync(0xffffffff,x2, offset);
            xy += __shfl_down_sync(0xffffffff,xy, offset);
        }

        if (tid == 0)
        {
            atomicAdd(&d_z[0], x);
            atomicAdd(&d_z[1], y);
            atomicAdd(&d_z[2], x2);
            atomicAdd(&d_z[3], xy);

        }

        if(n == 0) {
            double fenzi = d_z[3] - (d_z[0] * d_z[1] / N);
            double fenmu = d_z[2] - (d_z[0] * d_z[0] / N);
            d_z[4] = fenzi / fenmu;
        }
    }
int main(int argc, char* argv[]) {
        double *a = (double *) malloc(sizeof(double)*N);
        double *a2 = (double *) malloc(sizeof(double)*N);
        double *b = (double *) malloc(sizeof(double)*10);
        for (int x = 0 ;x < N;x ++) {
            a[x] = 11 + 1.0 * x;
            a2[x] = 11 + 3.0124 * x;
        }
        double *d_x;
        double *d_y;
        cudaMalloc((void **)&d_x,sizeof(double)*N);
        cudaMalloc((void **)&d_y,sizeof(double)*N);
        cudaMemcpy(d_x,a,sizeof(double)*N,cudaMemcpyHostToDevice);
        cudaMemcpy(d_y,a2,sizeof(double)*N,cudaMemcpyHostToDevice);
        const int block_size = 128;
        const int grid_size =  (N + block_size - 1) / block_size;
        sum<<<grid_size,block_size>>>(d_x,d_y);
        cudaDeviceSynchronize();
        cudaMemcpyFromSymbol(b,d_z,sizeof(double)*10);
        printf("%f \n",b[0]);
        printf("%f \n",b[1]);
        printf("%f \n",b[2]);
        printf("%f \n",b[3]);
        printf("%f \n",b[4]);
}
