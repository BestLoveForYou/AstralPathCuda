#include<stdio.h>
using namespace std;
//现在开始全局变量书写
     const int N = 10000;//global
//现在开始全局内存书写
__device__  double y[N] =  {};//length:N
__device__  double y2[N] =  {};//length:N
__device__  double xy[N] =  {};//length:N
__device__  double x2[N] =  {};//length:N
__device__  double x[N] =  {};//length:N
__device__  double temp[10] =  {};//length:10
__global__ void cor(int limit) {
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int n = tid;
extern __shared__ double __shared__x[];
extern __shared__ double __shared__x2[];
extern __shared__ double __shared__y[];
extern __shared__ double __shared__y2[];
extern __shared__ double __shared__xy[];
        if (n < limit) {
            __shared__x[tid] = __ldg(&x[n]);
            __shared__y[tid] = __ldg(&y[n]);
            __shared__x2[tid] = __shared__x[tid] * __shared__x[tid];
            __shared__y2[tid] = __shared__y[tid] * __shared__y[tid];
            __shared__xy[tid] = __shared__x[tid] * __shared__y[tid];
        }
        __syncthreads();

        for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
        {
            if (tid < offset)
            {
                __shared__x[tid] += __shared__x[tid + offset];
                __shared__x2[tid] += __shared__x2[tid + offset];
                __shared__y[tid] += __shared__y[tid + offset];
                __shared__y2[tid] += __shared__y2[tid + offset];
                __shared__xy[tid] += __shared__xy[tid + offset];
            }
            __syncthreads();
        }
        double dx = __shared__x[tid];
        double dx2 = __shared__x2[tid];
        double dy = __shared__y[tid];
        double dy2 = __shared__y2[tid];
        double dxy = __shared__xy[tid];
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            dx += __shfl_down_sync(0xffffffff, dx, offset);
            dx2 += __shfl_down_sync(0xffffffff, dx2, offset);
            dy += __shfl_down_sync(0xffffffff, dy, offset);
            dy2 += __shfl_down_sync(0xffffffff, dy2, offset);
            dxy += __shfl_down_sync(0xffffffff, dxy, offset);
        }

        if (tid == 0)
        {
            atomicAdd(&temp[0], dx);
            atomicAdd(&temp[1], dx2);
            atomicAdd(&temp[2], dy);
            atomicAdd(&temp[3], dy2);
            atomicAdd(&temp[4], dxy);
        }
        if (n == 0) {
            dx = temp[0];
            dx2 = temp[1];
            dy = temp[2];
            dy2 = temp[3];
            dxy = temp[4];

            double fenzi = dxy - ((dx * dy) / limit);
            double fenmu = sqrt((dx2 - (dx / limit)) * (dy2 - (dy / limit)));
            temp[5] = fenzi / fenmu;
        }
    }
int main(int argc, char* argv[]) {
        double hx[N] = {};
        double hy[N] = {};
        for(int cx = 0; cx < N ; ++cx) {
            hx[cx] = 1.000 * cx;
            hy[cx] = 2.000 * cx;
        }
        cudaMemcpyToSymbol(x,hx,sizeof(double)*N);
        cudaMemcpyToSymbol(y,hy,sizeof(double)*N);

        const int block_size = 128;
        const int grid_size =  (N + block_size - 1) / block_size;
        cor<<<block_size,grid_size,128>>>(N);//tags:<<<block_size,grid_size,128>>>


        int device_id = 0;
        cudaMemcpyFromSymbol(hx,temp,sizeof(double)*2);
        printf("c=%d\n",hx[0]);
}
