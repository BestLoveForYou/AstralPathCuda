package org.test;

import org.dao.ICuda;

public class CorDemo implements ICuda {
    public final int N = 10000;//global
    public double[] __device__x;//length:N
    public double[] __device__y;//length:N
    public double[] __device__temp;//length:10
    public void __global__cor(int limit) {
        final int tid = threadIdxx;
        final int n = threadIdxx;
        double[] __shared__x = {};//extern
        double[] __shared__x2 = {};//extern
        double[] __shared__y = {};//extern
        double[] __shared__y2 = {};//extern
        double[] __shared__xy = {};//extern
        if (n < limit) {
            System.out.print("20");
            __shared__x[tid] = ICuda.__ldg(__device__x[n]);
            __shared__y[tid] = ICuda.__ldg(__device__y[n]);
            System.out.print("30");
            __shared__x2[tid] = __shared__x[tid] * __shared__x[tid];
            __shared__y2[tid] = __shared__y[tid] * __shared__y[tid];
            __shared__xy[tid] = __shared__x[tid] * __shared__y[tid];
        }
        ICuda.__syncthreads();
        System.out.print("20");
        for (int offset = blockDimx >> 1; offset >= 32; offset >>= 1)
        {
            if (tid < offset)
            {
                __shared__x[tid] += __shared__x[tid + offset];
                __shared__x2[tid] += __shared__x2[tid + offset];
                __shared__y[tid] += __shared__y[tid + offset];
                __shared__y2[tid] += __shared__y2[tid + offset];
                __shared__xy[tid] += __shared__xy[tid + offset];
            }
            ICuda.__syncthreads();
        }
        System.out.print("0");
        double dx = __shared__x[tid];
        double dx2 = __shared__x2[tid];
        double dy = __shared__y[tid];
        double dy2 = __shared__y2[tid];
        double dxy = __shared__xy[tid];
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            dx += ICuda.__shfl_down_sync("0xffffffff", dx, offset);
            dx2 += ICuda.__shfl_down_sync("0xffffffff", dx2, offset);
            dy += ICuda.__shfl_down_sync("0xffffffff", dy, offset);
            dy2 += ICuda.__shfl_down_sync("0xffffffff", dy2, offset);
            dxy += ICuda.__shfl_down_sync("0xffffffff", dxy, offset);
        }
        System.out.print("1");
        if (tid == 0)
        {
            ICuda.atomicAdd(__device__temp[0], dx);
            ICuda.atomicAdd(__device__temp[1], dx2);
            ICuda.atomicAdd(__device__temp[2], dy);
            ICuda.atomicAdd(__device__temp[3], dy2);
            ICuda.atomicAdd(__device__temp[4], dxy);

        }
        System.out.print("2\n");
        if (n == 1) {
            System.out.print("1");
            dx = __device__temp[0];
            dx2 = __device__temp[1];
            dy = __device__temp[2];
            dy2 = __device__temp[3];
            dxy = __device__temp[4];

            double fenzi = dxy - ((dx * dy) / limit);
            double fenmu = ICuda.sqrt((dx2 - (dx / limit)) * (dy2 - (dy / limit)));
            __device__temp[5] = fenzi / fenmu;
            System.out.printf("r: %f",fenzi / fenmu);
        }
    }
    @Override
    public void main() {
        double[] hx = new double[N];
        double[] hy = new double[N];
        for(int cx = 0; cx < N ; ++cx) {
            hx[cx] = 1.000 * cx + 1;
            hy[cx] = 2.000 * cx + 2;
        }
        ICuda.cudaMemcpyToSymbol("x","hx",ICuda.sizeof("double")*N);
        ICuda.cudaMemcpyToSymbol("y","hy",ICuda.sizeof("double")*N);
        final int block_size = 128;
        final int grid_size =  (N + block_size - 1) / block_size;
        __global__cor(1000);//tags:<<<block_size,grid_size,128>>>
        ICuda.cudaDeviceSynchronize();
        double[] re = new double[10];
        ICuda.cudaMemcpyFromSymbol("re","temp",ICuda.sizeof("double")*10);
        System.out.printf("c=%f\n",re[5]);
    }//End
}
