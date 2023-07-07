package org.test;

import org.dao.ICuda;
import org.dao.base.th.blockDim;
import org.dao.base.th.blockIdx;
import org.dao.base.th.gridDim;
import org.dao.base.th.threadIdx;

public class ReduceSHFL implements ICuda {
    public final int N = 5;//global
    public double[] __device__d_z;//length:10
    public void __global__sum(double[] $d_x,double[] $d_y) {
        final int tid = threadIdx.x;
        final int bid = blockIdx.x;
        double[] __shared__s_x = {};//length:128
        double[] __shared__s_x2 = {};//length:128
        double[] __shared__s_y = {};//length:128
        double[] __shared__s_xy = {};//length:128
        final int n = bid * blockDim.x + tid;
        double x = 0.0;
        double y = 0.0;
        double x2 = 0.0;
        double xy = 0.0;
        __shared__s_x[tid] = (n < N) ? $d_x[n] : 0.0;
        __shared__s_x2[tid] = (n < N) ? (__shared__s_x[tid] * __shared__s_x[tid]) : 0.0;
        __shared__s_y[tid] = (n < N) ? $d_y[n] : 0.0;

        __shared__s_xy[tid] = (n < N) ? __shared__s_x[tid] * __shared__s_y[tid] : 0.0;
        ICuda.__syncthreads();
        for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
        {
            if (tid < offset)
            {
                __shared__s_x[tid] += __shared__s_x[tid + offset];
                __shared__s_y[tid] += __shared__s_y[tid + offset];
                __shared__s_x2[tid] += __shared__s_x2[tid + offset];
                __shared__s_xy[tid] += __shared__s_xy[tid + offset];
            }
            ICuda.__syncthreads();
        }

        y = __shared__s_y[tid];
        x = __shared__s_x[tid];
        x2 = __shared__s_x2[tid];
        xy = __shared__s_xy[tid];
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            y += ICuda.__shfl_down_sync("0xffffffff",y, offset);
            x += ICuda.__shfl_down_sync("0xffffffff",x, offset);
            x2 += ICuda.__shfl_down_sync("0xffffffff",x2, offset);
            xy += ICuda.__shfl_down_sync("0xffffffff",xy, offset);
        }

        if (tid == 0)
        {
            ICuda.atomicAdd(__device__d_z[0], x);
            ICuda.atomicAdd(__device__d_z[1], y);
            ICuda.atomicAdd(__device__d_z[2], x2);
            ICuda.atomicAdd(__device__d_z[3], xy);

        }

        if(n == 0) {
            double fenzi = __device__d_z[3] - (__device__d_z[0] * __device__d_z[1] / N);
            double fenmu = __device__d_z[2] - (__device__d_z[0] * __device__d_z[0] / N);
            __device__d_z[4] = fenzi / fenmu;
            __device__d_z[5] = (__device__d_z[1] / N) - ((__device__d_z[0] / N) * __device__d_z[4]);

        }
    }
    @Override
    public void main() {
        double[] $a = ICuda.malloc(ICuda.sizeof("double")*N);
        double[] $a2 = ICuda.malloc(ICuda.sizeof("double")*N);
        double[] $b = ICuda.malloc(ICuda.sizeof("double")*10);
        for (int x = 0 ;x < N;x ++) {
            $a[x] = 11 + 1.0 * x;
            $a2[x] = 11 + 3.0 * x;
        }
        double[] $d_x = {};
        double[] $d_y = {};
        ICuda.cudaMalloc("$d_x",ICuda.sizeof("double")*N);
        ICuda.cudaMalloc("$d_y",ICuda.sizeof("double")*N);
        ICuda.cudaMemcpy("$d_x","$a",ICuda.sizeof("double")*N,cudaMemcpyHostToDevice);
        ICuda.cudaMemcpy("$d_y","$a2",ICuda.sizeof("double")*N,cudaMemcpyHostToDevice);
        final int block_size = 128;
        final int grid_size =  (N + block_size - 1) / block_size;
        __global__sum($d_x,$d_y);
        ICuda.cudaDeviceSynchronize();
        ICuda.cudaMemcpyFromSymbol("b","d_z",ICuda.sizeof("double")*10);
        System.out.printf("x sum:  %f \n",$b[0]);
        System.out.printf("y sum: %f \n",$b[1]);
        System.out.printf("x2 sum: %f \n",$b[2]);
        System.out.printf("xy sum: %f \n",$b[3]);
        System.out.printf("Result: y =  %f * x + (%f) \n",$b[4],$b[5]);
    }//End
}