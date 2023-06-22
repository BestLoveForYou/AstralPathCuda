package org.test;

import org.dao.ICuda;
import org.dao.base.th.blockDim;
import org.dao.base.th.blockIdx;
import org.dao.base.th.gridDim;
import org.dao.base.th.threadIdx;

public class ReduceSHFL implements ICuda {
    public final int N = 1000;//global
    public double[] __device__d_y;//length:2
    public void __global__sum(double[] $d_x) {
        final int tid = threadIdx.x;
        final int bid = blockIdx.x;
        double[] __shared__s_y = {};//length:128
        final int n = bid * blockDim.x + tid;
        double y = 0.0;
        __shared__s_y[tid] = (n < N) ? $d_x[n] : 0.0;
        ICuda.__syncthreads();
        for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
        {
            if (tid < offset)
            {
                __shared__s_y[tid] += __shared__s_y[tid + offset];
            }
            ICuda.__syncthreads();
        }

        y = __shared__s_y[tid];

        for (int offset = 16; offset > 0; offset >>= 1)
        {
            y += ICuda.__shfl_down_sync("0xffffffff",y, offset);
        }

        if (tid == 0)
        {
            ICuda.atomicAdd(__device__d_y[0], y);
            System.out.printf("%f\n",y);
        }
    }
    @Override
    public void main() {
        double[] $a = ICuda.malloc(ICuda.sizeof("double")*N);
        double[] $b = ICuda.malloc(ICuda.sizeof("double")*2);
        for (int x = 0 ;x < N;x ++) {
            $a[x] = 11.0;
        }
        double[] $d_x = {};
        double[] $d_x1 = {};
        ICuda.cudaMalloc("$d_x",ICuda.sizeof("double")*N);
        ICuda.cudaMemcpy("$d_x","$a",ICuda.sizeof("double")*N,cudaMemcpyHostToDevice);
        final int block_size = 128;
        final int grid_size =  (N + block_size - 1) / block_size;
        __global__sum($d_x);
        ICuda.cudaDeviceSynchronize();
        ICuda.cudaMemcpyFromSymbol("b","d_y",ICuda.sizeof("double")*2);
        {
            System.out.printf("hello world");
        }
        System.out.printf("%f",$b[0]);
    }//End
}
