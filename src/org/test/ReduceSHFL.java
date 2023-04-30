package org.test;

import org.dao.ICuda;
import org.dao.base.cudaDeviceProp;

public class ReduceSHFL implements ICuda {
    public final int N = 1000;//global
    public int[] __device__d_x;//length:N
    public int[] __device__d;//length:2
    public void __global__reduce() {
        final int tid = threadIdxx;
        final int bid = blockIdxx;
        final int n = tid;
        int[] __shared__s_y = {};//extern
        if (n < N) {
            __shared__s_y[tid] = ICuda.__ldg(__device__d_x[n]);
        }
        ICuda.__syncthreads();

        for (int offset = blockDimx >> 1; offset >= 32; offset >>= 1)
        {
            if (tid < offset)
            {
                __shared__s_y[tid] += __shared__s_y[tid + offset];
            }
            ICuda.__syncthreads();
        }

        int y = __shared__s_y[tid];
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            y += ICuda.__shfl_down_sync("0xffffffff", y, offset);
        }

        if (tid == 0)
        {
            ICuda.atomicAdd(__device__d[0], y);
        }
    }

    @Override
    public void main() {
        int[] ha = new int[N];
        for(int x = 0; x < N ; ++x) {
            ha[x] = 1;
        }
        ha[0] = ICuda.atoi(argv[1]);

        ICuda.cudaMemcpyToSymbol("d_x","ha",ICuda.sizeof("int")*N);

        final int block_size = 128;
        final int grid_size =  (N + block_size - 1) / block_size;

        __global__reduce();//tags:<<<block_size,grid_size,128>>>


        int device_id = 0;
        ICuda.cudaMemcpyFromSymbol("ha","d",ICuda.sizeof("int")*2);
        System.out.printf("c=%d\n",ha[0]);
    }//End
}
