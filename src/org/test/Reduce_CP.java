
/**
package org.test;

import org.dao.ICuda;
import org.dao.base.thread_block_tile;
public class Reduce_CP implements ICuda {
    public float[] __device__d_x;//length:10000
    public float[] __device__d_y;//length:2
    public void __global__reduce_cp(final int N) {
        final int tid = threadIdxx;
        final int bid = blockIdxx;
        final int n = bid * blockDimx + tid;
        float __shared__s_y[] = {};//length:128
        __shared__s_y[tid] = (n < N) ? __device__d_x[n] : (float) 0.0;
        ICuda.__syncthreads();

        for (int offset = blockDimx >> 1; offset >= 32; offset >>= 1)
        {
            if (tid < offset)
            {
                __shared__s_y[tid] += __shared__s_y[tid + offset];
            }
            ICuda.__syncthreads();
        }

        float y = __shared__s_y[tid];

        thread_block_tile[] g32 = ICuda.tiled_partition(ICuda.this_thread_block());//length:32
        for (int i = g32.size() >> 1; i > 0; i >>= 1)
        {
            y += g32.shfl_down(y, i);
        }

        if (tid == 0)
        {
            ICuda.atomicAdd(__device__d_y, y);
        }
    }
}
**/