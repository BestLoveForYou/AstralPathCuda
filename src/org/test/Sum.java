package org.test;

import org.dao.ICuda;

public class Sum implements ICuda {
    public final int N = 128;//global
    public int[] __device__d;//length:128
    public int[] __device__e;//length:128
    public void __global__hello() {//Start
        final int x = threadIdxx;
        final int y = threadIdxy;
        int n = blockDimx * blockIdxx + threadIdxx;
        int a = __device__d[n];
        int b = __device__e[n];
        __device__d[n] = a*b;
    }//End

    @Override
    public void main() {
        int[] ha = {};//length:128
        for(int x = 0; x < 128 ; ++x) {
            ha[x] = 5;
        }
        ICuda.cudaMemcpyToSymbol("d","ha",ICuda.sizeof("int")*2);
        for(int x = 0; x < 128 ; ++x) {
            ha[x] = 7;
        }
        ICuda.cudaMemcpyToSymbol("e","ha",ICuda.sizeof("int")*2);
        final int block_size = 128;
        final int grid_size = N / block_size;

        __global__hello();
        ICuda.cudaMemcpyFromSymbol("ha","d",ICuda.sizeof("int")*2);
        System.out.printf("c=%d\n",ha[0]);
    }//End

}
