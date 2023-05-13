package org.test;

import org.dao.ICuda;
import org.dao.base.cudaStream_t;

public class TestNew implements ICuda {
    public final int N = 100;//global
    public void __global__hello(int[] $h,int[] $b) {
        int n = threadIdxx;
        System.out.printf("(%d,%d)\n",threadIdxx,threadIdxy);
        $b[n] = $h[n];

    }//End


    @Override
    public void main() {
        int M = ICuda.sizeof("int")*N;
        int[] $h_x = ICuda.malloc(M);
        int[] $h = {};

        for (int x = 0;x < N;x ++) {
            $h_x[x] = 1;
        }
        int[] $d_x1 = {};

        int[] $d_y1 = {};

        ICuda.cudaMalloc("d_x1",M);

        ICuda.cudaMemcpy("d_x1","h_x",M,cudaMemcpyHostToDevice);

        ICuda.cudaMalloc("d_y1",M);

        final int block_size = 128;
        final int grid_size =  (N + block_size - 1) / block_size;

        __global__hello($d_x1,$d_y1);

        ICuda.cudaDeviceSynchronize();

        ICuda.cudaMemcpy("h_x","d_y1",M,cudaMemcpyDeviceToHost);


        ICuda.free($h_x);
        ICuda.cudaFree($d_x1);

        ICuda.cudaFree($d_y1);

    }//End
}
