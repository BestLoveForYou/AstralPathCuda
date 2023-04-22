package org.test;

import org.dao.ICuda;
import org.dao.base.cudaStream_t;

public class TestNew implements ICuda {
    public final int N = 100;//global
    public void __global__hello(int[] $h,int[] $b) {
        int n = blockDimx * blockIdxx + threadIdxx;
        System.out.printf("(%d,%d,%d)\n",blockIdxx,blockIdxy,blockIdxz);
        $b[n] = $h[n];
    }//End
    @Override
    public void main() {
        int M = ICuda.sizeof("int")*N;
        int[] $h_x = (int[]) ICuda.malloc(M);
        int[] $h = {};

        for (int x = 0;x < N;x ++) {
            $h_x[x] = 1;
        }
        int[] $d_x1 = {};
        int[] $d_x2 = {};
        int[] $d_x3 = {};
        int[] $d_y1 = {};
        int[] $d_y2 = {};
        int[] $d_y3 = {};
        ICuda.cudaMalloc("d_x1",M);
        ICuda.cudaMalloc("d_x2",M);
        ICuda.cudaMalloc("d_x3",M);
        ICuda.cudaMemcpy("d_x1","h_x",M,"cudaMemcpyHostToDevice");
        ICuda.cudaMemcpy("d_x2","h_x",M,"cudaMemcpyHostToDevice");
        ICuda.cudaMemcpy("d_x3","h_x",M,"cudaMemcpyHostToDevice");
        ICuda.cudaMalloc("d_y1",M);
        ICuda.cudaMalloc("d_y2",M);
        ICuda.cudaMalloc("d_y3",M);

        final int block_size = 128;
        final int grid_size =  (N + block_size - 1) / block_size;
        cudaStream_t stream_t1 = new cudaStream_t();
        cudaStream_t stream_t2 = new cudaStream_t();
        cudaStream_t stream_t3 = new cudaStream_t();
        ICuda.cudaStreamCreate(stream_t1);
        ICuda.cudaStreamCreate(stream_t2);
        ICuda.cudaStreamCreate(stream_t3);
        __global__hello($d_x1,$d_y1);//[tags:<<<grid_size,block_size,0,stream_t1>>>]
        __global__hello($d_x2,$d_y2);//[tags:<<<grid_size,block_size,0,stream_t2>>>]
        __global__hello($d_x3,$d_y3);//[tags:<<<grid_size,block_size,0,stream_t3>>>]
        ICuda.cudaDeviceSynchronize();
        ICuda.cudaStreamSynchronize(stream_t1);
        ICuda.cudaStreamSynchronize(stream_t2);
        ICuda.cudaStreamSynchronize(stream_t3);
        ICuda.cudaMemcpy("h_x","d_y1",M,"cudaMemcpyDeviceToHost");


        ICuda.free($h_x);
        ICuda.cudaFree($d_x1);
        ICuda.cudaFree($d_x2);
        ICuda.cudaFree($d_x3);
        ICuda.cudaFree($d_y1);
        ICuda.cudaFree($d_y2);
        ICuda.cudaFree($d_y3);
        ICuda.cudaStreamDestroy(stream_t1);
        ICuda.cudaStreamDestroy(stream_t2);
        ICuda.cudaStreamDestroy(stream_t3);
    }//End
}
