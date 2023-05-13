package org.test;

import org.dao.ICuda;
import org.dao.base.cudaStream_t;

public class TestStream implements ICuda {
    public final int N = 1000;//global
    public void __global__test(int[] $h,int[] $b,int x) {
        System.out.printf("Stream: %d - (%d,%d)\n",x,threadIdxx,threadIdxy);
        $b[threadIdxx] = $h[threadIdxx];
    }
    @Override
    public void main() {
        cudaStream_t stream_1 = new cudaStream_t();
        cudaStream_t stream_2 = new cudaStream_t();

        ICuda.cudaStreamCreate(stream_1);
        ICuda.cudaStreamCreate(stream_2);

        int M = ICuda.sizeof("int")*N;
        int[] $h_x = ICuda.malloc(M);
        for (int x = 0;x < N;x ++) {
            $h_x[x] = 1;
        }
        int[] $d_x1 = {};
        int[] $d_y1 = {};
        ICuda.cudaMallocHost("d_x1",M);
        ICuda.cudaMemcpyAsync("d_x1","h_x",M,cudaMemcpyHostToDevice,stream_1);
        ICuda.cudaMemcpyAsync("d_x1","h_x",M,cudaMemcpyHostToDevice,stream_2);
        ICuda.cudaMallocHost("d_y1",M);

        final int block_size = 128;
        final int grid_size =  (N + block_size - 1) / block_size;

        __global__test($d_x1,$d_y1,0);//tags:<<<grid_size,block_size,0,stream_1>>>
        __global__test($d_x1,$d_y1,1);//tags:<<<grid_size,block_size,0,stream_2>>>

        ICuda.cudaStreamSynchronize(stream_1);
        ICuda.cudaStreamSynchronize(stream_2);

        ICuda.free($h_x);
        ICuda.cudaFreeHost($d_x1);
        ICuda.cudaFreeHost($d_y1);
        ICuda.cudaStreamDestroy(stream_1);
        ICuda.cudaStreamDestroy(stream_2);
    }//End
}
