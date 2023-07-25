package org.test;

import org.dao.ICuda;
import org.dao.base.cudaEvent_t;
import org.dao.base.th.blockDim;
import org.dao.base.th.blockIdx;
import org.dao.base.th.threadIdx;

public class Sum implements ICuda {
    public int N = 1;//global
    public int[] __device__d_z;//length:10
    public void __global__sum(int[] $d_x,int N) {
        final int tid = threadIdx.x;
        final int bid = blockIdx.x;
        int[] __shared__s_x = {};//length:128
        final int n = bid * blockDim.x + tid;
        int x = 0;
        if(n < N) {
            __shared__s_x[n] = $d_x[n];
        }
        ICuda.__syncthreads();
        for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
        {
            if (tid < offset)
            {
                __shared__s_x[tid] += __shared__s_x[tid + offset];
            }
            ICuda.__syncthreads();
        }
        x = __shared__s_x[tid];
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            x += ICuda.__shfl_down_sync("0xffffffff",x, offset);
        }


        if (tid == 0)
        {

            ICuda.atomicAdd(__device__d_z[0], x);
        }
    }//End
    @Override
    public void main() {
        N = argc - 1;
        int[] $a = (int[]) ICuda.malloc(ICuda.sizeof("int")*N);
        int[] $b = ICuda.malloc(ICuda.sizeof("int")*10);
        for (int x = 0 ;x < N;x ++) {
            $a[x] = ICuda.atoi(argv[x + 1]);
        }
        int[] $d_x = {};
        ICuda.cudaMalloc("$d_x",ICuda.sizeof("int")*N);
        ICuda.cudaMemcpy("$d_x","$a",ICuda.sizeof("int")*N,cudaMemcpyHostToDevice);
        final int block_size = 128;
        final int grid_size =  (N + block_size - 1) / block_size;
        cudaEvent_t start = new cudaEvent_t();
        cudaEvent_t stop = new cudaEvent_t();
        ICuda.cudaEventCreate(start);
        ICuda.cudaEventCreate(stop);
        ICuda.cudaEventRecord(start);

        __global__sum($d_x,N);
        ICuda.cudaDeviceSynchronize();

        ICuda.cudaEventRecord(stop);
        ICuda.cudaEventSynchronize(stop);
        float elapsed_time = 0;
        ICuda.cudaEventElapsedTime(elapsed_time,start,stop);
        System.out.printf("Time:%g ms\n",elapsed_time);

        ICuda.cudaDeviceSynchronize();
        ICuda.cudaMemcpyFromSymbol("b","d_z",ICuda.sizeof("int")*10);
        System.out.printf("x:%d \n",$b[0]);

        ICuda.free($a);
        ICuda.free($b);
        ICuda.cudaFree($d_x);
    }//End
}
