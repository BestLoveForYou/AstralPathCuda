package org.test;

import org.dao.ICuda;
import org.dao.base.th.blockDim;
import org.dao.base.th.blockIdx;
import org.dao.base.th.threadIdx;
import org.dao.cuRAND.curandGenerator_t;

public class ConnectT implements ICuda {
    public final int N = 640;//global
    public void __global__connect(double[] $health,double[] $attract,double[] $point) {
        final int tid = threadIdx.x;
        final int bid = blockIdx.x;
        final int n = bid * blockDim.x + tid;

    }
    @Override
    public void main() {
        curandGenerator_t generator = new curandGenerator_t();
        ICuda.curandCreateGenerator(generator, CURAND_RNG_QUASI_SOBOL32);

        double[] $g_health = {};
        double[] $g_attract = {};
        double[] $g_point = {};
        ICuda.curandSetPseudoRandomGeneratorSeed(generator, 1234);
        ICuda.cudaMalloc($g_health, ICuda.sizeof("double") * N);
        ICuda.curandGenerateNormalDouble(generator, $g_health, N,100.0,20.0);
        ICuda.curandSetPseudoRandomGeneratorSeed(generator, 12345);
        ICuda.cudaMalloc($g_attract, ICuda.sizeof("double") * N);
        ICuda.curandGenerateNormalDouble(generator, $g_attract, N,100.0,20.0);
        ICuda.cudaMalloc($g_point, ICuda.sizeof("double") * N);

        final int block_size = 128;
        final int grid_size =  (N + block_size - 1) / block_size;
        __global__connect($g_health,$g_attract,$g_point);
        ICuda.cudaDeviceSynchronize();

        ICuda.cudaFree($g_health);

    }//End
}
