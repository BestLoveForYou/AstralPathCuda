package org.test;

import org.dao.ICuda;
import org.dao.cuRAND.curandGenerator_t;

public class Rand implements ICuda {

    @Override
    public void main() {
        curandGenerator_t generator = new curandGenerator_t();
        ICuda.curandCreateGenerator(generator, CURAND_RNG_QUASI_SOBOL32);
        ICuda.curandSetPseudoRandomGeneratorSeed(generator, 1234);
        int N = 100000;
        double[] $g_x = {};
        ICuda.cudaMalloc($g_x, ICuda.sizeof("double") * N);
        ICuda.curandGenerateUniformDouble(generator, $g_x, N);
        double[] $x = ICuda.calloc(N, ICuda.sizeof("double"));
        ICuda.cudaMemcpy("x", "g_x", ICuda.sizeof("double") * N, cudaMemcpyDeviceToHost);

        for (int y = 0; y < N ;y ++) {
            System.out.printf("%g\n",$x[y]);
        }
        ICuda.cudaFree($g_x);
        ICuda.free($x);
    }//End
}
