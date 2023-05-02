package org.dao;

import org.dao.base.*;
import org.dao.cuRAND.*;

public interface ICuda {
    String cudaMemcpyDeviceToHost = null;
    String cudaMemcpyHostToDevice = null;

    int N = 0;
    
    int argc = Integer.MAX_VALUE;
    String[] argv = new String[Integer.MAX_VALUE];
    
    int gridDimx = 0;
    int gridDimy = 0;
    int gridDimz = 0;

    int blockIdxx = 0;
    int blockIdxy = 0;
    int blockIdxz = 0;
    int threadIdxx = 0;
    int threadIdxy = 0;
    int threadIdxz = 0;
    int blockDimx = 128;
    int blockDimy = 0;
    int blockDimz = 0;


    /**
     * 这些是Cuda和C语言提供的部分函数
     */
    static <T> T malloc(int t) {return null;}
    static <T> void cudaMalloc(T s, int t) {}
    static <T> T calloc(int n, int size) {return null;}

    static void free(Object... o) {}
    static void cudaFree(Object... o) {}

    static float sqrt(float x) {return 1.0F;}
    static float sqrtf(float x) {return 1.0F;}
    static double sqrt(double x) {return 1.0;}
    static float __fsqrt_rd(float x) {return 1.0F;}
    static float __fsqrt_rn(float x) {return 1.0F;}
    static float __fsqrt_ru(float x) {return 1.0F;}
    static float __fsqrt_rz(float x) {return 1.0F;}
    static double __fsqrt_rd(double x) {return 1.0;}
    static double __fsqrt_rn(double x) {return 1.0;}
    static double __fsqrt_ru(double x) {return 1.0;}
    static double __fsqrt_rz(double x) {return 1.0;}

    static <T> T atomicAdd(T address, T val) {return null;}
    static <T> T atomicSub(T address, T val) {return null;}
    static <T> T atomicExch(T address, T val) {return null;}
    static <T> T atomicMin(T address, T val) {return null;}
    static <T> T atomicMax(T address, T val) {return null;}
    static <T> T atomicInc(T address, T val) {return null;}
    static <T> T atomicDec(T address, T val) {return null;}
    static <T> T atomicCAS(T address, T compare,T val) {return null;}
    static <T> T atomicAnd(T address, T val) {return null;}
    static <T> T atomicOr(T address, T val) {return null;}
    static <T> T atomicXor(T address, T val) {return null;}

    static cudaError_t cudaMemcpy(final String symbol, final String src, int count,String fromto){return null;}
    static cudaError_t cudaMemcpyToSymbol(final String symbol, final String src, int count){return null;}
    static cudaError_t cudaMemcpyFromSymbol(final String src,final String symbol,int count){return null;}
    static void __syncthreads(){}//线程同步
    static void __syncwarp(){}//线程束同步

    /**
     * 随机数
     */
    curndRngType_t CURAND_RNG_PSEUDO_DEFAULT = new curndRngType_t();
    curndRngType_t CURAND_RNG_PSEUDO_XORWOW = new curndRngType_t();
    curndRngType_t CURAND_RNG_PSEUDO_MPG19937 = new curndRngType_t();
    curndRngType_t CURAND_RNG_PSEUDO_MRG32K3A = new curndRngType_t();
    curndRngType_t CURAND_RNG_PSEUDO_PHILOX4_32_10 = new curndRngType_t();
    curndRngType_t CURAND_RNG_PSEUDO_MT19937 = new curndRngType_t();
    //上面是伪随机数生成
    //下面是拟随机数数生成
    curndRngType_t CURAND_RNG_QUASI_SOBOL32 = new curndRngType_t();
    curndRngType_t CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = new curndRngType_t();
    curndRngType_t CURAND_RNG_QUASI_SOBOL64 = new curndRngType_t();
    curndRngType_t CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = new curndRngType_t();


    static void curandCreateGenerator(curandGenerator_t c, curndRngType_t rng_type) {}//c是指针
    static curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t c,long seed) {return null;}
    static curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator,String seed) {
        return null;
    }
    static curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator,unsigned seed) {
        return null;
    }
    static curandStatus_t curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator, int num_dimensions) {
        return null;
    }
    static curandStatus_t curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator, unsigned num_dimensions) {
        return null;
    }
    static curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator, unsigned offset) {
        return null;
    }
    static curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator, long offset) {
        return null;
    }
    curandOrdering_t CURAND_ORDERING_PSEUDO_DEFAULT = null;
    curandOrdering_t CURAND_ORDERING_PSEUDO_LEGACY = null;
    curandOrdering_t CURAND_ORDERING_PSEUDO_BEST = null;
    curandOrdering_t CURAND_ORDERING_PSEUDO_SEEDED = null;
    curandOrdering_t CURAND_ORDERING_QUASI_DEFAULT = null;

    static curandStatus_t curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order) {
        return null;
    }
    static curandStatus_t curandSetStream(curandGenerator_t generator, cudaStream_t stream) {
        return null;
    }


    static <T> void curandGenerateUniformDouble(curandGenerator_t c, T memory, int N) {}
    static <T> void curandGenerateNormalDouble(curandGenerator_t generator, T $gX, int N,double a,double b) {}
    static <T> void curandGenerateNormalDouble(curandGenerator_t generator, double[] $gX, int N,double a,double b) {}
    /**
     * 下面是Device使用的API;上面是Host使用的API
     */
    static <T> void curand_init(T seed, T sequence, T offset, curandState_t state) {}

    static int curand (curandState_t state) {return 0;}

    /**
     * 运行时API函数
     */
    static void cudaDeviceSynchronize() {}

    static void cudaSetDevice(int id) {}
    static void cudaGetDeviceProperties(cudaDeviceProp pop,int id) {}

    static <T> void cudaGetSymbolAddress(T devPtr,T symbol) {}

    /**
     * 流
     */
    static void cudaStreamCreate(cudaStream_t c) {}
    static void cudaStreamDestroy(cudaStream_t c) {}
    static cudaError_t cudaStreamSynchronize(cudaStream_t c) {return new cudaError_t();}
    static cudaError_t cudaStreamQuery(cudaStream_t c) {return new cudaError_t();}
    /**
     * 线程数
     * @param mask
     * @param predicate
     * @return
     */
    static int __any_sync(unsigned mask,int predicate){return 0;}

    static int __all_sync(unsigned mask,int predicate){return 0;}

    static int __any_sync(String mask,int predicate){return 0;}

    static int __all_sync(String mask,int predicate){return 0;}

    static unsigned __ballot_sync(String mask, int predicate){return null;}

    static unsigned __ballot_sync(String mask,boolean pre){return null;}

    static <T> T __shfl_sync(String mask, T v, int srcLane,int w){return null;}

    static <T> T __shfl_up_sync(String mask, T v, int srcLane,int w){return null;}

    static <T> T __shfl_down_sync(String mask, T v, int srcLane,int w){return null;}

    static <T> T __shfl_xor_sync(String mask, T v, int srcLane,int w){return null;}

    static <T> T __shfl_sync(String mask, T v, int srcLane){return null;}

    static <T> T __shfl_up_sync(String mask, T v, int srcLane){return null;}

    static <T> T __shfl_down_sync(String mask, T v, int srcLane){return null;}

    static <T> T __shfl_xor_sync(String mask, T v, int srcLane){return null;}

    static int sizeof(String s) {return 1;}
    static int atoi(String c) {return 1;}

    static long atol(String c) {return 1;}

    static void printf(String format,Object... argv) {}

    static void print(String format,Object... argv) {}

    static <T> T __ldg(T j) {return null;}

    static int this_thread_block() {
        return 1;
    }
    static thread_group tiled_partition(thread_block a,int b) {
        return new thread_group();
    }
    static thread_group tiled_partition(thread_group a,int b) {
        return new thread_group();
    }
    static thread_block_tile[] tiled_partition(int b) {
        return new thread_block_tile[]{new thread_block_tile()};
    }
    void main();//主函数
}