#include<stdio.h>
using namespace std;
#include<curand.h>
//现在开始全局变量书写
//现在开始全局内存书写
int main(int argc, char* argv[]) {
        curandGenerator_t generator ;
        curandCreateGenerator(&generator, CURAND_RNG_QUASI_SOBOL32);
        curandSetPseudoRandomGeneratorSeed(generator, 1234);
        int N = 100000;
        double *g_x;
        cudaMalloc((void **)&g_x, sizeof(double) * N);
        curandGenerateUniformDouble(generator, g_x, N);
        double *x = (double*) calloc(N, sizeof(double));
        cudaMemcpy(x, g_x, sizeof(double) * N, cudaMemcpyDeviceToHost);

        for (int y = 0; y < N ;y ++) {
            printf("%g\n",x[y]);
        }
        cudaFree(g_x);
        free(x);
}
