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
        int *g_x;
        cudaMalloc((void **)&g_x, sizeof(int) * N);
        curandGenerateNormalInt(generator, g_x, N, 200,10);
        int *x = (int*) calloc(N, sizeof(int));
        cudaMemcpy(x, g_x, sizeof(int) * N, cudaMemcpyDeviceToHost);

        for (int y = 0; y < N ;y ++) {
            printf("%g\n",x[y]);
        }
        cudaFree(g_x);
        free(x);
}
