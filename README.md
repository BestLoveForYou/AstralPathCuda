
@[TOC]

# AstralPathCuda 文档
2023-04-22 20:57:09 星期六

## 介绍
使用GPU完成科学计算不仅能提高计算的稳健性,更能提供一个无与伦比的速度,Java也是一种常用的CPU语言,开源平台为其提供了丰富的环境,但Cuda和Java的混合使用一直都是个麻烦事,毕竟两者本来都不该有交集,但GPU的高并发能力强大无比,CSDN上基本都是Java通过JNI调用CUDA程序,但麻烦无比,而使用JCuda也需要提供核函数的PTX文件,更加麻烦,还增加了开发难度,鉴于此前景,我开始了JavaGPU混合开发辅助包AstralPathCuda(以下简称APC)的编写.
在项目的最开始,只做了一些很简单的功能,比如线程同步函数,简单的核函数,线程索引...
随着项目的逐渐深入,开发的功能和难度也是越来越高,一些最开始的不适合定义也在现在被更改,到目前为止,使用APC已经可以开发出一个有模有样的复杂Cuda程序了,从动静态的全局内存;共享内存,到原子函数,再到现在已经基本完成的线程束内函数,都已经逐渐完善了.
之前在解决Java中使用指针这个麻烦时,花了我很长的一段时间和精力来想如何实现...但好在,主要的问题已经解决了.
接下来就是对APC使用的介绍了,若有不清楚的,可以在这篇文章下留言,也是为这个项目做出一些贡献吧.


~~这里的显卡都是指的是英伟达的显卡~~

## Cuda下载与安装
[CUDA安装教程（超详细）](https://blog.csdn.net/m0_45447650/article/details/123704930?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167222191016782427411200%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=167222191016782427411200&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-123704930-null-null.142%5Ev68%5Econtrol,201%5Ev4%5Eadd_ask,213%5Ev2%5Et3_esquery_v2&utm_term=cuda%E5%AE%89%E8%A3%85&spm=1018.2226.3001.4187)

## 下载
[点击下载](http://47.96.154.95:9000/down/nNut6SNaoxkc.jar)

## 核心
在Cuda中,一个Cuda程序是由C/C++编写的,而APC使用的语言是Java,所以,或多或少的会有很多语言上的底层差异,对此,我这里的解决方法是```注释辅助+APC自动填充```,在实际编写中使用者会感受到这种方法的快捷和灵活的.
在APC在开发中最基本的单位是**类**,再导入AstralPaCuda开发包后,这个类需要使用ICuda的接口,ICuda提供了一些Cuda函数,变量,以及C的部分函数

## 数组
数组是一个比较特殊的存在,由于java中的数组和C中的数组定义方式有较大不同,所以需要设置提示符.
数组定义如下
例:一个叫a的int型数组,长度3
```java
int[] a = {1,2,3};//length:3
```
需要在数组后标注长度

## 指针
指针的```*```将以```$```形式出现,
如:
```java
double[] $a = (double[]) ICuda.malloc(ICuda.sizeof("double")*2); 
```
将会转义为
```c
double *a = (double*) malloc(sizeof(double)*2);
```
这是一个长度为2的double指针

## 全局内存

### 静态全局内存

定义
```java
public int[] __device__d_x;//length:128
```
*例:int类型,名为d_x,长度为128*,数组
```java
public int __device__d;
```
*例:int类型,名为d
在核函数中的调用
```java
int x = __device__d_x[1];
```

在主函数中向全局内存传递数据,如:
```java
double[] ha = {};//length:128
        for(int x = 0; x < N ; ++x) {
        ha[x] = 1.23;
        }
        ICuda.cudaMemcpyToSymbol("d_x","ha",ICuda.sizeof("double")*128);
```
将会转义为
```c
double ha[128] =  {};//length:128
for(int x = 0; x < N ; ++x) {
    ha[x] = 1.23;
}
cudaMemcpyToSymbol(d_x,ha,sizeof(double)*128);
```

在这里可以发现参数在全局内存的参数加了引号,这是必不可少的

```cudaMemcpyFromSymbol```是同样的用法

注意:
- 这里的核函数在核函数调用都是用```__device__d_x```名称
  而在像cudaMemcpyFromSymbol方法中使用的名称是```d_x```,而不是```__device__d_x```
- 支持```__ldg()```缓存加速

### 动态全局内存
例:
```java
double[] $a = {};
        ICuda.cudaMalloc("$a",ICuda.sizeof("double")*2);
```
用一个空指针,分配了显存,大小为2个
接下来,让我们看一下动态全局内存传递

```java
double[] $b = (double[]) ICuda.malloc(ICuda.sizeof("double")*20000);
        double[] $a = {};
        ICuda.cudaMalloc("$a",ICuda.sizeof("double")*20000);
        ICuda.cudaMemcpy("$a","$b",ICuda.sizeof("double")*20000,"cudaMemcpyHostToDevice");
```

同样的,动态全局内存可见性与cuda中的无异

## 共享内存
### 静态共享内存

定义一个类型为double长度为128的共享数组
```java
double[] __shared__s_y = {};//length:128
```
这个数组在核函数中调用名就为```__shared__s_y```,而不是像全局内存一样的```s_y```

### 动态共享内存
```java
double[] __shared__s_y = {};//extern
```
其次就是在核函数中设置了

## 函数

**从java的方法转为c中的函数**,共有2种必不可少,
- 第一是核函数,是在GPU上跑的函数
- 第二是main函数,就是主函数

以上函数在最后都需要在方法**最后的"}"**后标注**//End**(注释形式)

主函数自带2个变量```int argc, char* argv[]```这是启动参数,都可直接使用

在主函数的默认调用中,将会将核函数的java调用(普通方法调用)转为Cuda调用
如:

核函数转义时，若无任何注释，如：
```java
__global__reduce();
```
将会转义为
```c
reduce<<<grid_size,block_size>>>();
```
这也意味着在此函数的上方应该定义了grid_size和block_size,如:
```java
final int block_size = 128;
final int grid_size =  (N + block_size - 1) / block_size;
        __global__reduce();
```
若是有注释提示,如：
```
__global__reduce();//tags:<<<block_size,grid_size,128>>>
```
则会转义为

若是有参数,也会同时带上的
```
__global__reduce<<<block_size,grid_size,128>>>();//tags:<<<block_size,grid_size,128>>>
```


也可以用复杂一点的```dim3```类型
```java
dim3 grid_size = new dim3(2,2,2);
        dim3 block_size = new dim3(3,3,3)
        __global__reduce();
```

其中的任何一个变量名都不建议更改和重复使用

### 核函数

核函数的定义:
```java
public void __global__核函数名() {
//省略
        }//End
```
**注意**:最后的//End不能省略

### 设备函数
```java
public 任意返回值 __device__核函数名() {

}//End
```
*目前设备函数任有较多缺陷,需要逐步解决*


## 原子函数

支持:
- atomicAdd
- atomicSub
- atomicExch
- atomicCAS
- atomicInc
- atomicDec
- atomicMax
- atomicMin
- atomicAnd
- atomicOr
- atomicXor
  所有原子函数使用方法不变,可以正常使用,无需更改

## 线程束

支持线程束内同步函数```__syncwarp()```
支持线程束表决函数
```java
	static int __any_sync(unsigned mask,int predicate){return 0;};
static int __all_sync(unsigned mask,int predicate){return 0;};
static int __any_sync(String mask,int predicate){return 0;};
static int __all_sync(String mask,int predicate){return 0;};
static unsigned __ballot_sync(String mask, int predicate){return null;};
static unsigned __ballot_sync(String mask,boolean pre){return null;};
```
这里不仅包含unsigned的类型,也有String类型,为的是开发方便
支持线程束洗牌函数
```java
static <T> T __shfl_sync(String mask, T v, int srcLane,int w){return null;};
static <T> T __shfl_up_sync(String mask, T v, int srcLane,int w){return null;};
static <T> T __shfl_down_sync(String mask, T v, int srcLane,int w){return null;};
static <T> T __shfl_xor_sync(String mask, T v, int srcLane,int w){return null;};
```
上面是带参数w的,
下面是不带参数w的.
```java
    static <T> T __shfl_sync(String mask, T v, int srcLane){return null;};
static <T> T __shfl_up_sync(String mask, T v, int srcLane){return null;};
static <T> T __shfl_down_sync(String mask, T v, int srcLane){return null;};
static <T> T __shfl_xor_sync(String mask, T v, int srcLane){return null;};
```

## 运行时API函数

```java
static void cudaDeviceSynchronize() {};
static void cudaSetDevice(int id) {}
static void cudaGetDeviceProperties(cudaDeviceProp pop,int id) {};
```
以及可以查看设备的API函数
```cudaDeviceProp类```
使用如下如:
```java
		int device_id = 0;
        ICuda.cudaSetDevice(device_id);
        cudaDeviceProp prop = new cudaDeviceProp();
        ICuda.cudaGetDeviceProperties(prop,device_id);
        ICuda.printf("Device id:                                 %d\n",
        device_id);
        ICuda.printf("Device name:                               %s\n",
        prop.name);
```
将会输出显卡的信息

-----

## curand标准库
**已支持所有在Host主机上实现的方法!**
### Host调用API
- 获取算法
```java
static curndRngType_t CURAND_RNG_PSEUDO_DEFAULT = new curndRngType_t();
    static curndRngType_t CURAND_RNG_PSEUDO_XORWOW = new curndRngType_t();
    static curndRngType_t CURAND_RNG_PSEUDO_MPG19937 = new curndRngType_t();
    static curndRngType_t CURAND_RNG_PSEUDO_MRG32K3A = new curndRngType_t();
    static curndRngType_t CURAND_RNG_PSEUDO_PHILOX4_32_10 = new curndRngType_t();
    static curndRngType_t CURAND_RNG_PSEUDO_MT19937 = new curndRngType_t();
	//上面是伪随机数生成
    //下面是拟随机数数生成
    static curndRngType_t CURAND_RNG_QUASI_SOBOL32 = new curndRngType_t();
    static curndRngType_t CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = new curndRngType_t();
    static curndRngType_t CURAND_RNG_QUASI_SOBOL64 = new curndRngType_t();
    static curndRngType_t CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = new curndRngType_t();
```
- 创建:
```java
 static void curandCreateGenerator(curandGenerator_t c, curndRngType_t rng_type) {}//c是指针
```
- Generator Options
```java
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
```
-- Seed
是一个64-bit integar，用来初始化伪随机数生成器的起始状态，相同的seed经常生成相同的随机数序列
-- Offset
offset参数用来跳过随机数序列的开头，第一个随机数从序列的第offset个开始取，这使得多次运行同一程序，从相同随机数序列生成的随机数不重叠。is not available for the CURAND_RNG_PSEUDO_MTGP32 and CURAND_RNG_PSEUDO_MT19937 generators
-- Order
用来选择结果如何在全局内存中排序
CURAND_ORDERING_PSEUDO_DEFAULT
CURAND_ORDERING_PSEUDO_LEGACY
CURAND_ORDERING_PSEUDO_BEST
CURAND_ORDERING_PSEUDO_SEEDED
CURAND_ORDERING_QUASI_DEFAULT (用于拟随机数的选项)
- Generation Functions
```java
static <T> void curandGenerateUniformDouble(curandGenerator_t c, T memory, int N) {}
    static <T> void curandGenerateNormalDouble(curandGenerator_t generator, T $gX, int N,double a,double b) {}
    static <T> void curandGenerateNormalDouble(curandGenerator_t generator, double[] $gX, int N,double a,double b) {}
    static <T> void curandGenerateNormalFloat(curandGenerator_t generator, float[] $gX, int N,float a,float b) {}
    static <T> void curandGenerateNormalInt(curandGenerator_t generator, int[] $gX, int N,int a,int b) {}
    static <T> void curandGenerateNormalLong(curandGenerator_t generator, long[] $gX, int N,long a,long b) {}
```

~~至于Device上的API,还正在做~~
[========]


# 附页

## 示例程序

### 使用线程束内函数的数组求和

Java源码:
```java
public class ReduceSHFL implements ICuda {
  public final int N = 100000;//global
  public double[] __device__d_x;//length:100000
  public double[] __device__d;//length:2
  public void __global__reduce() {
    final int tid = threadIdxx;
    final int bid = blockIdxx;
    final int n = bid * blockDimx + tid;
    double[] __shared__s_y = {};//length:128
    if (n < N) {
      __shared__s_y[tid] = ICuda.__ldg(__device__d_x[n]);
    }
    ICuda.__syncthreads();

    for (int offset = blockDimx >> 1; offset >= 32; offset >>= 1)
    {
      if (tid < offset)
      {
        __shared__s_y[tid] += __shared__s_y[tid + offset];
      }
      ICuda.__syncthreads();
    }

    double y = __shared__s_y[tid];
    for (int offset = 16; offset > 0; offset >>= 1)
    {
      y += ICuda.__shfl_down_sync("0xffffffff", y, offset);
    }

    if (tid == 0)
    {
      ICuda.atomicAdd(__device__d[0], y);
    }
  }//End

  @Override
  public void main() {
    double[] ha = {};//length:100000
    for(int x = 0; x < N ; ++x) {
      ha[x] = 1.23;
    }

    ICuda.cudaMemcpyToSymbol("d_x","ha",ICuda.sizeof("double")*N);
    final int block_size = 128;
    final int grid_size =  (N + block_size - 1) / block_size;
    __global__reduce();
    ICuda.cudaMemcpyFromSymbol("ha","d",ICuda.sizeof("double")*2);
    System.out.printf("c=%f\n",ha[0]);
  }//End
}
```
转义的C程序
```c
#include<stdio.h>
//现在开始全局内存书写
__device__  double d_x[100000] =  {};//length:100000
__device__  double d[2] =  {};//length:2
//现在开始全局变量书写
     const int N = 100000;//global
__global__ void reduce() {
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int n = bid * blockDim.x + tid;
__shared__ double __shared__s_y[128];
        if (n < N) {
            __shared__s_y[tid] = __ldg(&d_x[n]);
        }
        __syncthreads();

        for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
        {
            if (tid < offset)
            {
                __shared__s_y[tid] += __shared__s_y[tid + offset];
            }
            __syncthreads();
        }

        double y = __shared__s_y[tid];
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            y += __shfl_down_sync(0xffffffff, y, offset);
        }

        if (tid == 0)
        {
            atomicAdd(&d[0], y);
        }
}
int main(int argc, char* argv[]) {
double ha[100000] =  {};//length:100000
        for(int x = 0; x < N ; ++x) {
            ha[x] = 1.23;
        }

        cudaMemcpyToSymbol(d_x,ha,sizeof(double)*N);
        const int block_size = 128;
        const int grid_size =  (N + block_size - 1) / block_size;
        reduce<<<grid_size,block_size>>>();
        cudaMemcpyFromSymbol(ha,d,sizeof(double)*2);
        printf("c=%f\n",ha[0]);
}
```

### 线程束测试

Java源码:

```java
public class TestWarp implements ICuda {
  public final int WIDTH = 8;//global
  public final int N = 32;//global

  public void __global__hello() {//Start
    int tid = threadIdxx;
    int lane_id = tid % WIDTH;

    if (tid == 0) System.out.print("threadIdx.x: ");
    System.out.printf("%2d ", tid);
    if (tid == 0) System.out.print("\n");

    if (tid == 0) System.out.print("lane_id:     ");
    System.out.printf("%2d ", lane_id);
    if (tid == 0) System.out.print("\n");

    unsigned mask1 = ICuda.__ballot_sync("0xffffffff", tid > 0);
    unsigned mask2 = ICuda.__ballot_sync("0xffffffff", tid == 0);
    if (tid == 0) System.out.printf("FULL_MASK = %x\n", "0xffffffff");
    if (tid == 1) System.out.printf("mask1     = %x\n", mask1);
    if (tid == 0) System.out.printf("mask2     = %x\n", mask2);

    int result = ICuda.__all_sync("0xffffffff", tid);
    if (tid == 0) System.out.printf("all_sync (FULL_MASK): %d\n", result);

    int value = ICuda.__shfl_sync("0xffffffff", tid, 2, WIDTH);
    if (tid == 0) System.out.print("shfl:      ");
    System.out.printf("%2d ", value);
    if (tid == 0) System.out.print("\n");

    value = ICuda.__shfl_up_sync("0xffffffff", tid, 1, WIDTH);
    if (tid == 0) System.out.print("shfl_up:   ");
    System.out.printf("%2d ", value);
    if (tid == 0) System.out.print("\n");

    value = ICuda.__shfl_down_sync("0xffffffff", tid, 1, WIDTH);
    if (tid == 0) System.out.print("shfl_down: ");
    System.out.printf("%2d ", value);
    if (tid == 0) System.out.print("\n");

    value = ICuda.__shfl_xor_sync("0xffffffff", tid, 1, WIDTH);
    if (tid == 0) System.out.print("shfl_xor:  ");
    System.out.printf("%2d ", value);
    if (tid == 0) System.out.print("\n");
  }//End

  @Override
  public void main() {

    final int block_size = 16;
    final int grid_size = 1;

    __global__hello();

  }//End
}

```
转义的C程序:
```c
#include<stdio.h>
//现在开始全局内存书写
//现在开始全局变量书写
     const int WIDTH = 8;//global
     const int N = 32;//global
__global__ void hello() {
        int tid = threadIdx.x;
        int lane_id = tid % WIDTH;

        if (tid == 0) printf("threadIdx.x: ");
        printf("%2d ", tid);
        if (tid == 0) printf("\n");

        if (tid == 0) printf("lane_id:     ");
        printf("%2d ", lane_id);
        if (tid == 0) printf("\n");

        unsigned mask1 = __ballot_sync(0xffffffff, tid > 0);
        unsigned mask2 = __ballot_sync(0xffffffff, tid == 0);
        if (tid == 0) printf("FULL_MASK = %x\n", "0xffffffff");
        if (tid == 1) printf("mask1     = %x\n", mask1);
        if (tid == 0) printf("mask2     = %x\n", mask2);

        int result = __all_sync(0xffffffff, tid);
        if (tid == 0) printf("all_sync (FULL_MASK): %d\n", result);

        int value = __shfl_sync(0xffffffff, tid, 2, WIDTH);
        if (tid == 0) printf("shfl:      ");
        printf("%2d ", value);
        if (tid == 0) printf("\n");

        value = __shfl_up_sync(0xffffffff, tid, 1, WIDTH);
        if (tid == 0) printf("shfl_up:   ");
        printf("%2d ", value);
        if (tid == 0) printf("\n");

        value = __shfl_down_sync(0xffffffff, tid, 1, WIDTH);
        if (tid == 0) printf("shfl_down: ");
        printf("%2d ", value);
        if (tid == 0) printf("\n");

        value = __shfl_xor_sync(0xffffffff, tid, 1, WIDTH);
        if (tid == 0) printf("shfl_xor:  ");
        printf("%2d ", value);
        if (tid == 0) printf("\n");
}
int main(int argc, char* argv[]) {

        const int block_size = 16;
        const int grid_size = 1;

        hello<<<grid_size,block_size>>>();

}
```

## 关于ICuda

### 所有变量

```java
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
```
### 所有函数

```java
/**
 * 这些是Cuda和C语言提供的部分函数
 */
/**
 * 这些是Cuda和C语言提供的部分函数
 */
static <T> T malloc(int t) {return null;}
static <T> void cudaMalloc(String s, int t) {}
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
 * 运行时API函数
 */
static void cudaDeviceSynchronize() {};
static void cudaSetDevice(int id) {}
static void cudaGetDeviceProperties(cudaDeviceProp pop,int id) {};
/**
 * 线程数
 * @param mask
 * @param predicate
 * @return
 */


static int __any_sync(unsigned mask,int predicate){return 0;};
static int __all_sync(unsigned mask,int predicate){return 0;};
static int __any_sync(String mask,int predicate){return 0;};
static int __all_sync(String mask,int predicate){return 0;};
static unsigned __ballot_sync(String mask, int predicate){return null;};
static unsigned __ballot_sync(String mask,boolean pre){return null;};

static <T> T __shfl_sync(String mask, T v, int srcLane,int w){return null;};
static <T> T __shfl_up_sync(String mask, T v, int srcLane,int w){return null;};
static <T> T __shfl_down_sync(String mask, T v, int srcLane,int w){return null;};
static <T> T __shfl_xor_sync(String mask, T v, int srcLane,int w){return null;};
static <T> T __shfl_sync(String mask, T v, int srcLane){return null;};
static <T> T __shfl_up_sync(String mask, T v, int srcLane){return null;};
static <T> T __shfl_down_sync(String mask, T v, int srcLane){return null;};
static <T> T __shfl_xor_sync(String mask, T v, int srcLane){return null;};

static int sizeof(String s) {return 1;}
static int atoi(String c) {return 1;};
static long atol(String c) {return 1;};
static void printf(String format,Object... argv) {};
static void print(String format,Object... argv) {};
static <T> T __ldg(T j) {return null;};

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
```

# 更新日志:

### v1.2023.0422.06
- 支持使用动态共享内存
- 部分支持使用curand标准库
- 修复了指针输出的漏洞
- 优化了输出和转义
- 支持使用cuda流来加速程序
### v1.2023.0407.05
- 指针内存(包括显存)分配问题修复
- 支持编写使用多个核函数

### v1.2023.0405.04
- 核函数支持dim3类型的参数
- 支持了Cuda运行时API函数,尤其是显卡信息

### v1.2023.0401.03
- 第一个发行版本
- 新增了启动时的参数传递
- 完善了线程束内函数的缺陷
- 修复了全局内存两个放在一行出错的bug
### v1.2023.0325.02
- 新增了线程束内函数
- 改变了主函数格式,废除了beforekernel函数,afterkernel函数,新增了main主函数
- 新增了对核函数直接调用的方法
### v1.2023.0318.01
- 初代版本,提供了原子函数,共享,全局内存