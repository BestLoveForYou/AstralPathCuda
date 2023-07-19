AstralPathCuda Documentation
2023-07-19, Wednesday

## Introduction
The **AstralPathCuda** project is a toolkit for GPU development in Java, developed by AstralPath Studio. It provides extensive functionality for CUDA development in Java and is particularly useful for scientific computing. With AstralPathCuda, Java programmers can develop complete CUDA programs quickly without the need for C language knowledge.

### Author's Note
Using GPUs for scientific computing not only improves computational robustness but also provides unparalleled speed. Java is a widely used CPU language, and the open-source platform provides a rich environment for it. However, combining CUDA and Java has always been a challenging task since the two were not originally meant to intersect. However, the GPU's powerful parallel computing capabilities are well-known. While there are many examples of Java programs invoking CUDA programs through JNI on CSDN, it can be quite troublesome. Using JCuda also requires providing PTX files for kernel functions, which further complicates development. Considering this landscape, I started developing a Java-GPU hybrid development toolkit called AstralPathCuda (APC).

In the early stages of the project, only some basic functionalities were implemented, such as thread synchronization functions, simple kernel functions, and thread indices. As the project delved deeper, the developed features and complexity increased. Some initial definitions that were not suitable were modified along the way. Up to now, using APC, it is possible to develop complex CUDA programs that have a proper structure. From dynamic and static global memory, shared memory, to atomic functions, and now the mostly completed warp functions, everything has gradually improved.

I spent a long time and a lot of effort figuring out how to solve the trouble of using pointers in Java. Thankfully, most of the major problems have been resolved. Now, it's time to introduce the usage of APC. If there's anything unclear, feel free to leave a comment below and contribute to this project.

~~The term "graphics card" mentioned here specifically refers to NVIDIA graphics cards.~~

## CUDA Download and Installation
[CUDA Installation Guide (Detailed)](https://blog.csdn.net/m0_45447650/article/details/123704930?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167222191016782427411200%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=167222191016782427411200&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-123704930-null-null.142%5Ev68%5Econtrol,201%5Ev4%5Eadd_ask,213%5Ev2%5Et3_esquery_v2&utm_term=cuda%E5%AE%89%E8%A3%85&spm=1018.2226.3001.4187)

## Github Links
[Click to Download](https://github.com/BestLoveForYou/AstralPathCuda/)

## Download
[Click to Download](https://github.com/BestLoveForYou/AstralPathCuda/releases/)

## Core
In CUDA, a CUDA program is written in C/C++, while APC is developed using Java. Therefore, there are some differences in the low-level language usage. To address this, I have implemented a solution using "commentary assistance + APC autofill" in actual coding, which provides convenience and flexibility to the users.
The basic unit in APC development is a **class**. After importing the AstralPathCuda development package, this class needs to use the ICuda interface, which provides CUDA functions, variables, and some C functions.

## Thread Indices
The following four classes are provided: threadIdx, blockIdx, blockDimx, and gridDimx. You can directly use variables like `threadIdx.x` (CUDA register variable that can be used directly). Additionally, the earlier versions provided variables like `threadIdxx`, `threadIdxy`, etc., in the ICuda interface, which can also be used normally.

### Format Output
You can directly use the format output provided by Java:
```java
System.out.printf(String format, Object... args);
```
Alternatively, you can use the `printf(String format, Object... args)` function provided by ICuda.

## Arrays
Arrays are a special type, as the array definition in Java is quite different from that in C. Therefore, a prompt needs to be set. The array definition is as follows:
For example, an integer array named `a` with a length of 3:
```java
int[] a = {1, 2, 3}; // length: 3
```
The length needs to be indicated after the array.

## Pointers
The `*` for pointers will be represented as `$`. For example:
```java
double[] $a = (double[]) ICuda.malloc(ICuda.sizeof("double") * 2);
```
will be translated as:
```c
double *a = (double *) malloc(sizeof(double) * 2);
```
This is a double pointer with a length of 2.

## Global Memory

### Static Global Memory
Definition:
```java
public int[] __device__d_x; // length: 128
```
Example: An integer variable `d_x` with a length of 128, an array.
In a kernel function, it can be called as:
```java
int x = __device__d_x[1];
```

To pass data to global memory in the main function, for example:
```java
double[] ha = {}; // length: 128
for (int x = 0; x < N; ++x) {
    ha[x] = 1.23;
}
ICuda.cudaMemcpyToSymbol("d_x", "ha", ICuda.sizeof("double") * 128);
```
will be translated as:
```c
double ha[128] = {}; // length: 128
for (int x = 0; x < N; ++x) {
    ha[x] = 1.23;
}
cudaMemcpyToSymbol(d_x, ha, sizeof(double) * 128);
```

Here, it can be observed that the parameters in global memory are enclosed in quotation marks, which is essential.

`cudaMemcpyFromSymbol` follows the same usage.

```java
static cudaError_t cudaMemcpyToSymbol(final String symbol, final String src, int count) { return null; }
static cudaError_t cudaMemcpyFromSymbol(final String src, final String symbol, int count) { return null; }
```
Note:
- The kernel function uses the name `__device__d_x` when calling the kernel function. However, in methods like `cudaMemcpyFromSymbol`, the name used is `d_x`, not `__device__d_x`.
- Supports caching acceleration using `__ldg()`.

### Dynamic Global Memory
Example:
```java
double[] $a = {};
ICuda.cudaMalloc("$a", ICuda.sizeof("double") * 2);
```
With an empty pointer, memory is allocated on the GPU with a size of 2. Let's take a look at the dynamic global memory transfer next.

```java
double[] $b = (double[]) ICuda.malloc(ICuda.sizeof("double")*20000);
        double[] $a = {};
        ICuda.cudaMalloc("$a",ICuda.sizeof("double")*20000);
        ICuda.cudaMemcpy("$a","$b",ICuda.sizeof("double")*20000,cudaMemcpyHostToDevice);
```

Similarly, dynamic global memory visibility is the same as in CUDA.

## Shared Memory
### Static Shared Memory

Define a shared array of type double with a length of 128:
```java
double[] __shared__s_y = {};//length:128
```
In the kernel function, this array can be accessed using the name `__shared__s_y`, instead of the traditional `s_y` used for global memory.

### Dynamic Shared Memory
```java
double[] __shared__s_y = {};//extern
```
Next, in the kernel function, set:

## Functions

All functions should have the comment `//End` at the end of the method.

### Main Function
```java
@Override
public void main() {

}//End
```
The main function is an override of the method in ICuda and is essential. The `//End` marker at the end is also necessary.

The main function comes with two variables: `int argc` and `char* argv[]`, which are the command-line arguments and can be used directly.

In the default invocation of the main function, the Java invocation of the kernel function (ordinary method call) will be converted to a CUDA invocation.

For example:

If a kernel function is converted without any comments, like this:
```java
__global__reduce();
```
It will be translated to:
```c
reduce<<<grid_size,block_size>>>();
```
This means that the `grid_size` and `block_size` should be defined above this function, for example:
```java
final int block_size = 128;
final int grid_size =  (N + block_size - 1) / block_size;
        __global__reduce();
```
If there is a comment indicating the configuration, such as:
```
__global__reduce();//tags:<<<block_size,grid_size,128>>>
```
It will be translated accordingly.

If there are arguments, they will also be included:
```
__global__reduce<<<block_size,grid_size,128>>>();//tags:<<<block_size,grid_size,128>>>
```

You can also use the more complex `dim3` type:
```java
dim3 grid_size = new dim3(2,2,2);
        dim3 block_size = new dim3(3,3,3)
        __global__reduce();
```

It is not recommended to change or reuse any of the variable names used in the annotations.

### Kernel Function

Kernel function definition:
```java
public void __global__kernelFunctionName() {
// omitted
        }//End
```
- The `//End` comment at the end is necessary. Some versions may support omitting `//End`, but it has been proven to greatly reduce generality and program correctness. Therefore, starting from version `v1.2023.0719`, this kind of omission is completely banned.

### Device Functions

#### Device Functions with Return Value
```java
public ReturnType __device__kernelFunctionName() {

}//End
```

#### Device Functions without Return Value (using pointers)
```java
public void __device__kernelFunctionName() {

}//End
```
~Reference functions are currently not supported.~

### Atomic Functions

Supported atomic functions include:
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

The usage of all atomic functions remains the same and can be used normally without any changes.

### Warp Functions

Supports thread-level synchronization function `__syncwarp()`.

Supports warp voting functions:
```java
static int __any_sync(unsigned mask,int predicate){return 0;};
static int __all_sync(unsigned mask,int predicate){return 0;};
static int __any_sync(String mask,int predicate){return 0;};
static int __all_sync(String mask,int predicate){return 0;};
static unsigned __ballot_sync(String mask, int predicate){return null;};
static unsigned __ballot_sync(String mask,boolean pre){return null;};
```
This includes both `unsigned` type and `String` type for convenience in development.

Supports warp shuffle functions:
```java
static <T> T __shfl_sync(String mask, T v, int srcLane,int w){return null;};
static <T> T __shfl_up_sync(String mask, T v, int srcLane,int w){return null;};
static <T> T __shfl_down_sync(String mask, T v, int srcLane,int w){return null;};
static <T> T __shfl_xor_sync(String mask, T v, int srcLane,int w){return null;};
```
The above functions have an additional parameter `w`.

Below are the versions without the `w` parameter:
```java
static <T> T __shfl_sync(String mask, T v, int srcLane){return null;};
static <T> T __shfl_up_sync(String mask, T v, int srcLane){return null;};
static <T> T __shfl_down_sync(String mask, T v, int srcLane){return null;};
static <T> T __shfl_xor_sync(String mask, T v, int srcLane){return null;};
```

## Runtime API Functions

```java
static void cudaDeviceSynchronize() {};
static void cudaSetDevice(int id) {}
static void cudaGetDeviceProperties(cudaDeviceProp pop,int id) {};
```
Additionally, you can view the device properties using the API function `cudaDeviceProp`:
```cudaDeviceProp class```

Here is an example of usage:
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
This will output the information of the graphics card.

-----

## CUDA Streams
Using multiple streams can accelerate CUDA operations, so in APC, I provide functions for this purpose.

To create non-default CUDA streams (2 streams):
```java
        cudaStream_t stream_1 = new cudaStream_t();
        cudaStream_t stream_2 = new cudaStream_t();
        ICuda.cudaStreamCreate(stream_1);
        ICuda.cudaStreamCreate(stream_2);
```
Use `cudaStreamDestroy` to destroy the streams:
```
ICuda.cudaStreamDestroy(stream_1);
ICuda.cudaStreamDestroy(stream_2);
```
Two runtime functions are also supported to check if the operations have completed:
```
static cudaError_t cudaStreamSynchronize(cudaStream_t c) {return new cudaError_t();}
static cudaError_t cudaStreamQuery(cudaStream_t c) {return new cudaError_t();}
```
Note that `cudaStreamSynchronize` will force the host to block, while `cudaStreamQuery` will not.

```java
static <T> void cudaMallocHost(T s, int t) {}
static <T> void cudaHostAlloc(T s, int t,T flags) {}
```
**Asynchronous Data Transfer Functions**
```java
static <T> T cudaMemcpyAsync (String dst,String src,T count,cudaMemcpyKind kind,cudaStream_t stream) {return null;}
```
These functions are similar to their counterparts in CUDA, with an additional parameter specifying the stream to use.

**Freeing Pinned Memory Pointers**
```java
static void cudaFreeHost(Object... o) {}
```
The usage is similar to the regular `cudaFree` function.

Example:
```java
int M = ICuda.sizeof("int")*N;
        int[] $h_x = ICuda.malloc(M);
        for (int x = 0;x < N;x ++) {
			$h_x[x] = 1;
        }
        int[] $d_x1 = {};
        int[] $d_y1 = {};
        ICuda.cudaMallocHost("d_x1",M);
ICuda.cudaMallocHost("d_y1",M);
       ICuda.cudaMemcpyAsync("d_x1","h_x",M,cudaMemcpyHostToDevice,stream_1);
       ICuda.cudaMemcpyAsync("d_x1","h_x",M,cudaMemcpyHostToDevice,stream_2);
        ICuda.free($h_x);
        ICuda.cudaFreeHost($d_x1);
        ICuda.cudaFreeHost($d_y1);
```

In the above example, pinned memory pointers `d_x1` and `d_y1` are created, and the data from `h_x` is asynchronously transferred to `d_x1` using the specified streams. Finally, the pointers are freed.

---
## curand Standard Library
**All methods implemented on the Host are supported!**
### Host API Calls
- Algorithm Types
```java
static curndRngType_t CURAND_RNG_PSEUDO_DEFAULT = new curndRngType_t();
    static curndRngType_t CURAND_RNG_PSEUDO_XORWOW = new curndRngType_t();
    static curndRngType_t CURAND_RNG_PSEUDO_MRG19937 = new curndRngType_t();
    static curndRngType_t CURAND_RNG_PSEUDO_MRG32K3A = new curndRngType_t();
    static curndRngType_t CURAND_RNG_PSEUDO_PHILOX4_32_10 = new curndRngType_t();
    static curndRngType_t CURAND_RNG_PSEUDO_MT19937 = new curndRngType_t();
	// The above are pseudo-random number generators
    // The following are quasi-random number generators
    static curndRngType_t CURAND_RNG_QUASI_SOBOL32 = new curndRngType_t();
    static curndRngType_t CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = new curndRngType_t();
    static curndRngType_t CURAND_RNG_QUASI_SOBOL64 = new curndRngType_t();
    static curndRngType_t CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = new curndRngType_t();
```
- Creation:
```java
 static void curandCreateGenerator(curandGenerator_t c, curndRngType_t rng_type) {}//c is a pointer
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

- Seed

A 64-bit integer used to initialize the starting state of the pseudo-random number generator. The same seed often generates the same sequence of random numbers.

- Offset

The offset parameter is used to skip the beginning of the random number sequence. The first random number is taken from the offset-th position in the sequence. This ensures that the random numbers generated from the same sequence do not overlap when the same program is run multiple times. Offset is not available for the CURAND_RNG_PSEUDO_MTGP32 and CURAND_RNG_PSEUDO_MT19937 generators.

- Order

Used to select how results are ordered in global memory.
CURAND_ORDERING_PSEUDO_DEFAULT
CURAND_ORDERING_PSEUDO_LEGACY
CURAND_ORDERING_PSEUDO_BEST
CURAND_ORDERING_PSEUDO_SEEDED
CURAND_ORDERING_QUASI_DEFAULT (options for quasi-random numbers)
-
- Generation Functions
```java
static <T> void curandGenerateUniformDouble(curandGenerator_t c, T memory, int N) {}
    static <T> void curandGenerateNormalDouble(curandGenerator_t generator, T $gX, int N,double a,double b) {}
    static <T> void curandGenerateNormalDouble(curandGenerator_t generator, double[] $gX, int N,double a,double b) {}
    static <T> void curandGenerateNormalFloat(curandGenerator_t generator, float[] $gX, int N,float a,float b) {}
    static <T> void curandGenerateNormalInt(curandGenerator_t generator, int[] $gX, int N,int a,int b) {}
    static <T> void curandGenerateNormalLong(curandGenerator_t generator, long[] $gX, int N,long a,long b) {}
```

~~As for the Device API, it is still in progress~~

---

# Changelog:
### v1.2023.0719.11
- Abolished the option to omit the //End tag
- Fixed several usage vulnerabilities in device functions
- Improved comments and usage instructions for ICuda functions
- Added TestNew test demo
### v1.2023.0622.10
- Added a more flexible memory allocation method for pointers
- Fixed bugs in the sample program
### v1.2023.0617.09
- Fixed cudaMemcpy parameter bugs

### v1.2023.0513.08
- Optimized the usage of cudaMemcpy
- Provided support for CUDA stream asynchronous transfer functions and pinned memory

### v1.2023.0430.07
- This version is a transitional version!
- Support kernel functions without using the //End tag
- Fixed bugs in the curand library
- Supported new array creation method

### v1.2023.0422.06
- Supported dynamic shared memory
- Partially supported the curand standard library
- Fixed pointer output vulnerability
- Optimized output and translation
- Supported CUDA streams for program acceleration
### v1.2023.0407.05
- Fixed memory allocation issue for pointers (including device memory)
- Supported writing multiple kernel functions

### v1.2023.0405.04
- Supported dim3 type parameters for kernel functions
- Supported CUDA runtime API functions, especially for GPU information

### v1.2023.0401.03
- First official release
- Added support for atomic functions, shared memory, and global memory
- Improved the format of the main function, abolished the beforekernel and afterkernel functions, and introduced the main function
- Added support for direct invocation of kernel functions
### v1.2023.0325.02
- Initial version, provided atomic functions, shared memory, and global memory