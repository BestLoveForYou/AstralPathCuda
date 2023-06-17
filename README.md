# AstralPathCuda Document
Saturday, June 17, 2023 21:38:41

## Introduction
The **AstralPathCuda** project is a toolkit for GPU development in Java developed by AstralPath Studio. It has a rich set of features and can be used for CUDA development in most scenarios, especially in scientific computing. Using AstralPathCuda, Java programmers can quickly develop comprehensive CUDA programs without the need for a C language foundation.

### Author's Note
Using GPU for scientific computing can not only improve the robustness of calculations, but also provide unparalleled speed. Java is a commonly used CPU language, and the open-source platform provides a rich environment for it. However, the combination of Cuda and Java has always been a troublesome issue, as the two were not originally meant to intersect. However, the powerful concurrency of the GPU makes it a valuable asset. On CSDN, it is common to see Java calling CUDA programs through JNI, but this method is very cumbersome. Using JCuda also requires providing the PTX file for the kernel function, which is even more troublesome and increases development difficulty. Given this prospect, I started developing the JavaGPU hybrid development package AstralPathCuda (hereinafter referred to as APC).
At the beginning of the project, only some simple functions were implemented, such as thread synchronization functions, simple kernel functions, thread indexes, etc. As the project gradually deepened, the development functions and difficulties also increased. Some parts that were not suitable for definition at the beginning have also been modified now. So far, using APC can develop a complex Cuda program with a complete structure, from dynamic and static global memory, shared memory, atomic functions, to the now almost completed thread bundle function.
When solving the problem of using pointers in Java, it took me a long time and effort to think about how to implement it. But fortunately, the main problem has been solved.
Next, I will introduce how to use APC. If there is anything unclear, you can leave a message under this article, and also contribute to this project.
Note: The graphics cards mentioned here refer to Nvidia graphics cards.

## Cuda Download and Installation
[CUDA Installation Tutorial (Super Detailed)](https://blog.csdn.net/m0_45447650/article/details/123704930)
## Github Link
[Click to download](https://github.com/BestLoveForYou/AstralPathCuda/)
## Download
[Click to download](https://github.com/BestLoveForYou/AstralPathCuda/releases/)
## Core
In Cuda, a Cuda program is written in C/C++, while APC uses Java as its language. Therefore, there may be more or less low-level differences in the language. To solve this, my solution is `comment assistance + APC automatic filling`. In actual coding, users will feel the convenience and flexibility of this method.

In APC, the most basic unit in development is the `class`, which needs to use the `ICuda` interface after importing the `AstralPaCuda` development package. `ICuda` provides some Cuda functions, variables, and some C functions.

## Thread Indices
There are four classes available: ```threadIdx```, ```blockIdx```, ```blockDimx```,and ```gridDimx``` that can be directly used in the program. For example, variables like threadIdx.x can be used directly as register variables in CUDA.

In addition, variables like threadIdxx and threadIdxy, which were provided in earlier versions within ICuda, can also be used normally.

## Formatted Output
You can directly use the Java provided:
```java
System.out.printf(String format, Object ...args[]);
```
Alternatively, you can use the printf(String format, Object ...args[]) function provided in ICuda.

## Array
Arrays are a somewhat special entity. Because the way arrays are defined in Java is quite different from that in C, prompt symbols need to be set. 

Array definition is as follows:

For example, an integer array called "a" with a length of 3:

```java
int[] a = {1,2,3}; // length: 3
```

The length needs to be marked after the array.
## Pointers
The `*` symbol for pointers will appear as `$`.

For example:

```java
double[] $a = (double[]) ICuda.malloc(ICuda.sizeof("double")*2); 
```

will be translated to

```c
double *a = (double*) malloc(sizeof(double)*2);
```

This is a double pointer with a length of 2.

## Global Memory

### Static Global Memory

Definition:

```java
public int[] __device__d_x;//length:128
```

For example, an integer array called "d_x" with a length of 128, which is an array.

```java
public int __device__d;
```

For example, an integer called "d" which is not an array.

Calling in a kernel function:

```java
int x = __device__d_x[1];
```

Passing data to global memory in the main function:

```java
double[] ha = {};//length:128
for(int x = 0; x < N ; ++x) {
    ha[x] = 1.23;
}
ICuda.cudaMemcpyToSymbol("d_x","ha",ICuda.sizeof("double")*128);
```

This will be translated to:

```c
double ha[128] = {};//length:128
for(int x = 0; x < N ; ++x) {
    ha[x] = 1.23;
}
cudaMemcpyToSymbol(d_x,ha,sizeof(double)*128);
```

Here, it can be seen that the parameter in the global memory parameter has quotation marks, which is essential.

The usage of `cudaMemcpyFromSymbol` is the same.

Note:
- The name used for the kernel function in the kernel function call is `__device__d_x`, while the name used in methods like `cudaMemcpyFromSymbol` is `d_x`, not `__device__d_x`.
- Supports caching acceleration with `__ldg()`.

### Dynamic Global Memory

For example:

```java
double[] $a = {};
ICuda.cudaMalloc("$a",ICuda.sizeof("double")*2);
```

An empty pointer is used to allocate video memory with a size of 2. 

Next, let's take a look at the passing of dynamic global memory:

```java
double[] $b = (double[]) ICuda.malloc(ICuda.sizeof("double")*20000);
double[] $a = {};
ICuda.cudaMalloc("$a",ICuda.sizeof("double")*20000);
ICuda.cudaMemcpy("$a","$b",ICuda.sizeof("double")*20000,cudaMemcpyHostToDevice);
```

Similarly, the visibility of dynamic global memory is no different from that in cuda.

## Shared Memory

### Static Shared Memory

Define a shared array of type double with a length of 128:

```java
double[] __shared__s_y = {};//length:128
```

This array is called `__shared__s_y` in the kernel function, not `s_y` like in global memory.

### Dynamic Shared Memory

```java
double[] __shared__s_y = {};//extern
```

Next, it is set in the kernel function.

## Functions

Functions need to be marked with `//End` after the last "}" in the method.

### Main Function

```java
@Override
public void main() {
	
}//End
```

The main function is a method that is overloaded in ICuda and is essential. The final index `//End` is also indispensable.

The main function comes with two variables: `int argc` and `char* argv[]`, which are startup parameters and can be used directly.

In the default call of the main function, the Java call (ordinary method call) of the kernel function will be converted to a Cuda call. For example:

When the kernel function is translated, if there are no comments, such as:

```java
__global__reduce();
```

It will be translated to:

```c
reduce<<<grid_size,block_size>>>();
```

This also means that `grid_size` and `block_size` should be defined above this function, such as:

```java
final int block_size = 128;
final int grid_size =  (N + block_size - 1) / block_size;
__global__reduce();
```

If there is a comment prompt, such as:

```
__global__reduce();//tags:<<<block_size,grid_size,128>>>
```

It will be translated to:

If there are parameters, they will be brought along:

```
__global__reduce<<<block_size,grid_size,128>>>();//tags:<<<block_size,grid_size,128>>>
```

A more complex `dim3` type can also be used:

```java
dim3 grid_size = new dim3(2,2,2);
dim3 block_size = new dim3(3,3,3)
__global__reduce();
```

None of the variable names should be changed or reused.

### Kernel Function

Definition of kernel function:
```java
public void __global__ kernelFunctionName() {
// omitted
}// End
```
- Before version `v1.2023.0422.06`, it is necessary to add `// End`, but it is not necessary to add it after that.

### Device Function
```java
public any_return_type __device__ kernelFunctionName() {
    
}// End
```
*Currently, there are many defects in device functions, which need to be solved gradually.*

## Atomic Function

Supported:
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

All atomic functions can be used normally without modification.

## Thread Block

Supports synchronization function within thread block `__syncwarp()`.
Supports thread block voting functions:
```java
    static int __any_sync(unsigned mask,int predicate){return 0;};
    static int __all_sync(unsigned mask,int predicate){return 0;};
    static int __any_sync(String mask,int predicate){return 0;};
    static int __all_sync(String mask,int predicate){return 0;};
    static unsigned __ballot_sync(String mask, int predicate){return null;};
    static unsigned __ballot_sync(String mask,boolean pre){return null;};
```
Here, not only `unsigned` type is included, but also `String` type for development convenience.
Supports thread block shuffle functions:
```java
    static <T> T __shfl_sync(String mask, T v, int srcLane,int w){return null;};
    static <T> T __shfl_up_sync(String mask, T v, int srcLane,int w){return null;};
    static <T> T __shfl_down_sync(String mask, T v, int srcLane,int w){return null;};
    static <T> T __shfl_xor_sync(String mask, T v, int srcLane,int w){return null;};
```
The above functions have a parameter `w`.
Below are the functions without `w` parameter:
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
And API functions to view the device:
```cudaDeviceProp class```
Use it like this:
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
The output will display information about the graphics card.

## CUDA Stream
Using multiple streams can accelerate CUDA operations, so I have provided such functions in APC.

To create non-default CUDA streams, use the following code (for 2 streams):
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
Two runtime functions are supported to determine whether an operation has completed:
```
static cudaError_t cudaStreamSynchronize(cudaStream_t c) {return new cudaError_t();}
static cudaError_t cudaStreamQuery(cudaStream_t c) {return new cudaError_t();}
```
It is worth noting that `cudaStreamSynchronize` will force the host to block, while `cudaStreamQuery` will not. It only checks whether all operations in the CUDA stream have been completed.

**Calling kernel functions in the main function**
```java
__global__test();//tags:<<<grid_size,block_size,0,stream_1>>>
__global__test();//tags:<<<grid_size,block_size,0,stream_2>>>
```
Use the stream names in the comments to mark the streams used.

---
**Allocation of non-paged memory**
```java
static <T> void cudaMallocHost(T s, int t) {}
static <T> void cudaHostAlloc(T s, int t,T flags) {}
```
**Asynchronous data transfer function**
```java
static <T> T cudaMemcpyAsync (String dst,String src,T count,cudaMemcpyKind kind,cudaStream_t stream) {return null;}
```
As you can see, the difference is that there is an additional parameter for specifying which stream to use.

**Freeing non-paged memory pointers**
```java
static void cudaFreeHost(Object... o) {}
```
There is not much difference from normal memory freeing.

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

In the example above, two non-paged memory pointers, `d_x1` and `d_y1`, were created. The data in `h_x` was transferred asynchronously to `d_x1`, and the pointers were destroyed at the end.

---
## curand standard library
**All methods implemented on the Host are supported!**
### Host API calls
```java
static curndRngType_t CURAND_RNG_PSEUDO_DEFAULT = new curndRngType_t();
static curndRngType_t CURAND_RNG_PSEUDO_XORWOW = new curndRngType_t();
static curndRngType_t CURAND_RNG_PSEUDO_MPG19937 = new curndRngType_t();
static curndRngType_t CURAND_RNG_PSEUDO_MRG32K3A = new curndRngType_t();
static curndRngType_t CURAND_RNG_PSEUDO_PHILOX4_32_10 = new curndRngType_t();
static curndRngType_t CURAND_RNG_PSEUDO_MT19937 = new curndRngType_t();
// The above are for pseudo-random number generation
// The following are for quasi-random number generation
static curndRngType_t CURAND_RNG_QUASI_SOBOL32 = new curndRngType_t();
static curndRngType_t CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = new curndRngType_t();
static curndRngType_t CURAND_RNG_QUASI_SOBOL64 = new curndRngType_t();
static curndRngType_t CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = new curndRngType_t();
```
- Creation:
```java
static void curandCreateGenerator(curandGenerator_t c, curndRngType_t rng_type) {}// c is a pointer
```
- Generator Options:
```java
static curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t c, long seed) { return null; }
static curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, String seed) { return null; }
static curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned seed) { return null; }
static curandStatus_t curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator, int num_dimensions) { return null; }
static curandStatus_t curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator, unsigned num_dimensions) { return null; }
static curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator, unsigned offset) { return null; }
static curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator, long offset) { return null; }
```
-- Seed
The seed is a 64-bit integer used to initialize the starting state of the pseudo-random number generator. The same seed often generates the same sequence of random numbers.
-- Offset
The offset parameter is used to skip the beginning of the random number sequence, with the first random number being taken from the offset-th element of the sequence. This ensures that random numbers generated from the same sequence do not overlap when the same program is run multiple times. This is not available for the CURAND_RNG_PSEUDO_MTGP32 and CURAND_RNG_PSEUDO_MT19937 generators.
-- Order
Used to select how the results are sorted in global memory.
CURAND_ORDERING_PSEUDO_DEFAULT
CURAND_ORDERING_PSEUDO_LEGACY
CURAND_ORDERING_PSEUDO_BEST
CURAND_ORDERING_PSEUDO_SEEDED
CURAND_ORDERING_QUASI_DEFAULT (for quasi-random numbers)
- Generation Functions:
```java
static <T> void curandGenerateUniformDouble(curandGenerator_t c, T memory, int N) {}
static <T> void curandGenerateNormalDouble(curandGenerator_t generator, T $gX, int N,double a,double b) {}
static <T> void curandGenerateNormalDouble(curandGenerator_t generator, double[] $gX, int N,double a,double b) {}
static <T> void curandGenerateNormalFloat(curandGenerator_t generator, float[] $gX, int N,float a,float b) {}
static <T> void curandGenerateNormalInt(curandGenerator_t generator, int[] $gX, int N,int a,int b) {}
static <T> void curandGenerateNormalLong(curandGenerator_t generator, long[] $gX, int N,long a,long b) {}
```

~~As for the API on the Device, it is still in progress.~~

---

# Update Log:
### v1.2023.0617.09
- Fixed the bug in the parameter processing of "cudaMemcpy";
### v1.2023.0513.08
- Optimized the use of cudaMemcpy
- Provided functions for asynchronous transfer with CUDA streams and non-pageable memory

### v1.2023.0430.07
- This version is a transitional version!
- Supports kernel functions that do not require the use of the //End tag to complete reading
- Fixed some bugs in the curand library
- Supports creating arrays using the new method

### v1.2023.0422.06
- Supports the use of dynamic shared memory
- Partially supports the use of the curand standard library
- Fixed a bug with pointer output
- Optimized output and escape characters
- Supports using CUDA streams to speed up programs

### v1.2023.0407.05
- Fixed pointer memory (including video memory) allocation issues
- Supports writing programs using multiple kernel functions

### v1.2023.0405.04
- Kernel functions support dim3 type parameters
- Supports CUDA runtime API functions, especially for GPU information

### v1.2023.0401.03
- First release version
- Added parameter passing at startup
- Improved the defects of functions inside thread blocks
- Fixed a bug where two global memory variables on the same line caused an error

### v1.2023.0325.02
- Added functions inside thread blocks
- Changed the main function format, abolished the beforekernel function and afterkernel function, and added the main function
- Added a method for directly calling kernel functions

### v1.2023.0318.01
- Initial version, providing atomic functions, shared and global memory.

