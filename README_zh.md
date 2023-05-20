# AstralPathCuda 文档
2023-05-13 21:13:53 星期六

## 介绍
使用GPU完成科学计算不仅能提高计算的稳健性,更能提供一个无与伦比的速度,Java也是一种常用的CPU语言,开源平台为其提供了丰富的环境,但Cuda和Java的混合使用一直都是个麻烦事,毕竟两者本来都不该有交集,但GPU的高并发能力强大无比,CSDN上基本都是Java通过JNI调用CUDA程序,但麻烦无比,而使用JCuda也需要提供核函数的PTX文件,更加麻烦,还增加了开发难度,鉴于此前景,我开始了JavaGPU混合开发辅助包AstralPathCuda(以下简称APC)的编写.
在项目的最开始,只做了一些很简单的功能,比如线程同步函数,简单的核函数,线程索引...
随着项目的逐渐深入,开发的功能和难度也是越来越高,一些最开始的不适合定义也在现在被更改,到目前为止,使用APC已经可以开发出一个有模有样的复杂Cuda程序了,从动静态的全局内存;共享内存,到原子函数,再到现在已经基本完成的线程束内函数,都已经逐渐完善了.
之前在解决Java中使用指针这个麻烦时,花了我很长的一段时间和精力来想如何实现...但好在,主要的问题已经解决了.
接下来就是对APC使用的介绍了,若有不清楚的,可以在这篇文章下留言,也是为这个项目做出一些贡献吧.


~~这里的显卡都是指的是英伟达的显卡~~

## Cuda下载与安装
[CUDA安装教程（超详细）](https://blog.csdn.net/m0_45447650/article/details/123704930?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167222191016782427411200%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=167222191016782427411200&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-123704930-null-null.142%5Ev68%5Econtrol,201%5Ev4%5Eadd_ask,213%5Ev2%5Et3_esquery_v2&utm_term=cuda%E5%AE%89%E8%A3%85&spm=1018.2226.3001.4187)
## Github链接
[点击下载](https://github.com/BestLoveForYou/AstralPathCuda/)
## 下载
[点击下载](https://github.com/BestLoveForYou/AstralPathCuda/releases/)

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
        ICuda.cudaMemcpy("$a","$b",ICuda.sizeof("double")*20000,cudaMemcpyHostToDevice);
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

函数在最后都需要在方法**最后的"}"**后标注**//End**(注释形式)

### 主函数
```java
@Override
public void main() {

}//End
```
主函数是重载ICuda里的方法,必不可少
最后的下标//End也是不可缺少的

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
- 在```v1.2023.0422.06```版本之前需要加必须要加//End,在之后无需加了.

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

## CUDA流
使用多个流可以加速cuda操作,所以在APC中我提供了这类函数

创建非默认的cuda流如下(2个):
```java
        cudaStream_t stream_1 = new cudaStream_t();
        cudaStream_t stream_2 = new cudaStream_t();
        ICuda.cudaStreamCreate(stream_1);
        ICuda.cudaStreamCreate(stream_2);
```
使用cudaStreamDestroy销毁
```
ICuda.cudaStreamDestroy(stream_1);
ICuda.cudaStreamDestroy(stream_2);
```
同时支持了以下两个运行时函数来判断操作是否完成
```
static cudaError_t cudaStreamSynchronize(cudaStream_t c) {return new cudaError_t();}
static cudaError_t cudaStreamQuery(cudaStream_t c) {return new cudaError_t();}
```
值得注意的是,cudaStreamSynchronize会强制阻塞主机而cudaStreamQuery不会,它只会检查CUDA流中的所有操作是否都执行完成

**在主函数中调用核函数**
```java
__global__test();//tags:<<<grid_size,block_size,0,stream_1>>>
__global__test();//tags:<<<grid_size,block_size,0,stream_2>>>
```
在注释中标记使用的流名称

---
**不可分页内存的分配**
```java
static <T> void cudaMallocHost(T s, int t) {}
static <T> void cudaHostAlloc(T s, int t,T flags) {}
```
**异步数据传递函数**
```java
static <T> T cudaMemcpyAsync (String dst,String src,T count,cudaMemcpyKind kind,cudaStream_t stream) {return null;}
```
可以看到区别在于最后多了一个使用什么流的参数.

**不可分页内存指针的销毁**
```java
static void cudaFreeHost(Object... o) {}
```
与正常的差别并不大

实例:
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

以上实例中创建了d_x1与d_y1两个不可分页内存的指针,并使用异步传递向d_x1了h_x的数据,并在最后销毁了指针.

---
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

---

# 更新日志:

### v1.2023.0513.08
- 优化了cudaMemcpy使用
- 提供了用于CUDA流异步传输函数和不可分页内存的函数

### v1.2023.0430.07
- 此版本是一个过渡版本!
- 支持核函数无需使用//End标签完成读取
- 修复了curand库的部分Bug
- 支持了new创建数组的方式

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