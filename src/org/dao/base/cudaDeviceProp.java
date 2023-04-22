package org.dao.base;

public class cudaDeviceProp {
    public static String name;
    public static int major;
    public static int minor;
    public static int totalGlobalMem;
    public static int totalConstMem;
    public static int[] maxGridSize = new int[3];
    public static int[] maxThreadsDim = new int[3];
    public static int multiProcessorCount;
    public static int sharedMemPerBlock;
    public static int sharedMemPerMultiprocessor;
    public static int regsPerBlock;
    public static int maxThreadsPerBlock;
    public static int maxThreadsPerMultiProcessor;
    public int regsPerMultiprocessor;
}
