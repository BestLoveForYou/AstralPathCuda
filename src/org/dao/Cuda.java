package org.dao;

import org.dao.log.Logger;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Cuda {
    private final String VERSION = "v1.2023.0719.11";
    private static String exename = "astralpathcuda.exe";
    private String parameter;
    private static boolean cnRAND = false;
    private static Map<Integer,Map> dev = new HashMap<>();
    private static Map<Integer,Map> gev = new HashMap<>();
    private static Map<Integer,String> glovarmap = new TreeMap<>();
    private static Map<Integer,String> afterkernelmap = new TreeMap<>();
    private static Map<Integer,String> main = new TreeMap<>();
    private static Map<String,String> Memorymap = new IdentityHashMap<>();
    private static boolean flag = true;
    private static boolean dflag = true;
    private static String filepath = "\\AstralPathCuda\\";
    private static final Logger log = new Logger();
    public int create(File file, boolean delete) throws ClassNotFoundException, IOException {
        log.writeINFO("-------------------------新纪录--------------------------");
        log.writeINFO("使用版本:" + VERSION);

        BufferedReader br = new BufferedReader(new FileReader(file));

        List<String> gpurun = new ArrayList<>();
        List<String> canshu = new ArrayList<>();

        List<String> dgpurun = new ArrayList<>();
        List<String> dcanshu = new ArrayList<>();
        List<String> dreturn = new ArrayList<>();
        String contentLine = br.readLine();
        int device=0;
        int global2 = 0;
        for (int x = 0 ;contentLine != null; x ++) {
            if (contentLine.contains("curandGenerator")) {
                cnRAND = true;
                log.writeINFO("使用cuRAND库");
            }
            if(contentLine.contains("__device__")) {
                if (contentLine.contains("(")) {
                    dgpurun.add("__device__" + contentLine.substring(contentLine.indexOf("__device__") + 10,contentLine.lastIndexOf(" {")));
                    dcanshu.add(contentLine.substring(contentLine.indexOf("("),contentLine.lastIndexOf(")")));
                    dreturn.add(contentLine.substring(contentLine.indexOf("public"),contentLine.lastIndexOf("__device__")));
                    Map<Integer,String> devicemap = new HashMap<>();
                    for (int y = 0 ;dflag; y ++) {
                        devicemap.put(y,contentLine);
                        contentLine = br.readLine();
                        if(contentLine.contains("//End")) {
                            devicemap.put(y + 1,"}");
                            break;
                        }
                    }
                    dev.put(device,devicemap);
                    device++;

                    dflag = true;
                }else {
                    Memorymap.put(contentLine.substring(contentLine.indexOf("public ") + 7, contentLine.indexOf("_") - 1), contentLine.substring(contentLine.indexOf("__device__")));
                }
            }
            if (contentLine.contains("__global__")) {
                gpurun.add(contentLine.substring(contentLine.indexOf("__global__") + 10,contentLine.lastIndexOf(" {")));
                canshu.add(contentLine.substring(contentLine.indexOf("("),contentLine.lastIndexOf(")")));
                Map<Integer,String> kernelmap = new TreeMap<>();
                for (int y = 0 ;flag; y ++) {
                    kernelmap.put(y,contentLine);
                    contentLine = br.readLine();
                    if(contentLine.contains("//End")) {
                        kernelmap.put(y + 1,"}");
                        break;
                    }
                }
                gev.put(global2,kernelmap);
                global2++;
            }
            if (contentLine.contains("//global")) {
                glovarmap.put(glovarmap.size() + 1,contentLine);
            }
            if (contentLine.contains(" main")) {
                for (int y = 0 ;flag; y ++) {
                    main.put(y,contentLine);
                    contentLine = br.readLine();
                    if(contentLine.contains("//End")) {
                        break;
                    }
                }
                flag = true;
            }
            contentLine = br.readLine();
            }
            boolean a = true;
            for (int x = 0; x < canshu.size();x++) {
                try {
                    String tep1 = canshu.get(x).substring(1);
                } catch (Exception e) {
                    a = false;
                }

            }
            log.writeINFO("核函数数:" + canshu.size());
            log.writeINFO("设备函数数:" + device);

            Date d = new Date();
            SimpleDateFormat format0 = new SimpleDateFormat("yyyyMMddHHmm");
            String time = format0.format(d.getTime());//这个就是把时间戳经过处理得到期望格式的时间

            File f = new File(".\\" + filepath + "\\temp\\" + time + ".cu");
            if(!f.getParentFile().exists()) {
                f.getParentFile().mkdirs();
            }
            if(!f.exists()) {
                f.createNewFile();
            }
            //创建初始文件
            PrintWriter pu = new PrintWriter(f);
            pu.println("#include<stdio.h>");
            pu.println("#include <cuda_runtime.h>");
            pu.println("using namespace std;");

            if (cnRAND) {
                pu.println("#include<curand.h>");
                this.parameter = this.parameter + " -lcurand";
                System.out.print("您使用了curand标准库，请自己将使用的文件添加进项目根目录(编码时已自动添加#include信息)");
            }


            writeMemory(pu);

        //以上是变量部分
        //设备函数
        try {
            if (a) {
                for (int x = 0; x < dgpurun.size();x++) {
                    pu.println("__device__ " + dreturn.get(x).replaceAll("public ","") + java2cuda(dgpurun.get(x) + "(GLOBAL)") + " {");
                    for (int y = 1; y < dev.get(x).size(); y++ ) {
                        if(!dev.get(x).get(y).toString().contains("for")) {
                            pu.println(java2cuda((String) dev.get(x).get(y)));
                        } else if(!dev.get(x).get(y).toString().contains("while")) {
                            pu.println(java2cuda((String) dev.get(x).get(y)));
                        } else {
                            if (dev.get(x).get(y).toString().replaceFirst(";","").contains(";")) {
                                String c[] = dev.get(x).get(y).toString().split(";");
                                for (int x1 = 0;x1 < c.length; x1 ++) {
                                    pu.println(java2cuda(c[x1] + ";"));
                                }
                            } else {
                                pu.println(java2cuda(dev.get(x).get(y).toString()));
                            }
                        }

                    }
                }
            }
        }catch (Exception e) {
        }
            //核函数
            if (a) {
                for (int x = 0; x < gpurun.size();x++) {
                    pu.println("__global__ void " + java2cuda(gpurun.get(x) + "(GLOBAL)") + " {");
                    for (int y = 1; y < gev.get(x).size(); y++ ) {
                        if(!gev.get(x).get(y).toString().contains("for")) {
                            pu.println(java2cuda((String) gev.get(x).get(y)));
                        } else if(!gev.get(x).get(y).toString().contains("while")) {
                            pu.println(java2cuda(gev.get(x).get(y).toString()));
                        } else {
                            if (gev.get(x).get(y).toString().replaceFirst(";", "").contains(";")) {
                                String c[] = gev.get(x).get(y).toString().split(";");
                                for (int x1 = 0; x1 < c.length; x1++) {
                                    pu.println(java2cuda(c[x1] + ";"));
                                }
                            } else {
                                pu.println(java2cuda((String) gev.get(x).get(y)));
                            }
                        }
                    }
                }
            }



            writeMain(pu);
            //主函数完成
            pu.close();
            File f1 = new File("");
            System.out.println("开始nvcc编译");

            System.out.println(f1.getAbsolutePath() + filepath + "\\temp\\" + time + ".cu");
            if(this.nvcc(f1.getAbsolutePath() + filepath + "\\temp\\" + time + ".cu") == 1) {
                System.out.println("编译成功");
            } else {
                System.err.println("编译失败");
            }
            br.close();
            if(delete) {
                f.delete();
            }
            log.close();


        return 1;
    }
    public static String java2cuda(String java) {
        String log1 = java;
        long start = System.currentTimeMillis();
        /**
        if (java.contains("__device__")) {
            String t[] = java.split("__device__");
            if((t[1].indexOf("[") - t[1].indexOf("(")) > 0) {

            }else {
                java = java.replaceAll("__device__","");
            }
        }*/
        java = java.replaceAll("__device__","");
        java = java.replaceAll("public","__device__");
        if(java.contains("__global__")) {
            if (java.contains("<<<")) {
                String[] t = java.split("\\(");
                String[] t2 = java.split("<<<");
                t2[1] = t2[1].substring(0,t2[1].indexOf(">>>"));
                java = t[0] + "<<<" + t2[1] + ">>>(" +t[1];
                java = java.replaceAll("__global__", "").replaceAll("\\$", "");
            } else {
                String[] t = java.split("\\(");
                java = t[0] + "<<<grid_size,block_size>>>(" + t[1];
                java = java.replaceAll("__global__", "").replaceAll("\\$", "");
            }
        }
        if (java.contains("__shfl_")) {
            java = java.replaceAll("\"","");
        }
        if (java.contains("__ballot_")) {
            java = java.replaceAll("\"","");
        }
        if (java.contains("__all_")) {
            java = java.replaceAll("\"","");
        }
        if (java.contains("__any_")) {
            java = java.replaceAll("\"","");
        }
        if(java.contains("atomic")) {
            String[] t = java.split("atomic");
            String[] t2 = t[1].split("\\(");
            java = t[0] + "atomic" + t2[0] + "(&" + t2[1];
        }
        if(java.contains("ldg")) {
            String[] t = java.split("ldg");
            String[] t2 = t[1].split("\\(");
            java = t[0] + "ldg" + t2[0] + "(&" + t2[1];
        }
        if(java.contains("cudaStreamCreate")) {
            String[] t = java.split("cudaStreamCreate");
            String[] t2 = t[1].split("\\(");
            java = t[0] + "cudaStreamCreate" + t2[0] + "(&" + t2[1];
        }
        if(java.contains("cudaGetSymbolAddress")) {
            String[] t = java.split("cudaGetSymbolAddress");
            String[] t2 = t[1].split("\\(");
            java = t[0] + "cudaGetSymbolAddress" + t2[0] + "((void**)&" + t2[1];
        }
        if(java.contains("curandCreateGenerator")) {
            String[] t = java.split("curandCreateGenerator");
            String[] t2 = t[1].split("\\(");
            java = t[0] + "curandCreateGenerator" + t2[0] + "(&" + t2[1];
        }
        if (java.contains("new int[")) {
            String[] t = java.split("new int");
            String temp = t[0].split(" = ")[0].replaceAll("\\[\\]","");
            java = temp + t[1].split("]")[0]  + "] = {};";
        }
        if (java.contains("new float[")) {
            String[] t = java.split("new float");
            String temp = t[0].split(" = ")[0].replaceAll("\\[\\]","");
            java = temp + t[1].split("]")[0]  + "] = {};";
        }
        if (java.contains("new double[")) {
            String[] t = java.split("new double");
            String temp = t[0].split(" = ")[0].replaceAll("\\[\\]","");
            java = temp + t[1].split("]")[0]  + "] = {};";
        }
        if (java.contains("new long[")) {
            String[] t = java.split("new long");
            String temp = t[0].split(" = ")[0].replaceAll("\\[\\]","");
            java = temp + t[1].split("]")[0]  + "] = {};";
        }
        if(java.contains("shared") ) {
            if (java.contains("= {")) {
                if (java.contains("extern")) {
                    java = java.replaceAll("\\[]", "");
                    String[] t = java.split("\\s+");
                    java = "extern __shared__" + " " + t[1] + " " + t[2] + "[];";
                } else {
                    String temp = java.substring(java.indexOf("length:") + 7);
                    temp = temp.replaceAll("length", "");
                    java = java.replaceAll("\\[]", "");
                    String[] t = java.split("\\s+");
                    java = "__shared__" + " " + t[1] + " " + t[2] + "[" + temp + "];";
                }
            }

        } else
        if(java.contains("length")) {
            if (!java.contains("$")) {
                java = " " +java;
                if (java.contains("=")) {
                    String temp = java.substring(java.indexOf("length:"));
                    temp= temp.replaceAll("length:","");
                    java =java.replaceAll("\\[]","");
                    String[] t = java.split("\\s+");
                    String[] t2 = java.split("=");
                    try {
                        java = t[1] + " " +t[2] + "[" + temp + "]" + " = " + t2[1];
                    } catch (Exception e) {
                    }
                } else {
                    String[] ts = java.split(";");
                    java = ts[0] + " = {}" + ";" + ts[1];
                    String temp = java.substring(java.indexOf("length:"));
                    temp= temp.replaceAll("length:","");
                    java =java.replaceAll("\\[]","");
                    String[] t = java.split("\\s+");
                    String[] t2 = java.split("=");
                    try {
                        java = t[1] + " " +t[2] + "[" + temp + "]" + " = " + t2[1];
                        java = java.replaceAll(" = \\{}","");
                    } catch (Exception e) {
                    }
                }
            }
        }



        if (java.contains("malloc(")) {
            if(java.contains("(double")) {
            }else if(java.contains("(int")) {
            }else if(java.contains("(long")) {
            }else if(java.contains("(float")) {
            }else {
                if(java.contains("double[]")) {
                    java = java.split("malloc")[0] + "(double *) malloc" +  java.split("malloc")[1];
                }else if(java.contains("int[]")) {
                    java = java.split("malloc")[0] + "(int *) malloc" +  java.split("malloc")[1];
                }else if(java.contains("long[]")) {
                    java = java.split("malloc")[0] + "(long *) malloc" +  java.split("malloc")[1];
                }else if(java.contains("float[]")) {
                    java = java.split("malloc")[0] + "(float *) malloc" +  java.split("malloc")[1];
                }
            }
            java = java.replaceAll("\"","");
        }
        if (java.contains("calloc(")) {
            java = java.replaceAll("\"","");
        }
        if(java.contains("cudaMemcpy")) {
            java = java.replaceAll("\"","");
        }
        if(java.contains("curandGenerate")) {
            java = java.replaceAll("\"","");
        }

        if(java.contains("cudaMallocHost")) {
            String[] t = java.split("cudaMallocHost");
            java = t[0] + "cudaMallocHost((void **)&" + t[1].replaceFirst("\\$","").replaceFirst("\\(","");
            java = java.replaceAll("\"","");
        } else if(java.contains("cudaMalloc")) {
            String[] t = java.split("cudaMalloc");
            java = t[0] + "cudaMalloc((void **)&" + t[1].replaceFirst("\\$","").replaceFirst("\\(","");
            java = java.replaceAll("\"","");
        }

        if (java.contains("$")) {
            if (!java.contains("print")) {
                java = java.replaceAll("\"","");
            }

            java = java.replaceFirst("\\[]","");
            if(java.contains(",")) {
                if (!java.contains("ICuda")) {
                    if(!java.contains(";")) {
                        java = java.replaceAll("\\[]","");
                    }
                }
            }else{
                if (java.contains("[]")) {
                    String[] a = java.split("\\[]");
                    java = a[0] + "$" + a[1];
                    if (java.contains("(GLOBAL)")) {java = a[0] + a[1];}
                }
            }
            if (java.contains("cudaMalloc")) {
                String[] t = java.split("cudaMalloc\\(");
                java = t[0] + "cudaMalloc((void **)&" + t[1].replaceFirst("\\$","");
                java = java.replaceAll("\"","");
            } else if(java.contains("__global__")) {

            } else if(java.contains("malloc")){}
            else if(java.contains("calloc")){}
            else if (java.contains(" = {")) {}
            else if(java.contains("(GLOBAL)")) {}

            else{
                java = java.replaceAll("\\$","");
            }
            if (java.contains(" = {")) {
                String[] temp = java.split(" = \\{");
                java = temp[0] + ";";
            }
            java = java.replaceAll("\\$","*");
        }
        if (java.contains("dim3")) {
            String[] t = java.split("=");
            String[] t2 = t[1].split("new dim3");
            java = t[0] + t2[1];
        }
        if (java.contains("new cudaStream_t")) {
            String[] t = java.split("=");
            String[] t2 = t[1].split("new cudaStream_t");
            java = t[0] + t2[1].replaceFirst("\\(\\)","");
        }
        if (java.contains("new cudaEvent_t")) {
            String[] t = java.split("=");
            String[] t2 = t[1].split("new cudaEvent_t");
            java = t[0] + t2[1].replaceFirst("\\(\\)","");
        }
        if (java.contains("new cudaDeviceProp")) {
            String[] t = java.split("=");
            String[] t2 = t[1].split("new cudaDeviceProp");
            java = t[0] + t2[1].replaceFirst("\\(\\)","");
        }
        if (java.contains("curandGenerator_t")) {
            String[] t = java.split("=");
            String[] t2 = t[1].split("new curandGenerator_t");
            java = t[0] + t2[1].replaceFirst("\\(\\)","");
        }
        if (java.contains("cudaGetDeviceProperties(")) {
            String[] a = java.split("cudaGetDeviceProperties\\(");
            java = a[0] +"cudaGetDeviceProperties(&" + a[1];
        }
        if (java.contains("cudaEventCreate(")) {
            String[] a = java.split("cudaEventCreate\\(");
            java = a[0] +"cudaEventCreate(&" + a[1];
        }
        if (java.contains("cudaEventElapsedTime(")) {
            String[] a = java.split("cudaEventElapsedTime\\(");
            java = a[0] +"cudaEventElapsedTime(&" + a[1];
        }
        if (java.contains("sizeof")) {
            java = java.replaceAll("\"","");
        }
        java = java.replaceAll("\\(GLOBAL\\)","");
        java = java.replaceAll("ICuda.","");
        //这里是核函数自带变量模块
        java = java.replaceAll("gridDimz","gridDim.z").replaceAll("gridDimy","gridDim.y").replaceAll("gridDimx","gridDim.x").replaceAll("blockDimz","blockDim.z").replaceAll("blockDimy","blockDim.y").replaceAll("blockIdxx","blockIdx.x").replaceAll("blockIdxy","blockIdx.y").replaceAll("blockIdxz","blockIdx.z").replaceAll("threadIdxx","threadIdx.x").replaceAll("threadIdxy","threadIdx.y").replaceAll("threadIdxz","threadIdx.z").replaceAll("blockDimx","blockDim.x");
        java = java.replaceAll("final","const").replaceAll("System.out.printf","printf");
        log.writeINFO("(Java2Cuda)耗时: " + (System.currentTimeMillis() - start) + " ms " + log1 + " -> " + java);
        return java;
    }
    public static int writeMemory(PrintWriter pu) {
        pu.println("//现在开始全局变量书写");
        Set<Map.Entry<Integer, String>> entrys2=glovarmap.entrySet();
        for(Map.Entry<Integer, String> entry:entrys2) {
            pu.println(java2cuda(entry.getValue().replaceAll("public","")));

        }
        Set<Map.Entry<String,String>> entrys1=Memorymap.entrySet();
        pu.println("//现在开始全局内存书写");
        for(Map.Entry<String,String> entry:entrys1) {

            if(entry.getKey().contains("[]")) {
                pu.println("__device__  " + java2cuda(entry.getKey() + " " + entry.getValue()));
            } else {
                pu.println("__device__ " + entry.getKey() +" " + entry.getValue().substring(entry.getValue().indexOf("__device__") + 10).replaceAll(";","") + ";");

            }
        }

        return 1;
    }
    public static int writeMain(PrintWriter pu) {
        pu.println("int main(int argc, char* argv[]) {");
        Set<Map.Entry<Integer, String>> entrys2= main.entrySet();
        for(Map.Entry<Integer, String> entry:entrys2) {
            if(entry.getValue().contains("public")) {

            }else {
                if(!entry.getValue().contains("for")) {
                    pu.println(java2cuda(entry.getValue()));
                } else if(!entry.getValue().contains("while")) {
                    pu.println(java2cuda(entry.getValue()));
                } else {
                    if (entry.getValue().replaceFirst(";","").contains(";")) {
                        String c[] = entry.getValue().split(";");
                        for (int x1 = 0;x1 < c.length; x1 ++) {
                            pu.println(java2cuda(c[x1] + ";"));
                        }
                    } else {
                        pu.println(java2cuda(entry.getValue()));
                    }
                }

            }

        }
        pu.println("}");

        return 1;
    }
    public static String cudainitialization() {
        String a = "false";
        try {
            Runtime rt = Runtime.getRuntime();
//            Process pr = rt.exec("cmd /c dir");
//            Process pr = rt.exec("D:/APP/Evernote/Evernote.exe");//open evernote program
            Process pr = rt.exec("nvcc --version") ;//open tim program
            BufferedReader input = new BufferedReader(new InputStreamReader(pr.getInputStream(),"GBK"));
            String line = null;
            while ((line = input.readLine())!=null){
                if (a.equals("false")) {
                    a = "";
                }
                a = a + "\n" + line;
            }
            int exitValue = pr.waitFor();
            System.out.println("Exited with error code "+exitValue);
        } catch (IOException e) {
            System.out.println(e);
            e.printStackTrace();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        return a;
    }
    public static String clinitialization() {
        String a = "false";
        try {
            Runtime rt = Runtime.getRuntime();
//            Process pr = rt.exec("cmd /c dir");
//            Process pr = rt.exec("D:/APP/Evernote/Evernote.exe");//open evernote program
            Process pr = rt.exec("cl") ;//open tim program
            BufferedReader input = new BufferedReader(new InputStreamReader(pr.getInputStream(),"GBK"));
            String line = null;
            while ((line = input.readLine())!=null){
                if (a.equals("false")) {
                    a = "";
                }
                a = a + "\n" + line;
            }
            int exitValue = pr.waitFor();
            System.out.println("Exited with error code "+exitValue);
        } catch (IOException e) {
            System.out.println(e);
            e.printStackTrace();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        return a;
    }

    public static String getExename() {
        return exename;
    }

    public static void setExename(String exename) {
        Cuda.exename = exename;
    }


    public static Map<Integer, String> getGlovarmap() {
        return glovarmap;
    }

    public static void setGlovarmap(Map<Integer, String> glovarmap) {
        Cuda.glovarmap = glovarmap;
    }

    public static Map<Integer, String> getAfterkernelmap() {
        return afterkernelmap;
    }

    public static void setAfterkernelmap(Map<Integer, String> afterkernelmap) {
        Cuda.afterkernelmap = afterkernelmap;
    }

    public static Map<Integer, String> getMain() {
        return main;
    }

    public static void setMain(Map<Integer, String> main) {
        Cuda.main = main;
    }

    public static Map<String, String> getMemorymap() {
        return Memorymap;
    }

    public static void setMemorymap(Map<String, String> memorymap) {
        Memorymap = memorymap;
    }

    public static boolean isFlag() {
        return flag;
    }

    public static void setFlag(boolean flag) {
        Cuda.flag = flag;
    }

    public static String getFilepath() {
        return filepath;
    }

    public static void setFilepath(String filepath) {
        Cuda.filepath = filepath;
    }
    public String getParameter() {
        return this.parameter;
    }

    public void setParameter(String parameter) {
        this.parameter = parameter;
    }

    public int nvcc(String filepath) {
        int a = 0;
        try {
            Runtime rt = Runtime.getRuntime();
//            Process pr = rt.exec("cmd /c dir");
//            Process pr = rt.exec("D:/APP/Evernote/Evernote.exe");//open evernote program
            Process pr = rt.exec("nvcc " + this.parameter+ " -o " + exename +" "+ filepath) ;//open tim program
            System.out.println("nvcc " + this.parameter+ " -o " + exename +" "+ filepath);
            BufferedReader input = new BufferedReader(new InputStreamReader(pr.getInputStream(),"GBK"));
            String line = null;
            String su = "";
            while ((line = input.readLine())!=null){
                System.out.println(line);
                su = su + line + "\n";
            }
            int exitValue = pr.waitFor();
            System.out.println("Exited with error code "+exitValue);
            if(exitValue == 1) {
                log.writeERROR(su);
            }
            if (exitValue == 0) {
                a = 1;
            }
        } catch (IOException e) {
            log.writeERROR(e.toString());
            System.out.println(e);
            e.printStackTrace();
            a = 0;
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        return a;
    }
    public static boolean deleteFile(String fileName) {
        File file = new File(fileName);
        // 如果文件路径只有单个文件
        if (file.exists() && file.isFile()) {
            if (file.delete()) {
                System.out.println("删除文件" + fileName + "成功！");
                return true;
            } else {
                System.out.println("删除文件" + fileName + "失败！");
                return false;
            }
        } else {
            System.out.println(fileName + "不存在！");
            return false;
        }
    }

    public static boolean deleteAllFile(String dir) {
        // 如果dir不以文件分隔符结尾，自动添加文件分隔符
//		if (!dir.endsWith(File.separator))
//			dir = dir + File.separator;
        File dirFile = new File(dir);
        // 如果dir对应的文件不存在，或者不是一个目录，则退出
        if ((!dirFile.exists()) || (!dirFile.isDirectory())) {
            System.out.println("删除文件夹失败：" + dir + "不存在！");
            return false;
        }
        boolean flag = true;
        // 删除文件夹中的所有文件包括子文件夹
        File[] files = dirFile.listFiles();
        for (int i = 0; i < files.length; i++) {
            // 删除子文件
            if (files[i].isFile()) {
                flag = deleteFile(files[i].getAbsolutePath());
                if (!flag)
                    break;
            }
            // 删除子文件夹
            else if (files[i].isDirectory()) {
                flag = deleteAllFile(files[i].getAbsolutePath());
                if (!flag)
                    break;
            }
        }
        if (!flag) {
            System.out.println("删除文件夹失败！");
            return false;
        }
        // 删除当前文件夹
        if (dirFile.delete()) {
            System.out.println("删除文件夹" + dir + "成功！");
            return true;
        } else {
            return false;
        }
    }

}
