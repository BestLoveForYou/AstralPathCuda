package org.test;

import org.dao.Cuda;
import org.dao.Exe;

import java.io.File;
import java.io.IOException;

import static org.dao.Cuda.clinitialization;
import static org.dao.Cuda.cudainitialization;

public class Main {
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        int a = 1;
        String f = "D:\\idea_astralpathtalk\\AstralPathCuda\\src\\org\\test\\ReduceSHFL.java";
        if (a == 0) {
            Cuda c= new Cuda();
            c.setParameter("-arch=sm_60");
            c.create(new File(f),false);
        } else {
            Exe e = new Exe();
            System.out.println(Exe.runexe("D:\\idea_astralpathtalk\\AstralPathCuda\\astralpathcuda.exe"));
        }


        /*

         */
    }
}
