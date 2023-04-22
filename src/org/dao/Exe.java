package org.dao;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class Exe {
    public static String runexe(String filepath,Object... objects) {
        String out = "";
        try {
            Runtime rt = Runtime.getRuntime();
//            Process pr = rt.exec("cmd /c dir");
//            Process pr = rt.exec("D:/APP/Evernote/Evernote.exe");//open evernote program
            String a = "";
            for (int x = 0 ;x < objects.length;x ++) {
                a = a + objects[x];
            }
            Process pr = rt.exec(filepath + " " + a) ;//open tim program
            BufferedReader input = new BufferedReader(new InputStreamReader(pr.getInputStream(),"GBK"));
            String line = null;
            while ((line = input.readLine())!=null){
                out = out+ "\n" + line;
            }
            int exitValue = pr.waitFor();
            System.out.println("Exited with error code "+exitValue);
        } catch (IOException e) {
            System.out.println(e.toString());
            e.printStackTrace();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        return out;
    }
}
