package org.dao.log;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.Date;

public class Logger {
    private static final SimpleDateFormat s = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss");
    private static final Date d = new Date();
    public static String Logfilepath = ".\\AstralPathCuda\\log\\log.txt";
    private static final PrintWriter pu;
    static {
        try {
            File log1 = new File(Logger.Logfilepath);
            if (!log1.exists()) {
                log1.getParentFile().mkdirs();
                log1.getParentFile().mkdir();
                log1.createNewFile();
            }
            pu = new PrintWriter(new FileWriter(log1, true));
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    public void writeINFO(String log) {
        pu.println("[INFO " + s.format(d)+ " ]"  + log);
    }
    public void writeERROR(String log) {
        pu.println("[ERROR " + s.format(d)+ " ]"  + log);
    }
    public void close() {
        pu.close();
    }
}
