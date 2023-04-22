package org.test;

import org.dao.ICuda;
import org.dao.base.unsigned;

public class TestWarp implements ICuda{
    public final int WIDTH = 8;//global
    public final int N = 32;//global
    public void __global__hello() {//Start
        int tid = threadIdxx;
        int lane_id = tid % WIDTH;

        if (tid == 0) System.out.printf("threadIdx.x: ");
        System.out.printf("%2d ", tid);
        if (tid == 0) System.out.printf("\n");

        if (tid == 0) System.out.printf("lane_id:     ");
        System.out.printf("%2d ", lane_id);
        if (tid == 0) System.out.printf("\n");

        unsigned mask1 = ICuda.__ballot_sync("0xffffffff", tid > 0);
        unsigned mask2 = ICuda.__ballot_sync("0xffffffff", tid == 0);
        if (tid == 0) System.out.printf("FULL_MASK = %x\n", "0xffffffff");
        if (tid == 1) System.out.printf("mask1     = %x\n", mask1);
        if (tid == 0) System.out.printf("mask2     = %x\n", mask2);

        int result = ICuda.__all_sync("0xffffffff", tid);
        if (tid == 0) System.out.printf("all_sync (FULL_MASK): %d\n", result);

        int value = ICuda.__shfl_sync("0xffffffff", tid, 2, WIDTH);
        if (tid == 0) System.out.printf("shfl:      ");
        System.out.printf("%2d ", value);
        if (tid == 0) System.out.printf("\n");

        value = ICuda.__shfl_up_sync("0xffffffff", tid, 1, WIDTH);
        if (tid == 0) System.out.printf("shfl_up:   ");
        System.out.printf("%2d ", value);
        if (tid == 0) System.out.printf("\n");

        value = ICuda.__shfl_down_sync("0xffffffff", tid, 1, WIDTH);
        if (tid == 0) System.out.printf("shfl_down: ");
        System.out.printf("%2d ", value);
        if (tid == 0) System.out.printf("\n");

        value = ICuda.__shfl_xor_sync("0xffffffff", tid, 1, WIDTH);
        if (tid == 0) System.out.printf("shfl_xor:  ");
        System.out.printf("%2d ", value);
        if (tid == 0) System.out.printf("\n");
    }//End
    @Override
    public void main() {

        final int block_size = 16;
        final int grid_size = 1;

        __global__hello();

    }//End
}
