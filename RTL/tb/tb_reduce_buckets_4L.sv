`timescale 1ns/1ps

module tb_reduce_buckets_4L;

    logic [255:0] B1_X;
    logic [255:0] B1_Y;
    logic [255:0] B1_Z;

    logic [255:0] B2_X;
    logic [255:0] B2_Y;
    logic [255:0] B2_Z;

    logic [255:0] B3_X;
    logic [255:0] B3_Y;
    logic [255:0] B3_Z;

    logic [255:0] OUT_X;
    logic [255:0] OUT_Y;
    logic [255:0] OUT_Z;

    reduce_buckets_4L dut (
        .i_B1_X(B1_X),
        .i_B1_Y(B1_Y),
        .i_B1_Z(B1_Z),

        .i_B2_X(B2_X),
        .i_B2_Y(B2_Y),
        .i_B2_Z(B2_Z),

        .i_B3_X(B3_X),
        .i_B3_Y(B3_Y),
        .i_B3_Z(B3_Z),

        .o_X(OUT_X),
        .o_Y(OUT_Y),
        .o_Z(OUT_Z)
    );


    task automatic check_no_x;
        begin
            if ((^OUT_X === 1'bx) || (^OUT_Y === 1'bx) || (^OUT_Z === 1'bx)) begin
                $display("[FAIL] Output contains X");
                $display("OUT_X = %h", OUT_X);
                $display("OUT_Y = %h", OUT_Y);
                $display("OUT_Z = %h", OUT_Z);
                $fatal;
            end else begin
                $display("[PASS] No X values");
                $display("OUT_X = %h", OUT_X);
                $display("OUT_Y = %h", OUT_Y);
                $display("OUT_Z = %h", OUT_Z);
            end
        end
    endtask


    task automatic check_expected;
        input logic [255:0] exp_x;
        input logic [255:0] exp_y;
        input logic [255:0] exp_z;
        begin
            check_no_x();

            if ((OUT_X !== exp_x) || (OUT_Y !== exp_y) || (OUT_Z !== exp_z)) begin
                $display("[FAIL] Output mismatch");
                $display("OUT_X  = %h", OUT_X);
                $display("EXP_X  = %h", exp_x);
                $display("OUT_Y  = %h", OUT_Y);
                $display("EXP_Y  = %h", exp_y);
                $display("OUT_Z  = %h", OUT_Z);
                $display("EXP_Z  = %h", exp_z);
                $fatal;
            end else begin
                $display("[PASS] Output matches expected Python reference");
            end
        end
    endtask


    initial begin
        $display("========================================");
        $display("=== tb_reduce_buckets_4L start ===");
        $display("========================================");


        // ------------------------------------------------------------
        // Test 1:
        //
        // Buckets:
        //   B1 = G
        //   B2 = 2G
        //   B3 = 3G
        //
        // Reduction:
        //   running_sum = INF
        //   result      = INF
        //
        //   i=3: running_sum = B3
        //        result      = B3
        //
        //   i=2: running_sum = B3 + B2
        //        result      = B3 + running_sum
        //
        //   i=1: running_sum = B3 + B2 + B1
        //        result      = result + running_sum
        //
        // Equivalent scalar result:
        //   result = 1*B1 + 2*B2 + 3*B3
        //          = 1*G + 2*(2G) + 3*(3G)
        //          = 14G
        //
        // Expected values generated from:
        //   tests/python/gen_expected_reduce4.py
        // ------------------------------------------------------------
        $display("");
        $display("[TEST 1] reduce buckets: B1=G, B2=2G, B3=3G");

        // B1 = G, Jacobian Montgomery
        B1_X = 256'h9981e643e9089f48979f48c033fd129c231e295329bc66dbd7362e5a487e2097;
        B1_Y = 256'hcf3f851fd4a582d670b6b59aac19c1368dfc5d5d1f1dc64db15ea6d2d3dbabe2;
        B1_Z = 256'h00000000000000000000000000000000000000000000000000000001000003d1;

        // B2 = 2G, Jacobian Montgomery
        B2_X = 256'h7c75dd9524177d593c03889b8dcd9b1cb05fb7d2a3da7fe8ba9f29b104e7db13;
        B2_Y = 256'h55debb381f4ad034cc27cb48a46449aaa87d43fdb563384b1cd20838e6fddc9f;
        B2_Z = 256'h9e7f0a3fa94b05ace16d6b355833826d1bf8baba3e3b8c9b62bd4da6a7b75b95;

        // B3 = 3G, Jacobian Montgomery
        B3_X = 256'h019fa59f6f459fc6748fa0a875006844fc39bed026e15b2769cd0e0931000a12;
        B3_Y = 256'hf03f524e8729a2d670f5f5be0a33eedc2fc8d898b67b2802b68ef68395abd131;
        B3_Z = 256'hc2c26ed3e5be9201db856e0c5e96b76d5d182c134369ed8ecd3f6a303370697b;

        #10;

        check_expected(
            256'h0b61ceeceaefba96315acc5aaa3a8f46abb9f8fb84d275c7ccf31e9f834bee9a,
            256'h075ee616368a466f1e2293ce50b237c588430d899396618b408578cdc6d7e092,
            256'hfb29610a7e974391507cf87d71f78fc3b2a93bb791e924d09b55d90516ca39fe
        );


        $display("");
        $display("========================================");
        $display("=== tb_reduce_buckets_4L PASSED ===");
        $display("========================================");

        $finish;
    end

endmodule