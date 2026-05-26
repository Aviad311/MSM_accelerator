`timescale 1ns/1ps

module tb_jacobian_mixed_addL;

    logic [255:0] X1;
    logic [255:0] Y1;
    logic [255:0] Z1;

    logic [255:0] X2;
    logic [255:0] Y2;

    logic [255:0] X3;
    logic [255:0] Y3;
    logic [255:0] Z3;

    jacobian_mixed_addL dut (
        .i_X1(X1),
        .i_Y1(Y1),
        .i_Z1(Z1),

        .i_X2(X2),
        .i_Y2(Y2),

        .o_X3(X3),
        .o_Y3(Y3),
        .o_Z3(Z3)
    );


    task automatic check_no_x;
        begin
            if ((^X3 === 1'bx) || (^Y3 === 1'bx) || (^Z3 === 1'bx)) begin
                $display("[FAIL] Output contains X");
                $display("X3 = %h", X3);
                $display("Y3 = %h", Y3);
                $display("Z3 = %h", Z3);
                $fatal;
            end else begin
                $display("[PASS] No X values");
                $display("X3 = %h", X3);
                $display("Y3 = %h", Y3);
                $display("Z3 = %h", Z3);
            end
        end
    endtask


    task automatic check_expected;
        input logic [255:0] exp_x;
        input logic [255:0] exp_y;
        input logic [255:0] exp_z;
        begin
            check_no_x();

            if ((X3 !== exp_x) || (Y3 !== exp_y) || (Z3 !== exp_z)) begin
                $display("[FAIL] Output mismatch");
                $display("X3      = %h", X3);
                $display("EXP_X3  = %h", exp_x);
                $display("Y3      = %h", Y3);
                $display("EXP_Y3  = %h", exp_y);
                $display("Z3      = %h", Z3);
                $display("EXP_Z3  = %h", exp_z);
                $fatal;
            end else begin
                $display("[PASS] Output matches expected Python reference");
            end
        end
    endtask


    initial begin
        $display("========================================");
        $display("=== tb_jacobian_mixed_addL start ===");
        $display("========================================");


        // ------------------------------------------------------------
        // Test 1:
        // P is infinity: Z1 = 0.
        // mixed_add(INF, Q) should return Q as Jacobian:
        // (X2, Y2, ONE_M)
        // ------------------------------------------------------------
        $display("");
        $display("[TEST 1] P is infinity, expect Q as Jacobian");

        X1 = 256'h0;
        Y1 = 256'h0;
        Z1 = 256'h0;

        X2 = 256'h9981e643e9089f48979f48c033fd129c231e295329bc66dbd7362e5a487e2097;
        Y2 = 256'hcf3f851fd4a582d670b6b59aac19c1368dfc5d5d1f1dc64db15ea6d2d3dbabe2;

        #10;

        check_expected(
            256'h9981e643e9089f48979f48c033fd129c231e295329bc66dbd7362e5a487e2097,
            256'hcf3f851fd4a582d670b6b59aac19c1368dfc5d5d1f1dc64db15ea6d2d3dbabe2,
            256'h00000000000000000000000000000000000000000000000000000001000003d1
        );


        // ------------------------------------------------------------
        // Test 2:
        // P = 2G in Jacobian Montgomery.
        // Q = G in affine Montgomery.
        //
        // Expected:
        // R = 2G + G = 3G
        //
        // Values generated from Python:
        // jacobian_mixed_add_mont(jacobian_double(G), G_aff)
        // ------------------------------------------------------------
        $display("");
        $display("[TEST 2] P=2G Jacobian, Q=G affine, expect 3G");

        X1 = 256'h7c75dd9524177d593c03889b8dcd9b1cb05fb7d2a3da7fe8ba9f29b104e7db13;
        Y1 = 256'h55debb381f4ad034cc27cb48a46449aaa87d43fdb563384b1cd20838e6fddc9f;
        Z1 = 256'h9e7f0a3fa94b05ace16d6b355833826d1bf8baba3e3b8c9b62bd4da6a7b75b95;

        X2 = 256'h9981e643e9089f48979f48c033fd129c231e295329bc66dbd7362e5a487e2097;
        Y2 = 256'hcf3f851fd4a582d670b6b59aac19c1368dfc5d5d1f1dc64db15ea6d2d3dbabe2;

        #10;

        check_expected(
            256'h019fa59f6f459fc6748fa0a875006844fc39bed026e15b2769cd0e0931000a12,
            256'hf03f524e8729a2d670f5f5be0a33eedc2fc8d898b67b2802b68ef68395abd131,
            256'hc2c26ed3e5be9201db856e0c5e96b76d5d182c134369ed8ecd3f6a303370697b
        );


        $display("");
        $display("========================================");
        $display("=== tb_jacobian_mixed_addL PASSED ===");
        $display("========================================");

        $finish;
    end

endmodule