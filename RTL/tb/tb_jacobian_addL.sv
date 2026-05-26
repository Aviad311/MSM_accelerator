`timescale 1ns/1ps

module tb_jacobian_addL;

    logic [255:0] X1;
    logic [255:0] Y1;
    logic [255:0] Z1;

    logic [255:0] X2;
    logic [255:0] Y2;
    logic [255:0] Z2;

    logic [255:0] X3;
    logic [255:0] Y3;
    logic [255:0] Z3;

    jacobian_addL dut (
        .i_X1(X1),
        .i_Y1(Y1),
        .i_Z1(Z1),

        .i_X2(X2),
        .i_Y2(Y2),
        .i_Z2(Z2),

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
        $display("=== tb_jacobian_addL start ===");
        $display("========================================");


        // ------------------------------------------------------------
        // Test 1:
        // P is infinity: Z1 = 0.
        // add(INF, Q) should return Q.
        // Q here is 3G in Jacobian Montgomery.
        // ------------------------------------------------------------
        $display("");
        $display("[TEST 1] P is infinity, expect Q");

        X1 = 256'h0;
        Y1 = 256'h0;
        Z1 = 256'h0;

        X2 = 256'h019fa59f6f459fc6748fa0a875006844fc39bed026e15b2769cd0e0931000a12;
        Y2 = 256'hf03f524e8729a2d670f5f5be0a33eedc2fc8d898b67b2802b68ef68395abd131;
        Z2 = 256'hc2c26ed3e5be9201db856e0c5e96b76d5d182c134369ed8ecd3f6a303370697b;

        #10;

        check_expected(
            256'h019fa59f6f459fc6748fa0a875006844fc39bed026e15b2769cd0e0931000a12,
            256'hf03f524e8729a2d670f5f5be0a33eedc2fc8d898b67b2802b68ef68395abd131,
            256'hc2c26ed3e5be9201db856e0c5e96b76d5d182c134369ed8ecd3f6a303370697b
        );


        // ------------------------------------------------------------
        // Test 2:
        // Q is infinity: Z2 = 0.
        // add(P, INF) should return P.
        // P here is 2G in Jacobian Montgomery.
        // ------------------------------------------------------------
        $display("");
        $display("[TEST 2] Q is infinity, expect P");

        X1 = 256'h7c75dd9524177d593c03889b8dcd9b1cb05fb7d2a3da7fe8ba9f29b104e7db13;
        Y1 = 256'h55debb381f4ad034cc27cb48a46449aaa87d43fdb563384b1cd20838e6fddc9f;
        Z1 = 256'h9e7f0a3fa94b05ace16d6b355833826d1bf8baba3e3b8c9b62bd4da6a7b75b95;

        X2 = 256'h0;
        Y2 = 256'h0;
        Z2 = 256'h0;

        #10;

        check_expected(
            256'h7c75dd9524177d593c03889b8dcd9b1cb05fb7d2a3da7fe8ba9f29b104e7db13,
            256'h55debb381f4ad034cc27cb48a46449aaa87d43fdb563384b1cd20838e6fddc9f,
            256'h9e7f0a3fa94b05ace16d6b355833826d1bf8baba3e3b8c9b62bd4da6a7b75b95
        );


        // ------------------------------------------------------------
        // Test 3:
        // P = 2G in Jacobian Montgomery.
        // Q = 3G in Jacobian Montgomery.
        //
        // Expected:
        // R = P + Q = 5G
        //
        // Values generated from Python:
        // P = jacobian_double(G)
        // Q = jacobian_mixed_add_mont(P, G_aff)
        // R = jacobian_add(P, Q)
        // ------------------------------------------------------------
        $display("");
        $display("[TEST 3] P=2G, Q=3G, expect 5G");

        X1 = 256'h7c75dd9524177d593c03889b8dcd9b1cb05fb7d2a3da7fe8ba9f29b104e7db13;
        Y1 = 256'h55debb381f4ad034cc27cb48a46449aaa87d43fdb563384b1cd20838e6fddc9f;
        Z1 = 256'h9e7f0a3fa94b05ace16d6b355833826d1bf8baba3e3b8c9b62bd4da6a7b75b95;

        X2 = 256'h019fa59f6f459fc6748fa0a875006844fc39bed026e15b2769cd0e0931000a12;
        Y2 = 256'hf03f524e8729a2d670f5f5be0a33eedc2fc8d898b67b2802b68ef68395abd131;
        Z2 = 256'hc2c26ed3e5be9201db856e0c5e96b76d5d182c134369ed8ecd3f6a303370697b;

        #10;

        check_expected(
            256'hd58f7e0af9d42c90d3aece875205d29e83ac8edbe91410c1a6122e7254593a66,
            256'hef44009fb9beb725b99b9c8e00011f40a0b9a0f282d719fd93c849814ad0db94,
            256'h1c0b0e6420a1b01beafa6ac8188fc20a4b2ffe6564148902657dfc57b9afc66b
        );


        $display("");
        $display("========================================");
        $display("=== tb_jacobian_addL PASSED ===");
        $display("========================================");

        $finish;
    end

endmodule