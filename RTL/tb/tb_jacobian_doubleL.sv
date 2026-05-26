`timescale 1ns/1ps

module tb_jacobian_doubleL;

    logic [255:0] X1;
    logic [255:0] Y1;
    logic [255:0] Z1;

    logic [255:0] X3;
    logic [255:0] Y3;
    logic [255:0] Z3;

    jacobian_doubleL dut (
        .i_X1(X1),
        .i_Y1(Y1),
        .i_Z1(Z1),
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
        $display("=== tb_jacobian_doubleL start ===");
        $display("========================================");

        // ------------------------------------------------------------
        // Test 1:
        // Point at infinity.
        // In Jacobian representation, Z=0 means INF.
        // Our RTL returns (0,0,0) for this case.
        // ------------------------------------------------------------
        $display("");
        $display("[TEST 1] Infinity case: Z1 = 0");

        X1 = 256'h1234;
        Y1 = 256'h5678;
        Z1 = 256'h0;

        #10;

        if ((X3 !== 256'h0) || (Y3 !== 256'h0) || (Z3 !== 256'h0)) begin
            $display("[FAIL] Infinity test failed");
            $display("X3 = %h", X3);
            $display("Y3 = %h", Y3);
            $display("Z3 = %h", Z3);
            $fatal;
        end else begin
            $display("[PASS] Infinity test");
        end


        // ------------------------------------------------------------
        // Test 2:
        // Simple non-zero values.
        // This is not a mathematical correctness test.
        // It only checks that the combinational datapath does not
        // produce unknown X values.
        // ------------------------------------------------------------
        $display("");
        $display("[TEST 2] Simple non-zero smoke test");

        X1 = 256'h1;
        Y1 = 256'h2;
        Z1 = 256'h1;

        #10;

        check_no_x();


        // ------------------------------------------------------------
        // Test 3:
        // secp256k1 generator point, but converted to Montgomery domain.
        //
        // Generated from Python:
        //
        // P = (
        //   to_mont(Gx),
        //   to_mont(Gy),
        //   ONE_M
        // )
        //
        // D = jacobian_double(P)
        //
        // Expected output is also in Montgomery-domain Jacobian form.
        // ------------------------------------------------------------
        $display("");
        $display("[TEST 3] Generator doubling, Montgomery-domain, compare to Python");

        X1 = 256'h9981e643e9089f48979f48c033fd129c231e295329bc66dbd7362e5a487e2097;
        Y1 = 256'hcf3f851fd4a582d670b6b59aac19c1368dfc5d5d1f1dc64db15ea6d2d3dbabe2;
        Z1 = 256'h00000000000000000000000000000000000000000000000000000001000003d1;

        #10;

        check_expected(
            256'h7c75dd9524177d593c03889b8dcd9b1cb05fb7d2a3da7fe8ba9f29b104e7db13,
            256'h55debb381f4ad034cc27cb48a46449aaa87d43fdb563384b1cd20838e6fddc9f,
            256'h9e7f0a3fa94b05ace16d6b355833826d1bf8baba3e3b8c9b62bd4da6a7b75b95
        );


        $display("");
        $display("========================================");
        $display("=== tb_jacobian_doubleL PASSED ===");
        $display("========================================");

        $finish;
    end

endmodule