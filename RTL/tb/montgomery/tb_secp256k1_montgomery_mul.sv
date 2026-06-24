`timescale 1ns/1ps

module tb_secp256k1_montgomery_mul;

    localparam int WIDTH = 256;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    logic clk = 1'b0;
    logic rst_n;

    logic in_valid;
    logic [WIDTH-1:0] op_a;
    logic [WIDTH-1:0] op_b;

    logic out_valid;
    logic [WIDTH-1:0] result;
    logic ready;

    always #5 clk = ~clk;

    secp256k1_montgomery_mul #(
        .WIDTH(WIDTH)
    ) dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (in_valid),
        .op_a      (op_a),
        .op_b      (op_b),
        .out_valid (out_valid),
        .result    (result),
        .ready     (ready)
    );

    task automatic do_mul_check(
        input logic [WIDTH-1:0] a,
        input logic [WIDTH-1:0] b,
        input logic [WIDTH-1:0] expected,
        input string            name
    );
        int cycles;
        begin
            @(posedge clk);
            op_a     <= a;
            op_b     <= b;
            in_valid <= 1'b1;

            @(posedge clk);
            in_valid <= 1'b0;
            op_a     <= '0;
            op_b     <= '0;

            cycles = 0;
            while (!out_valid) begin
                @(posedge clk);
                cycles++;

                if (cycles > 1000) begin
                    $fatal(1, "[%s] TIMEOUT waiting for out_valid", name);
                end
            end

            $display("[%s] result = %h", name, result);
            $display("[%s] latency = %0d cycles", name, cycles);

            if (result !== expected) begin
                $display("[%s] EXPECTED = %h", name, expected);
                $display("[%s] GOT      = %h", name, result);
                $fatal(1, "[%s] FAILED", name);
            end else begin
                $display("[%s] PASSED", name);
            end

            repeat (5) @(posedge clk);
        end
    endtask

    initial begin
        rst_n    = 1'b0;
        in_valid = 1'b0;
        op_a     = '0;
        op_b     = '0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        $display("==============================================");
        $display(" tb_secp256k1_montgomery_mul START");
        $display("==============================================");

        // 0 * 0 = 0
        do_mul_check(
            256'h0,
            256'h0,
            256'h0,
            "zero_times_zero"
        );

        // ONE_M * ONE_M = ONE_M in Montgomery domain
        do_mul_check(
            ONE_M,
            ONE_M,
            ONE_M,
            "one_m_times_one_m"
        );

        // 0 * ONE_M = 0
        do_mul_check(
            256'h0,
            ONE_M,
            256'h0,
            "zero_times_one_m"
        );

        // ONE_M * 0 = 0
        do_mul_check(
            ONE_M,
            256'h0,
            256'h0,
            "one_m_times_zero"
        );

        $display("==============================================");
        $display(" tb_secp256k1_montgomery_mul PASSED");
        $display("==============================================");

        $finish;
    end

endmodule