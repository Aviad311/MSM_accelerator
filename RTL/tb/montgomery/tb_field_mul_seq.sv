`timescale 1ns/1ps

module tb_field_mul_seq;

    localparam int WIDTH = 256;

    // secp256k1 Montgomery-domain constants
    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    // secp256k1 generator coordinates in Montgomery domain
    localparam logic [255:0] GX_M =
        256'h9981E643E9089F48979F48C033FD129C231E295329BC66DBD7362E5A487E2097;

    localparam logic [255:0] GY_M =
        256'hCF3F851FD4A582D670B6B59AAC19C1368DFC5D5D1F1DC64DB15EA6D2D3DBABE2;

    logic clk = 1'b0;
    logic rst_n;

    logic start;
    logic [WIDTH-1:0] a;
    logic [WIDTH-1:0] b;

    logic busy;
    logic done;
    logic [WIDTH-1:0] result;

    // 10ns period clock
    always #5 clk = ~clk;

    field_mul_seq #(
        .WIDTH(WIDTH)
    ) dut (
        .clk    (clk),
        .rst_n  (rst_n),
        .start  (start),
        .a      (a),
        .b      (b),
        .busy   (busy),
        .done   (done),
        .result (result)
    );

    task automatic do_mul_check(
        input logic [WIDTH-1:0] aa,
        input logic [WIDTH-1:0] bb,
        input logic [WIDTH-1:0] expected,
        input string            name
    );
        int cycles;

        begin
            @(posedge clk);

            if (busy) begin
                $fatal(1, "[%s] Tried to start while busy", name);
            end

            // Pulse start for one cycle
            a     <= aa;
            b     <= bb;
            start <= 1'b1;

            @(posedge clk);
            start <= 1'b0;
            a     <= '0;
            b     <= '0;

            // Wait for wrapper done
            cycles = 0;
            while (!done) begin
                @(posedge clk);
                cycles++;

                if (cycles > 1000) begin
                    $fatal(1, "[%s] TIMEOUT waiting for done", name);
                end
            end

            $display("[%s] result = %h", name, result);
            $display("[%s] wrapper latency = %0d cycles", name, cycles);

            if (result !== expected) begin
                $display("[%s] EXPECTED = %h", name, expected);
                $display("[%s] GOT      = %h", name, result);
                $fatal(1, "[%s] FAILED", name);
            end else begin
                $display("[%s] PASSED", name);
            end

            // Give the FSM one cycle to return cleanly to IDLE
            @(posedge clk);
        end
    endtask

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        a = '0;
        b = '0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        $display("==============================================");
        $display(" tb_field_mul_seq START");
        $display("==============================================");

        // ---------------------------------------------------------
        // Basic zero tests
        // ---------------------------------------------------------
        do_mul_check(
            256'h0,
            256'h0,
            256'h0,
            "zero_times_zero"
        );

        do_mul_check(
            256'h0,
            ONE_M,
            256'h0,
            "zero_times_one_m"
        );

        do_mul_check(
            ONE_M,
            256'h0,
            256'h0,
            "one_m_times_zero"
        );

        // ---------------------------------------------------------
        // Montgomery identity tests
        // In Montgomery domain:
        //   ONE_M * X = X
        //   X * ONE_M = X
        // ---------------------------------------------------------
        do_mul_check(
            ONE_M,
            ONE_M,
            ONE_M,
            "one_m_times_one_m"
        );

        do_mul_check(
            GX_M,
            ONE_M,
            GX_M,
            "gx_m_times_one_m"
        );

        do_mul_check(
            ONE_M,
            GX_M,
            GX_M,
            "one_m_times_gx_m"
        );

        do_mul_check(
            GY_M,
            ONE_M,
            GY_M,
            "gy_m_times_one_m"
        );

        do_mul_check(
            ONE_M,
            GY_M,
            GY_M,
            "one_m_times_gy_m"
        );

        $display("==============================================");
        $display(" tb_field_mul_seq PASSED");
        $display("==============================================");

        $finish;
    end

endmodule