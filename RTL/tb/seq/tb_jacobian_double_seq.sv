`timescale 1ns/1ps

module tb_jacobian_double_seq;

    localparam int WIDTH = 256;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    // Generator G in Montgomery-domain Jacobian coordinates
    localparam logic [255:0] GX_M =
        256'h9981E643E9089F48979F48C033FD129C231E295329BC66DBD7362E5A487E2097;

    localparam logic [255:0] GY_M =
        256'hCF3F851FD4A582D670B6B59AAC19C1368DFC5D5D1F1DC64DB15EA6D2D3DBABE2;

    localparam logic [255:0] GZ_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    // Expected jacobian_double(G), in Montgomery-domain Jacobian coordinates
    localparam logic [255:0] EXP_2G_X =
        256'h7C75DD9524177D593C03889B8DCD9B1CB05FB7D2A3DA7FE8BA9F29B104E7DB13;

    localparam logic [255:0] EXP_2G_Y =
        256'h55DEBB381F4AD034CC27CB48A46449AAA87D43FDB563384B1CD20838E6FDDC9F;

    localparam logic [255:0] EXP_2G_Z =
        256'h9E7F0A3FA94B05ACE16D6B355833826D1BF8BABA3E3B8C9B62BD4DA6A7B75B95;

    logic clk = 1'b0;
    logic rst_n;

    logic start;
    logic [WIDTH-1:0] X1;
    logic [WIDTH-1:0] Y1;
    logic [WIDTH-1:0] Z1;

    logic busy;
    logic done;
    logic [WIDTH-1:0] X3;
    logic [WIDTH-1:0] Y3;
    logic [WIDTH-1:0] Z3;

    always #5 clk = ~clk;

    jacobian_double_seq #(
        .WIDTH(WIDTH)
    ) dut (
        .clk   (clk),
        .rst_n (rst_n),

        .start (start),
        .X1    (X1),
        .Y1    (Y1),
        .Z1    (Z1),

        .busy  (busy),
        .done  (done),
        .X3    (X3),
        .Y3    (Y3),
        .Z3    (Z3)
    );

    task automatic do_double_check(
        input logic [WIDTH-1:0] in_X,
        input logic [WIDTH-1:0] in_Y,
        input logic [WIDTH-1:0] in_Z,
        input logic [WIDTH-1:0] exp_X,
        input logic [WIDTH-1:0] exp_Y,
        input logic [WIDTH-1:0] exp_Z,
        input string            name
    );
        int cycles;

        begin
            @(posedge clk);

            if (busy) begin
                $fatal(1, "[%s] Tried to start while busy", name);
            end

            X1    <= in_X;
            Y1    <= in_Y;
            Z1    <= in_Z;
            start <= 1'b1;

            @(posedge clk);
            start <= 1'b0;
            X1    <= '0;
            Y1    <= '0;
            Z1    <= '0;

            cycles = 0;
            while (!done) begin
                @(posedge clk);
                cycles++;

                if (cycles > 1000) begin
                    $fatal(1, "[%s] TIMEOUT waiting for done", name);
                end
            end

            $display("[%s] latency = %0d cycles", name, cycles);
            $display("[%s] X3 = %h", name, X3);
            $display("[%s] Y3 = %h", name, Y3);
            $display("[%s] Z3 = %h", name, Z3);

            if (X3 !== exp_X || Y3 !== exp_Y || Z3 !== exp_Z) begin
                $display("[%s] EXPECTED X = %h", name, exp_X);
                $display("[%s] GOT      X = %h", name, X3);
                $display("[%s] EXPECTED Y = %h", name, exp_Y);
                $display("[%s] GOT      Y = %h", name, Y3);
                $display("[%s] EXPECTED Z = %h", name, exp_Z);
                $display("[%s] GOT      Z = %h", name, Z3);
                $fatal(1, "[%s] FAILED", name);
            end else begin
                $display("[%s] PASSED", name);
            end

            @(posedge clk);
        end
    endtask

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        X1 = '0;
        Y1 = '0;
        Z1 = '0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        $display("==============================================");
        $display(" tb_jacobian_double_seq START");
        $display("==============================================");

        // Infinity input: Z = 0
        // Expected INF representation: (0, ONE_M, 0)
        do_double_check(
            256'h0,
            ONE_M,
            256'h0,
            256'h0,
            ONE_M,
            256'h0,
            "infinity_case"
        );

        // Generator doubling: G -> 2G
        do_double_check(
            GX_M,
            GY_M,
            GZ_M,
            EXP_2G_X,
            EXP_2G_Y,
            EXP_2G_Z,
            "generator_G_double"
        );

        $display("==============================================");
        $display(" tb_jacobian_double_seq PASSED");
        $display("==============================================");

        $finish;
    end

endmodule