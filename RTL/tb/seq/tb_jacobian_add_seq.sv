`timescale 1ns/1ps

module tb_jacobian_add_seq;

    localparam int WIDTH = 256;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    // Generator G in Montgomery Jacobian coordinates
    localparam logic [255:0] GX_M =
        256'h9981E643E9089F48979F48C033FD129C231E295329BC66DBD7362E5A487E2097;

    localparam logic [255:0] GY_M =
        256'hCF3F851FD4A582D670B6B59AAC19C1368DFC5D5D1F1DC64DB15EA6D2D3DBABE2;

    localparam logic [255:0] GZ_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    // 2G in Montgomery Jacobian coordinates
    localparam logic [255:0] G2_X =
        256'h7C75DD9524177D593C03889B8DCD9B1CB05FB7D2A3DA7FE8BA9F29B104E7DB13;

    localparam logic [255:0] G2_Y =
        256'h55DEBB381F4AD034CC27CB48A46449AAA87D43FDB563384B1CD20838E6FDDC9F;

    localparam logic [255:0] G2_Z =
        256'h9E7F0A3FA94B05ACE16D6B355833826D1BF8BABA3E3B8C9B62BD4DA6A7B75B95;

    // 3G in Montgomery Jacobian coordinates
    localparam logic [255:0] G3_X =
        256'h019FA59F6F459FC6748FA0A875006844FC39BED026E15B2769CD0E0931000A12;

    localparam logic [255:0] G3_Y =
        256'hF03F524E8729A2D670F5F5BE0A33EEDC2FC8D898B67B2802B68EF68395ABD131;

    localparam logic [255:0] G3_Z =
        256'hC2C26ED3E5BE9201DB856E0C5E96B76D5D182C134369ED8ECD3F6A303370697B;

    // 5G in Montgomery Jacobian coordinates
    // Expected result for add(2G, 3G).
    localparam logic [255:0] G5_X =
        256'hD58F7E0AF9D42C90D3AECE875205D29E83AC8EDBE91410C1A6122E7254593A66;

    localparam logic [255:0] G5_Y =
        256'hEF44009FB9BEB725B99B9C8E00011F40A0B9A0F282D719FD93C849814AD0DB94;

    localparam logic [255:0] G5_Z =
        256'h1C0B0E6420A1B01BEAFA6AC8188FC20A4B2FFE6564148902657DFC57B9AFC66B;

    logic clk = 1'b0;
    logic rst_n;

    logic start;

    logic [WIDTH-1:0] X1;
    logic [WIDTH-1:0] Y1;
    logic [WIDTH-1:0] Z1;

    logic [WIDTH-1:0] X2;
    logic [WIDTH-1:0] Y2;
    logic [WIDTH-1:0] Z2;

    logic busy;
    logic done;

    logic [WIDTH-1:0] X3;
    logic [WIDTH-1:0] Y3;
    logic [WIDTH-1:0] Z3;

    always #5 clk = ~clk;

    jacobian_add_seq #(
        .WIDTH(WIDTH)
    ) dut (
        .clk   (clk),
        .rst_n (rst_n),

        .start (start),

        .X1    (X1),
        .Y1    (Y1),
        .Z1    (Z1),

        .X2    (X2),
        .Y2    (Y2),
        .Z2    (Z2),

        .busy  (busy),
        .done  (done),

        .X3    (X3),
        .Y3    (Y3),
        .Z3    (Z3)
    );

    task automatic do_add_check(
        input logic [WIDTH-1:0] in_X1,
        input logic [WIDTH-1:0] in_Y1,
        input logic [WIDTH-1:0] in_Z1,
        input logic [WIDTH-1:0] in_X2,
        input logic [WIDTH-1:0] in_Y2,
        input logic [WIDTH-1:0] in_Z2,
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

            X1    <= in_X1;
            Y1    <= in_Y1;
            Z1    <= in_Z1;

            X2    <= in_X2;
            Y2    <= in_Y2;
            Z2    <= in_Z2;

            start <= 1'b1;

            @(posedge clk);
            start <= 1'b0;

            X1 <= '0;
            Y1 <= '0;
            Z1 <= '0;

            X2 <= '0;
            Y2 <= '0;
            Z2 <= '0;

            cycles = 0;
            while (!done) begin
                @(posedge clk);
                cycles++;

                if (cycles > 3000) begin
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

        X2 = '0;
        Y2 = '0;
        Z2 = '0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        $display("==============================================");
        $display(" tb_jacobian_add_seq START");
        $display("==============================================");

        // ---------------------------------------------------------
        // Case 1:
        // INF + G = G
        // ---------------------------------------------------------
        do_add_check(
            256'h0,
            ONE_M,
            256'h0,

            GX_M,
            GY_M,
            GZ_M,

            GX_M,
            GY_M,
            GZ_M,

            "inf_plus_G"
        );

        // ---------------------------------------------------------
        // Case 2:
        // G + INF = G
        // ---------------------------------------------------------
        do_add_check(
            GX_M,
            GY_M,
            GZ_M,

            256'h0,
            ONE_M,
            256'h0,

            GX_M,
            GY_M,
            GZ_M,

            "G_plus_inf"
        );

        // ---------------------------------------------------------
        // Case 3:
        // G + G = 2G
        // This triggers the special double path.
        // ---------------------------------------------------------
        do_add_check(
            GX_M,
            GY_M,
            GZ_M,

            GX_M,
            GY_M,
            GZ_M,

            G2_X,
            G2_Y,
            G2_Z,

            "G_plus_G_special_double"
        );

        // ---------------------------------------------------------
        // Case 4:
        // 2G + 3G = 5G
        // Normal Jacobian add path.
        // ---------------------------------------------------------
        do_add_check(
            G2_X,
            G2_Y,
            G2_Z,

            G3_X,
            G3_Y,
            G3_Z,

            G5_X,
            G5_Y,
            G5_Z,

            "twoG_plus_threeG"
        );

        $display("==============================================");
        $display(" tb_jacobian_add_seq PASSED");
        $display("==============================================");

        $finish;
    end

endmodule