`timescale 1ns/1ps

module tb_bucket_build_seq_4;

    localparam int WIDTH = 256;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    // G affine, Montgomery domain
    localparam logic [255:0] G1_X =
        256'h9981E643E9089F48979F48C033FD129C231E295329BC66DBD7362E5A487E2097;

    localparam logic [255:0] G1_Y =
        256'hCF3F851FD4A582D670B6B59AAC19C1368DFC5D5D1F1DC64DB15EA6D2D3DBABE2;

    // 2G affine, Montgomery domain
    localparam logic [255:0] G2_AFF_X =
        256'hF918623CCBA0EE23CE0B62E1E014040471354AFC88B285A04E0640C981048D2C;

    localparam logic [255:0] G2_AFF_Y =
        256'h3C7F7712157B93134B3A0F64BDA2CC6584FD25167DC75CE17D12D622FFACCFBF;

    // 3G affine, Montgomery domain
    localparam logic [255:0] G3_AFF_X =
        256'h9497730FCDF4C0AD5940D07385985972066CEAFB22EB7BC42379D4BBD5FEA781;

    localparam logic [255:0] G3_AFF_Y =
        256'h3EC28DCD9215EC76CC6048BD84885650AC4964CDC5A1F91FAF18B0B0613F55A9;

    // Expected bucket1 = G + G = 2G, Jacobian Montgomery
    localparam logic [255:0] EXP_B1_X =
        256'h7C75DD9524177D593C03889B8DCD9B1CB05FB7D2A3DA7FE8BA9F29B104E7DB13;

    localparam logic [255:0] EXP_B1_Y =
        256'h55DEBB381F4AD034CC27CB48A46449AAA87D43FDB563384B1CD20838E6FDDC9F;

    localparam logic [255:0] EXP_B1_Z =
        256'h9E7F0A3FA94B05ACE16D6B355833826D1BF8BABA3E3B8C9B62BD4DA6A7B75B95;

    // Expected bucket2 = 2G inserted into empty bucket,
    // so it becomes affine 2G represented as Jacobian with Z=ONE_M.
    localparam logic [255:0] EXP_B2_X =
        256'hF918623CCBA0EE23CE0B62E1E014040471354AFC88B285A04E0640C981048D2C;

    localparam logic [255:0] EXP_B2_Y =
        256'h3C7F7712157B93134B3A0F64BDA2CC6584FD25167DC75CE17D12D622FFACCFBF;

    localparam logic [255:0] EXP_B2_Z =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    // Expected bucket3 = 3G inserted into empty bucket,
    // so it becomes affine 3G represented as Jacobian with Z=ONE_M.
    localparam logic [255:0] EXP_B3_X =
        256'h9497730FCDF4C0AD5940D07385985972066CEAFB22EB7BC42379D4BBD5FEA781;

    localparam logic [255:0] EXP_B3_Y =
        256'h3EC28DCD9215EC76CC6048BD84885650AC4964CDC5A1F91FAF18B0B0613F55A9;

    localparam logic [255:0] EXP_B3_Z =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    logic clk = 1'b0;
    logic rst_n;

    logic start;
    logic busy;
    logic done;

    logic [WIDTH-1:0] p0_x;
    logic [WIDTH-1:0] p0_y;
    logic [1:0]       p0_bid;

    logic [WIDTH-1:0] p1_x;
    logic [WIDTH-1:0] p1_y;
    logic [1:0]       p1_bid;

    logic [WIDTH-1:0] p2_x;
    logic [WIDTH-1:0] p2_y;
    logic [1:0]       p2_bid;

    logic [WIDTH-1:0] p3_x;
    logic [WIDTH-1:0] p3_y;
    logic [1:0]       p3_bid;

    logic [WIDTH-1:0] b1_x;
    logic [WIDTH-1:0] b1_y;
    logic [WIDTH-1:0] b1_z;

    logic [WIDTH-1:0] b2_x;
    logic [WIDTH-1:0] b2_y;
    logic [WIDTH-1:0] b2_z;

    logic [WIDTH-1:0] b3_x;
    logic [WIDTH-1:0] b3_y;
    logic [WIDTH-1:0] b3_z;

    always #5 clk = ~clk;

    bucket_build_seq_4 #(
        .WIDTH(WIDTH)
    ) dut (
        .clk   (clk),
        .rst_n (rst_n),

        .start (start),

        .p0_x   (p0_x),
        .p0_y   (p0_y),
        .p0_bid (p0_bid),

        .p1_x   (p1_x),
        .p1_y   (p1_y),
        .p1_bid (p1_bid),

        .p2_x   (p2_x),
        .p2_y   (p2_y),
        .p2_bid (p2_bid),

        .p3_x   (p3_x),
        .p3_y   (p3_y),
        .p3_bid (p3_bid),

        .busy (busy),
        .done (done),

        .b1_x (b1_x),
        .b1_y (b1_y),
        .b1_z (b1_z),

        .b2_x (b2_x),
        .b2_y (b2_y),
        .b2_z (b2_z),

        .b3_x (b3_x),
        .b3_y (b3_y),
        .b3_z (b3_z)
    );

    task automatic run_bucket_build_check;
        int cycles;

        begin
            @(posedge clk);

            if (busy) begin
                $fatal(1, "Tried to start while busy");
            end

            // P0 = G   -> bucket 1
            p0_x   <= G1_X;
            p0_y   <= G1_Y;
            p0_bid <= 2'd1;

            // P1 = 2G  -> bucket 2
            p1_x   <= G2_AFF_X;
            p1_y   <= G2_AFF_Y;
            p1_bid <= 2'd2;

            // P2 = 3G  -> bucket 3
            p2_x   <= G3_AFF_X;
            p2_y   <= G3_AFF_Y;
            p2_bid <= 2'd3;

            // P3 = G   -> bucket 1
            p3_x   <= G1_X;
            p3_y   <= G1_Y;
            p3_bid <= 2'd1;

            start <= 1'b1;

            @(posedge clk);
            start <= 1'b0;

            p0_x   <= '0;
            p0_y   <= '0;
            p0_bid <= 2'd0;

            p1_x   <= '0;
            p1_y   <= '0;
            p1_bid <= 2'd0;

            p2_x   <= '0;
            p2_y   <= '0;
            p2_bid <= 2'd0;

            p3_x   <= '0;
            p3_y   <= '0;
            p3_bid <= 2'd0;

            cycles = 0;
            while (!done) begin
                @(posedge clk);
                cycles++;

                if (cycles > 3000) begin
                    $fatal(1, "TIMEOUT waiting for done");
                end
            end

            $display("[bucket_build_seq_4] latency = %0d cycles", cycles);

            $display("[bucket1] X = %h", b1_x);
            $display("[bucket1] Y = %h", b1_y);
            $display("[bucket1] Z = %h", b1_z);

            $display("[bucket2] X = %h", b2_x);
            $display("[bucket2] Y = %h", b2_y);
            $display("[bucket2] Z = %h", b2_z);

            $display("[bucket3] X = %h", b3_x);
            $display("[bucket3] Y = %h", b3_y);
            $display("[bucket3] Z = %h", b3_z);

            if (b1_x !== EXP_B1_X || b1_y !== EXP_B1_Y || b1_z !== EXP_B1_Z) begin
                $display("BUCKET1 EXPECTED X = %h", EXP_B1_X);
                $display("BUCKET1 GOT      X = %h", b1_x);
                $display("BUCKET1 EXPECTED Y = %h", EXP_B1_Y);
                $display("BUCKET1 GOT      Y = %h", b1_y);
                $display("BUCKET1 EXPECTED Z = %h", EXP_B1_Z);
                $display("BUCKET1 GOT      Z = %h", b1_z);
                $fatal(1, "bucket1 mismatch");
            end

            if (b2_x !== EXP_B2_X || b2_y !== EXP_B2_Y || b2_z !== EXP_B2_Z) begin
                $display("BUCKET2 EXPECTED X = %h", EXP_B2_X);
                $display("BUCKET2 GOT      X = %h", b2_x);
                $display("BUCKET2 EXPECTED Y = %h", EXP_B2_Y);
                $display("BUCKET2 GOT      Y = %h", b2_y);
                $display("BUCKET2 EXPECTED Z = %h", EXP_B2_Z);
                $display("BUCKET2 GOT      Z = %h", b2_z);
                $fatal(1, "bucket2 mismatch");
            end

            if (b3_x !== EXP_B3_X || b3_y !== EXP_B3_Y || b3_z !== EXP_B3_Z) begin
                $display("BUCKET3 EXPECTED X = %h", EXP_B3_X);
                $display("BUCKET3 GOT      X = %h", b3_x);
                $display("BUCKET3 EXPECTED Y = %h", EXP_B3_Y);
                $display("BUCKET3 GOT      Y = %h", b3_y);
                $display("BUCKET3 EXPECTED Z = %h", EXP_B3_Z);
                $display("BUCKET3 GOT      Z = %h", b3_z);
                $fatal(1, "bucket3 mismatch");
            end

            $display("[bucket_build_seq_4] PASSED");

            @(posedge clk);
        end
    endtask

    initial begin
        rst_n = 1'b0;
        start = 1'b0;

        p0_x   = '0;
        p0_y   = '0;
        p0_bid = 2'd0;

        p1_x   = '0;
        p1_y   = '0;
        p1_bid = 2'd0;

        p2_x   = '0;
        p2_y   = '0;
        p2_bid = 2'd0;

        p3_x   = '0;
        p3_y   = '0;
        p3_bid = 2'd0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        $display("==============================================");
        $display(" tb_bucket_build_seq_4 START");
        $display("==============================================");

        run_bucket_build_check();

        $display("==============================================");
        $display(" tb_bucket_build_seq_4 PASSED");
        $display("==============================================");

        $finish;
    end

endmodule