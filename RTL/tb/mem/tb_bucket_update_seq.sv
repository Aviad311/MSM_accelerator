`timescale 1ns/1ps

module tb_bucket_update_seq;

    localparam int ADDR_W = 4;
    localparam int DATA_W = 256;
    localparam int DEPTH  = (1 << ADDR_W);

    localparam logic [255:0] ZERO =
        256'h0000000000000000000000000000000000000000000000000000000000000000;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    localparam logic [255:0] GX_M =
        256'h9981E643E9089F48979F48C033FD129C231E295329BC66DBD7362E5A487E2097;

    localparam logic [255:0] GY_M =
        256'hCF3F851FD4A582D670B6B59AAC19C1368DFC5D5D1F1DC64DB15EA6D2D3DBABE2;

    localparam logic [255:0] G2_AFF_X =
        256'hF918623CCBA0EE23CE0B62E1E014040471354AFC88B285A04E0640C981048D2C;

    localparam logic [255:0] G2_AFF_Y =
        256'h3C7F7712157B93134B3A0F64BDA2CC6584FD25167DC75CE17D12D622FFACCFBF;

    localparam logic [255:0] EXP_2G_X =
        256'h7C75DD9524177D593C03889B8DCD9B1CB05FB7D2A3DA7FE8BA9F29B104E7DB13;

    localparam logic [255:0] EXP_2G_Y =
        256'h55DEBB381F4AD034CC27CB48A46449AAA87D43FDB563384B1CD20838E6FDDC9F;

    localparam logic [255:0] EXP_2G_Z =
        256'h9E7F0A3FA94B05ACE16D6B355833826D1BF8BABA3E3B8C9B62BD4DA6A7B75B95;

    logic clk;
    logic rst_n;

    logic                start;
    logic                clear_all;
    logic [ADDR_W-1:0]   bucket_id;
    logic [DATA_W-1:0]   point_x;
    logic [DATA_W-1:0]   point_y;

    logic                busy;
    logic                done;
    logic                skipped;

    logic [DATA_W-1:0]   last_x;
    logic [DATA_W-1:0]   last_y;
    logic [DATA_W-1:0]   last_z;

    bucket_update_seq #(
        .ADDR_W(ADDR_W),
        .DATA_W(DATA_W),
        .DEPTH (DEPTH)
    ) dut (
        .clk       (clk),
        .rst_n     (rst_n),

        .start     (start),
        .clear_all (clear_all),

        .bucket_id (bucket_id),
        .point_x   (point_x),
        .point_y   (point_y),

        .busy      (busy),
        .done      (done),
        .skipped   (skipped),

        .last_x    (last_x),
        .last_y    (last_y),
        .last_z    (last_z)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    task automatic pulse_clear_all;
        begin
            @(negedge clk);
            start     = 1'b1;
            clear_all = 1'b1;
            bucket_id = '0;
            point_x   = '0;
            point_y   = '0;

            @(negedge clk);
            start     = 1'b0;
            clear_all = 1'b0;

            wait (done === 1'b1);
            @(posedge clk);

            $display("[TB] clear_all PASSED");
        end
    endtask

    task automatic do_update_and_check(
        input string             test_name,
        input logic [ADDR_W-1:0] bid,
        input logic [255:0]      px,
        input logic [255:0]      py,
        input logic [255:0]      exp_x,
        input logic [255:0]      exp_y,
        input logic [255:0]      exp_z
    );
        begin
            @(negedge clk);
            start     = 1'b1;
            clear_all = 1'b0;
            bucket_id = bid;
            point_x   = px;
            point_y   = py;

            @(negedge clk);
            start     = 1'b0;
            bucket_id = '0;
            point_x   = '0;
            point_y   = '0;

            wait (done === 1'b1);

            if (last_x !== exp_x || last_y !== exp_y || last_z !== exp_z) begin
                $display("[TB] %s FAILED", test_name);
                $display("[TB] EXPECTED X = %064h", exp_x);
                $display("[TB] GOT      X = %064h", last_x);
                $display("[TB] EXPECTED Y = %064h", exp_y);
                $display("[TB] GOT      Y = %064h", last_y);
                $display("[TB] EXPECTED Z = %064h", exp_z);
                $display("[TB] GOT      Z = %064h", last_z);
                $fatal(1, "[TB] bucket_update_seq mismatch");
            end else begin
                $display("[TB] %s PASSED", test_name);
            end

            @(posedge clk);
        end
    endtask

    initial begin
        start     = 1'b0;
        clear_all = 1'b0;
        bucket_id = '0;
        point_x   = '0;
        point_y   = '0;

        rst_n = 1'b0;
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        $display("[TB] Starting bucket_update_seq test");

        pulse_clear_all();

        do_update_and_check(
            "bucket1_INF_plus_G",
            4'd1,
            GX_M,
            GY_M,
            GX_M,
            GY_M,
            ONE_M
        );

        do_update_and_check(
            "bucket1_G_plus_G_equals_2G",
            4'd1,
            GX_M,
            GY_M,
            EXP_2G_X,
            EXP_2G_Y,
            EXP_2G_Z
        );

        do_update_and_check(
            "bucket2_INF_plus_2G_affine",
            4'd2,
            G2_AFF_X,
            G2_AFF_Y,
            G2_AFF_X,
            G2_AFF_Y,
            ONE_M
        );

        do_update_and_check(
            "bucket0_skip",
            4'd0,
            GX_M,
            GY_M,
            ZERO,
            ONE_M,
            ZERO
        );

        $display("[TB] tb_bucket_update_seq PASSED");
        #20;
        $finish;
    end

endmodule