`timescale 1ns/1ps

module tb_bucket_build_mem_seq_4_ext;

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

    localparam logic [255:0] G3_AFF_X =
        256'h9497730FCDF4C0AD5940D07385985972066CEAFB22EB7BC42379D4BBD5FEA781;

    localparam logic [255:0] G3_AFF_Y =
        256'h3EC28DCD9215EC76CC6048BD84885650AC4964CDC5A1F91FAF18B0B0613F55A9;

    localparam logic [255:0] EXP_2G_JAC_X =
        256'h7C75DD9524177D593C03889B8DCD9B1CB05FB7D2A3DA7FE8BA9F29B104E7DB13;

    localparam logic [255:0] EXP_2G_JAC_Y =
        256'h55DEBB381F4AD034CC27CB48A46449AAA87D43FDB563384B1CD20838E6FDDC9F;

    localparam logic [255:0] EXP_2G_JAC_Z =
        256'h9E7F0A3FA94B05ACE16D6B355833826D1BF8BABA3E3B8C9B62BD4DA6A7B75B95;

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;

    logic [2:0] processed_count;
    logic [2:0] skipped_count;

    logic [255:0] last_x;
    logic [255:0] last_y;
    logic [255:0] last_z;

    logic mem_valid;
    logic mem_write_en;
    logic [ADDR_W-1:0] mem_addr;

    logic [DATA_W-1:0] mem_wdata_x;
    logic [DATA_W-1:0] mem_wdata_y;
    logic [DATA_W-1:0] mem_wdata_z;

    logic mem_ready;
    logic mem_rvalid;

    logic [DATA_W-1:0] mem_rdata_x;
    logic [DATA_W-1:0] mem_rdata_y;
    logic [DATA_W-1:0] mem_rdata_z;

    bucket_mem_3coord #(
        .ADDR_W(ADDR_W),
        .DATA_W(DATA_W),
        .DEPTH (DEPTH)
    ) u_bucket_mem (
        .clk      (clk),
        .rst_n    (rst_n),

        .valid    (mem_valid),
        .write_en (mem_write_en),
        .addr     (mem_addr),

        .wdata_x  (mem_wdata_x),
        .wdata_y  (mem_wdata_y),
        .wdata_z  (mem_wdata_z),

        .ready    (mem_ready),
        .rvalid   (mem_rvalid),

        .rdata_x  (mem_rdata_x),
        .rdata_y  (mem_rdata_y),
        .rdata_z  (mem_rdata_z)
    );

    bucket_build_mem_seq_4_ext #(
        .ADDR_W(ADDR_W),
        .DATA_W(DATA_W),
        .DEPTH (DEPTH)
    ) dut (
        .clk   (clk),
        .rst_n (rst_n),

        .start (start),

        .p0_x  (GX_M),
        .p0_y  (GY_M),
        .b0    (4'd1),

        .p1_x  (G2_AFF_X),
        .p1_y  (G2_AFF_Y),
        .b1    (4'd2),

        .p2_x  (G3_AFF_X),
        .p2_y  (G3_AFF_Y),
        .b2    (4'd3),

        .p3_x  (GX_M),
        .p3_y  (GY_M),
        .b3    (4'd1),

        .busy  (busy),
        .done  (done),

        .processed_count(processed_count),
        .skipped_count  (skipped_count),

        .last_x(last_x),
        .last_y(last_y),
        .last_z(last_z),

        .mem_valid    (mem_valid),
        .mem_write_en (mem_write_en),
        .mem_addr     (mem_addr),

        .mem_wdata_x  (mem_wdata_x),
        .mem_wdata_y  (mem_wdata_y),
        .mem_wdata_z  (mem_wdata_z),

        .mem_ready    (mem_ready),
        .mem_rvalid   (mem_rvalid),

        .mem_rdata_x  (mem_rdata_x),
        .mem_rdata_y  (mem_rdata_y),
        .mem_rdata_z  (mem_rdata_z)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    task automatic check_bucket(
        input string       test_name,
        input int unsigned idx,
        input logic [255:0] exp_x,
        input logic [255:0] exp_y,
        input logic [255:0] exp_z
    );
        logic [255:0] got_x;
        logic [255:0] got_y;
        logic [255:0] got_z;
        begin
            got_x = u_bucket_mem.u_bucket_x_mem.mem[idx];
            got_y = u_bucket_mem.u_bucket_y_mem.mem[idx];
            got_z = u_bucket_mem.u_bucket_z_mem.mem[idx];

            if (got_x !== exp_x || got_y !== exp_y || got_z !== exp_z) begin
                $display("[TB] %s FAILED bucket=%0d", test_name, idx);
                $display("[TB] EXPECTED X = %064h", exp_x);
                $display("[TB] GOT      X = %064h", got_x);
                $display("[TB] EXPECTED Y = %064h", exp_y);
                $display("[TB] GOT      Y = %064h", got_y);
                $display("[TB] EXPECTED Z = %064h", exp_z);
                $display("[TB] GOT      Z = %064h", got_z);
                $fatal(1, "[TB] bucket_build_mem_seq_4_ext bucket mismatch");
            end else begin
                $display("[TB] %s PASSED bucket=%0d", test_name, idx);
            end
        end
    endtask

    initial begin
        start = 1'b0;

        rst_n = 1'b0;
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        $display("[TB] Starting bucket_build_mem_seq_4_ext test");

        @(negedge clk);
        start = 1'b1;

        @(negedge clk);
        start = 1'b0;

        wait (done === 1'b1);
        @(posedge clk);

        if (processed_count !== 3'd4) begin
            $fatal(1, "[TB] processed_count FAILED: got=%0d expected=4", processed_count);
        end else begin
            $display("[TB] processed_count PASSED");
        end

        if (skipped_count !== 3'd0) begin
            $fatal(1, "[TB] skipped_count FAILED: got=%0d expected=0", skipped_count);
        end else begin
            $display("[TB] skipped_count PASSED");
        end

        check_bucket(
            "bucket1_expected_2G",
            1,
            EXP_2G_JAC_X,
            EXP_2G_JAC_Y,
            EXP_2G_JAC_Z
        );

        check_bucket(
            "bucket2_expected_2G_affine",
            2,
            G2_AFF_X,
            G2_AFF_Y,
            ONE_M
        );

        check_bucket(
            "bucket3_expected_3G_affine",
            3,
            G3_AFF_X,
            G3_AFF_Y,
            ONE_M
        );

        check_bucket(
            "bucket4_still_INF",
            4,
            ZERO,
            ONE_M,
            ZERO
        );

        $display("[TB] tb_bucket_build_mem_seq_4_ext PASSED");
        #20;
        $finish;
    end

endmodule

