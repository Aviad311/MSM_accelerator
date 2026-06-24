`timescale 1ns/1ps

module tb_reduce_buckets_mem_seq_4;

    localparam int ADDR_W = 4;
    localparam int DATA_W = 256;
    localparam int DEPTH  = (1 << ADDR_W);

    localparam logic [255:0] ZERO =
        256'h0000000000000000000000000000000000000000000000000000000000000000;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

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

    localparam logic [255:0] EXP_15G_X =
        256'h095BC488048E05A5732C475C3A609EFCC38EC30F0B30A04E778684E3DD149772;

    localparam logic [255:0] EXP_15G_Y =
        256'hA95C53D653DEE15BE8482AD23B040A470B14DC6069A4204C751D6C1C6D8FDC7D;

    localparam logic [255:0] EXP_15G_Z =
        256'h93F59AC795686CC45912CC7F9918DE3F914DBDA84AD1331E5C2FD8DA0EBF9998;

    logic clk;
    logic rst_n;

    logic tb_drive_mem;

    logic tb_mem_valid;
    logic tb_mem_write_en;
    logic [ADDR_W-1:0] tb_mem_addr;

    logic [DATA_W-1:0] tb_mem_wdata_x;
    logic [DATA_W-1:0] tb_mem_wdata_y;
    logic [DATA_W-1:0] tb_mem_wdata_z;

    logic red_mem_valid;
    logic red_mem_write_en;
    logic [ADDR_W-1:0] red_mem_addr;

    logic [DATA_W-1:0] red_mem_wdata_x;
    logic [DATA_W-1:0] red_mem_wdata_y;
    logic [DATA_W-1:0] red_mem_wdata_z;

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

    logic start;
    logic busy;
    logic done;

    logic [DATA_W-1:0] result_x;
    logic [DATA_W-1:0] result_y;
    logic [DATA_W-1:0] result_z;

    assign mem_valid    = tb_drive_mem ? tb_mem_valid    : red_mem_valid;
    assign mem_write_en = tb_drive_mem ? tb_mem_write_en : red_mem_write_en;
    assign mem_addr     = tb_drive_mem ? tb_mem_addr     : red_mem_addr;

    assign mem_wdata_x  = tb_drive_mem ? tb_mem_wdata_x  : red_mem_wdata_x;
    assign mem_wdata_y  = tb_drive_mem ? tb_mem_wdata_y  : red_mem_wdata_y;
    assign mem_wdata_z  = tb_drive_mem ? tb_mem_wdata_z  : red_mem_wdata_z;

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

    reduce_buckets_mem_seq_4 #(
        .ADDR_W(ADDR_W),
        .DATA_W(DATA_W)
    ) dut (
        .clk          (clk),
        .rst_n        (rst_n),

        .start        (start),

        .busy         (busy),
        .done         (done),

        .result_x     (result_x),
        .result_y     (result_y),
        .result_z     (result_z),

        .mem_valid    (red_mem_valid),
        .mem_write_en (red_mem_write_en),
        .mem_addr     (red_mem_addr),

        .mem_wdata_x  (red_mem_wdata_x),
        .mem_wdata_y  (red_mem_wdata_y),
        .mem_wdata_z  (red_mem_wdata_z),

        .mem_ready    (mem_ready),
        .mem_rvalid   (mem_rvalid),

        .mem_rdata_x  (mem_rdata_x),
        .mem_rdata_y  (mem_rdata_y),
        .mem_rdata_z  (mem_rdata_z)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    task automatic write_bucket(
        input logic [ADDR_W-1:0] wr_addr,
        input logic [255:0]      x,
        input logic [255:0]      y,
        input logic [255:0]      z
    );
        begin
            @(negedge clk);
            tb_mem_valid    = 1'b1;
            tb_mem_write_en = 1'b1;
            tb_mem_addr     = wr_addr;
            tb_mem_wdata_x  = x;
            tb_mem_wdata_y  = y;
            tb_mem_wdata_z  = z;

            @(posedge clk);

            @(negedge clk);
            tb_mem_valid    = 1'b0;
            tb_mem_write_en = 1'b0;
            tb_mem_addr     = '0;
            tb_mem_wdata_x  = '0;
            tb_mem_wdata_y  = '0;
            tb_mem_wdata_z  = '0;
        end
    endtask

    initial begin
        start = 1'b0;

        tb_drive_mem     = 1'b1;
        tb_mem_valid     = 1'b0;
        tb_mem_write_en  = 1'b0;
        tb_mem_addr      = '0;
        tb_mem_wdata_x   = '0;
        tb_mem_wdata_y   = '0;
        tb_mem_wdata_z   = '0;

        rst_n = 1'b0;
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        $display("[TB] Starting reduce_buckets_mem_seq_4 test");

        // Preload memory with the same result produced by bucket_build_mem_seq_4:
        // bucket1 = 2G Jacobian
        // bucket2 = 2G affine
        // bucket3 = 3G affine

        write_bucket(4'd1, EXP_2G_JAC_X, EXP_2G_JAC_Y, EXP_2G_JAC_Z);
        write_bucket(4'd2, G2_AFF_X,     G2_AFF_Y,     ONE_M);
        write_bucket(4'd3, G3_AFF_X,     G3_AFF_Y,     ONE_M);

        tb_drive_mem = 1'b0;
        repeat (2) @(posedge clk);

        @(negedge clk);
        start = 1'b1;

        @(negedge clk);
        start = 1'b0;

        wait (done === 1'b1);

        if (result_x !== EXP_15G_X || result_y !== EXP_15G_Y || result_z !== EXP_15G_Z) begin
            $display("[TB] reduce_buckets_mem_seq_4 FAILED");
            $display("[TB] EXPECTED X = %064h", EXP_15G_X);
            $display("[TB] GOT      X = %064h", result_x);
            $display("[TB] EXPECTED Y = %064h", EXP_15G_Y);
            $display("[TB] GOT      Y = %064h", result_y);
            $display("[TB] EXPECTED Z = %064h", EXP_15G_Z);
            $display("[TB] GOT      Z = %064h", result_z);
            $fatal(1, "[TB] reduce result mismatch");
        end else begin
            $display("[TB] reduce_buckets_mem_seq_4 PASSED");
        end

        #20;
        $finish;
    end

endmodule








