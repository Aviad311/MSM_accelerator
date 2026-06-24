`timescale 1ns/1ps

module tb_bucket_mem_3coord;

    localparam int ADDR_W = 4;
    localparam int DATA_W = 256;
    localparam int DEPTH  = (1 << ADDR_W);

    localparam logic [255:0] ZERO  =
        256'h0000000000000000000000000000000000000000000000000000000000000000;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    localparam logic [255:0] GX_M =
        256'h9981E643E9089F48979F48C033FD129C231E295329BC66DBD7362E5A487E2097;

    localparam logic [255:0] GY_M =
        256'hCF3F851FD4A582D670B6B59AAC19C1368DFC5D5D1F1DC64DB15EA6D2D3DBABE2;

    localparam logic [255:0] GZ_M = ONE_M;

    localparam logic [255:0] G2_X =
        256'hF918623CCBA0EE23CE0B62E1E014040471354AFC88B285A04E0640C981048D2C;

    localparam logic [255:0] G2_Y =
        256'h3C7F7712157B93134B3A0F64BDA2CC6584FD25167DC75CE17D12D622FFACCFBF;

    localparam logic [255:0] G2_Z = ONE_M;

    logic clk;
    logic rst_n;

    logic                valid;
    logic                write_en;
    logic [ADDR_W-1:0]   addr;

    logic [DATA_W-1:0]   wdata_x;
    logic [DATA_W-1:0]   wdata_y;
    logic [DATA_W-1:0]   wdata_z;

    logic                ready;
    logic                rvalid;

    logic [DATA_W-1:0]   rdata_x;
    logic [DATA_W-1:0]   rdata_y;
    logic [DATA_W-1:0]   rdata_z;

    bucket_mem_3coord #(
        .ADDR_W(ADDR_W),
        .DATA_W(DATA_W),
        .DEPTH (DEPTH)
    ) dut (
        .clk     (clk),
        .rst_n   (rst_n),

        .valid   (valid),
        .write_en(write_en),
        .addr    (addr),

        .wdata_x (wdata_x),
        .wdata_y (wdata_y),
        .wdata_z (wdata_z),

        .ready   (ready),
        .rvalid  (rvalid),

        .rdata_x (rdata_x),
        .rdata_y (rdata_y),
        .rdata_z (rdata_z)
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
            valid    = 1'b1;
            write_en = 1'b1;
            addr     = wr_addr;
            wdata_x  = x;
            wdata_y  = y;
            wdata_z  = z;

            if (!ready) begin
                $fatal(1, "[TB] Memory not ready during write addr=%0d", wr_addr);
            end

            @(posedge clk);

            @(negedge clk);
            valid    = 1'b0;
            write_en = 1'b0;
            addr     = '0;
            wdata_x  = '0;
            wdata_y  = '0;
            wdata_z  = '0;
        end
    endtask

    task automatic read_and_check_bucket(
        input string             test_name,
        input logic [ADDR_W-1:0] rd_addr,
        input logic [255:0]      exp_x,
        input logic [255:0]      exp_y,
        input logic [255:0]      exp_z
    );
        begin
            @(negedge clk);
            valid    = 1'b1;
            write_en = 1'b0;
            addr     = rd_addr;
            wdata_x  = '0;
            wdata_y  = '0;
            wdata_z  = '0;

            if (!ready) begin
                $fatal(1, "[TB] Memory not ready during read addr=%0d", rd_addr);
            end

            @(posedge clk);

            @(negedge clk);
            valid    = 1'b0;
            write_en = 1'b0;
            addr     = '0;

            if (!rvalid) begin
                $fatal(1, "[TB] %s FAILED: rvalid was not asserted", test_name);
            end

            if (rdata_x !== exp_x || rdata_y !== exp_y || rdata_z !== exp_z) begin
                $display("[TB] %s FAILED", test_name);
                $display("[TB] EXPECTED X = %064h", exp_x);
                $display("[TB] GOT      X = %064h", rdata_x);
                $display("[TB] EXPECTED Y = %064h", exp_y);
                $display("[TB] GOT      Y = %064h", rdata_y);
                $display("[TB] EXPECTED Z = %064h", exp_z);
                $display("[TB] GOT      Z = %064h", rdata_z);
                $fatal(1, "[TB] Bucket memory mismatch");
            end else begin
                $display("[TB] %s PASSED addr=%0d", test_name, rd_addr);
            end
        end
    endtask

    initial begin
        valid    = 1'b0;
        write_en = 1'b0;
        addr     = '0;
        wdata_x  = '0;
        wdata_y  = '0;
        wdata_z  = '0;

        rst_n = 1'b0;
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        $display("[TB] Starting bucket_mem_3coord test");

        write_bucket(4'd3, GX_M, GY_M, GZ_M);
        read_and_check_bucket("write_read_G_bucket3", 4'd3, GX_M, GY_M, GZ_M);

        write_bucket(4'd7, ZERO, ONE_M, ZERO);
        read_and_check_bucket("write_read_INF_bucket7", 4'd7, ZERO, ONE_M, ZERO);

        write_bucket(4'd3, G2_X, G2_Y, G2_Z);
        read_and_check_bucket("overwrite_bucket3_with_2G", 4'd3, G2_X, G2_Y, G2_Z);

        read_and_check_bucket("bucket7_still_INF", 4'd7, ZERO, ONE_M, ZERO);

        $display("[TB] tb_bucket_mem_3coord PASSED");
        #20;
        $finish;
    end

endmodule