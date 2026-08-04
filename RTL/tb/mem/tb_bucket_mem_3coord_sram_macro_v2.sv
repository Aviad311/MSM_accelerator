`timescale 1ns/1ps

module tb_bucket_mem_3coord_sram_macro_v2;

    localparam int ADDR_W = 13;
    localparam int DATA_W = 256;
    localparam int GEN_W  = 16;

    logic                clk;
    logic                rst_n;

    logic                valid;
    logic                write_en;
    logic [ADDR_W-1:0]   addr;

    logic [DATA_W-1:0]   wdata_x;
    logic [DATA_W-1:0]   wdata_y;
    logic [DATA_W-1:0]   wdata_z;

    logic                tag_write_en;
    logic [GEN_W-1:0]    tag_wdata;

    logic                ready;
    logic                rvalid;

    logic [DATA_W-1:0]   rdata_x;
    logic [DATA_W-1:0]   rdata_y;
    logic [DATA_W-1:0]   rdata_z;
    logic [GEN_W-1:0]    tag_rdata;

    bucket_mem_3coord_sram_macro_v2 #(
        .ADDR_W          (ADDR_W),
        .DATA_W          (DATA_W),
        .DEPTH           (8192),
        .SRAM_RD_LATENCY (1),
        .GEN_W           (GEN_W)
    ) dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .valid        (valid),
        .write_en     (write_en),
        .addr         (addr),
        .wdata_x      (wdata_x),
        .wdata_y      (wdata_y),
        .wdata_z      (wdata_z),
        .tag_write_en (tag_write_en),
        .tag_wdata    (tag_wdata),
        .ready        (ready),
        .rvalid       (rvalid),
        .rdata_x      (rdata_x),
        .rdata_y      (rdata_y),
        .rdata_z      (rdata_z),
        .tag_rdata    (tag_rdata)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    task automatic write_bucket(
        input logic [ADDR_W-1:0] a,
        input logic [DATA_W-1:0] x,
        input logic [DATA_W-1:0] y,
        input logic [DATA_W-1:0] z,
        input logic [GEN_W-1:0]  tag
    );
        begin
            @(negedge clk);
            valid        = 1'b1;
            write_en     = 1'b1;
            tag_write_en = 1'b1;
            addr         = a;
            wdata_x      = x;
            wdata_y      = y;
            wdata_z      = z;
            tag_wdata    = tag;

            @(negedge clk);
            valid        = 1'b0;
            write_en     = 1'b0;
            tag_write_en = 1'b0;
            addr         = '0;
            wdata_x      = '0;
            wdata_y      = '0;
            wdata_z      = '0;
            tag_wdata    = '0;
        end
    endtask

    task automatic read_check(
        input logic [ADDR_W-1:0] a,
        input logic [DATA_W-1:0] exp_x,
        input logic [DATA_W-1:0] exp_y,
        input logic [DATA_W-1:0] exp_z,
        input logic [GEN_W-1:0]  exp_tag
    );
        begin
            @(negedge clk);
            valid        = 1'b1;
            write_en     = 1'b0;
            tag_write_en = 1'b0;
            addr         = a;

            @(posedge clk);
            #0.1;

            if (ready !== 1'b1)
                $fatal(1, "[TB_BUCKET_V2] ready is not asserted.");

            if (rvalid !== 1'b1)
                $fatal(1, "[TB_BUCKET_V2] rvalid missing for addr=%0d", a);

            if (rdata_x !== exp_x ||
                rdata_y !== exp_y ||
                rdata_z !== exp_z ||
                tag_rdata !== exp_tag) begin

                $display("[TB_BUCKET_V2] READ FAILED addr=%0d", a);
                $display("[TB_BUCKET_V2] expected X   = %064h", exp_x);
                $display("[TB_BUCKET_V2] got      X   = %064h", rdata_x);
                $display("[TB_BUCKET_V2] expected Y   = %064h", exp_y);
                $display("[TB_BUCKET_V2] got      Y   = %064h", rdata_y);
                $display("[TB_BUCKET_V2] expected Z   = %064h", exp_z);
                $display("[TB_BUCKET_V2] got      Z   = %064h", rdata_z);
                $display("[TB_BUCKET_V2] expected tag = %04h", exp_tag);
                $display("[TB_BUCKET_V2] got      tag = %04h", tag_rdata);

                $fatal(1, "[TB_BUCKET_V2] Bucket memory mismatch.");
            end

            $display(
                "[TB_BUCKET_V2] READ PASSED addr=%0d tag=%04h",
                a,
                tag_rdata
            );

            @(negedge clk);
            valid = 1'b0;
            addr  = '0;
        end
    endtask

    localparam logic [255:0] X0 =
        256'h0000000000000000000000000000000000000000000000000000000000001111;
    localparam logic [255:0] Y0 =
        256'h0000000000000000000000000000000000000000000000000000000000002222;
    localparam logic [255:0] Z0 =
        256'h0000000000000000000000000000000000000000000000000000000000003333;

    localparam logic [255:0] X1 =
        256'h0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef;
    localparam logic [255:0] Y1 =
        256'hfedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210;
    localparam logic [255:0] Z1 =
        256'haaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;

    localparam logic [255:0] X2 =
        256'h111122223333444455556666777788889999aaaabbbbccccddddeeeeffff0000;
    localparam logic [255:0] Y2 =
        256'h0000ffffeeeeddddccccbbbbaaaa999988887777666655554444333322221111;
    localparam logic [255:0] Z2 =
        256'h13579bdf2468ace013579bdf2468ace013579bdf2468ace013579bdf2468ace0;

    initial begin
        valid        = 1'b0;
        write_en     = 1'b0;
        addr         = '0;
        wdata_x      = '0;
        wdata_y      = '0;
        wdata_z      = '0;
        tag_write_en = 1'b0;
        tag_wdata    = '0;

        rst_n = 1'b0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        write_bucket(13'd0,    X0, Y0, Z0, 16'h0001);
        write_bucket(13'd4095, X1, Y1, Z1, 16'h1234);
        write_bucket(13'd8191, X2, Y2, Z2, 16'hBEEF);

        read_check(13'd0,    X0, Y0, Z0, 16'h0001);
        read_check(13'd4095, X1, Y1, Z1, 16'h1234);
        read_check(13'd8191, X2, Y2, Z2, 16'hBEEF);

        $display("");
        $display("[TB_BUCKET_V2] bucket_mem_3coord_sram_macro_v2 PASSED");
        $display("");

        #20;
        $finish;
    end

    initial begin
        #200000;
        $fatal(1, "[TB_BUCKET_V2] Watchdog timeout.");
    end

endmodule