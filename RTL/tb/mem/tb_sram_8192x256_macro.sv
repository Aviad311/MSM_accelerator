`timescale 1ns/1ps

module tb_sram_8192x256_macro;


localparam int ADDR_W = 13;
localparam int DATA_W = 256;

localparam logic [255:0] DATA_ADDR_0 =
    256'h3333333333333333222222222222222211111111111111110000000000000000;

localparam logic [255:0] DATA_ADDR_1 =
    256'hFEDCBA98765432100123456789ABCDEFA5A5A5A5A5A5A5A55A5A5A5A5A5A5A5A;

localparam logic [255:0] DATA_ADDR_123 =
    256'h9981E643E9089F48979F48C033FD129C231E295329BC66DBD7362E5A487E2097;

localparam logic [255:0] DATA_ADDR_LAST =
    256'hFFFFFFFFFFFFFFFF0000000000000000DEADBEEFCAFEBABE13579BDF2468ACE0;

logic              clk;
logic              rst_n;
logic              en;
logic              we;
logic [ADDR_W-1:0] addr;
logic [DATA_W-1:0] wdata;
logic [DATA_W-1:0] rdata;
logic              rvalid;

int unsigned pass_count;
int unsigned fail_count;

sram_8192x256_macro dut (
    .clk    (clk),
    .rst_n  (rst_n),
    .en     (en),
    .we     (we),
    .addr   (addr),
    .wdata  (wdata),
    .rdata  (rdata),
    .rvalid (rvalid)
);

initial clk = 1'b0;
always #5 clk = ~clk;

task automatic write_word(
    input logic [ADDR_W-1:0] wr_addr,
    input logic [DATA_W-1:0] wr_data
);
    begin
        @(negedge clk);

        en    = 1'b1;
        we    = 1'b1;
        addr  = wr_addr;
        wdata = wr_data;

        @(posedge clk);
        #1;

        if (rvalid !== 1'b0) begin
            $fatal(
                1,
                "[TB] rvalid asserted during write: addr=%0d rvalid=%b",
                wr_addr,
                rvalid
            );
        end

        @(negedge clk);

        en    = 1'b0;
        we    = 1'b0;
        addr  = '0;
        wdata = '0;

        $display(
            "[TB] WRITE addr=%0d data=%064h",
            wr_addr,
            wr_data
        );
    end
endtask

task automatic read_and_check(
    input string             test_name,
    input logic [ADDR_W-1:0] rd_addr,
    input logic [DATA_W-1:0] expected_data
);
    begin
        @(negedge clk);

        en    = 1'b1;
        we    = 1'b0;
        addr  = rd_addr;
        wdata = '0;

        @(posedge clk);
        #1;

        if (rvalid !== 1'b1) begin
            fail_count++;

            $fatal(
                1,
                "[TB] %s missing rvalid: addr=%0d rvalid=%b",
                test_name,
                rd_addr,
                rvalid
            );
        end

        if (rdata !== expected_data) begin
            fail_count++;

            $display(
                "[TB] %s FAILED: address=%0d",
                test_name,
                rd_addr
            );
            $display("[TB] EXPECTED = %064h", expected_data);
            $display("[TB] GOT      = %064h", rdata);

            $fatal(1, "[TB] SRAM read-data mismatch");
        end

        pass_count++;

        $display(
            "[TB] %s PASSED: address=%0d data=%064h",
            test_name,
            rd_addr,
            rdata
        );

        @(negedge clk);

        en    = 1'b0;
        we    = 1'b0;
        addr  = '0;
        wdata = '0;

        @(posedge clk);
        #1;

        if (rvalid !== 1'b0) begin
            $fatal(
                1,
                "[TB] rvalid remained asserted after %s",
                test_name
            );
        end
    end
endtask

initial begin
    pass_count = 0;
    fail_count = 0;

    rst_n = 1'b0;
    en    = 1'b0;
    we    = 1'b0;
    addr  = '0;
    wdata = '0;

    repeat (5) @(posedge clk);

    @(negedge clk);
    rst_n = 1'b1;

    repeat (2) @(posedge clk);

    $display("");
    $display("============================================================");
    $display("[TB] Starting sram_8192x256_macro test");
    $display("============================================================");

    write_word(13'd0,    DATA_ADDR_0);
    write_word(13'd1,    DATA_ADDR_1);
    write_word(13'h123,  DATA_ADDR_123);
    write_word(13'd8191, DATA_ADDR_LAST);

    read_and_check(
        "read_addr_1",
        13'd1,
        DATA_ADDR_1
    );

    read_and_check(
        "read_addr_0",
        13'd0,
        DATA_ADDR_0
    );

    read_and_check(
        "read_addr_8191",
        13'd8191,
        DATA_ADDR_LAST
    );

    read_and_check(
        "read_addr_0x123",
        13'h123,
        DATA_ADDR_123
    );

    $display("");
    $display("============================================================");
    $display("[TB] sram_8192x256_macro PASSED");
    $display("[TB] Successful checks = %0d", pass_count);
    $display("[TB] Failed checks     = %0d", fail_count);
    $display("============================================================");

    #20;
    $finish;
end

initial begin
    #5000;
    $fatal(1, "[TB] WATCHDOG TIMEOUT");
end


endmodule
