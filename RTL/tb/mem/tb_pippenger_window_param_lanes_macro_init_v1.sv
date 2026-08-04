`timescale 1ns/1ps

// Compile/elaboration and tag-initialization smoke test for one selected lane
// count. Override LANES_VALUE with xrun -defparam.
module tb_pippenger_window_param_lanes_macro_init_v1 #(
    parameter int LANES_VALUE = 8
);

    localparam int ADDR_W = 16;
    localparam int DATA_W = 256;

    logic clk;
    logic rst_n;
    logic start;

    logic in_valid;
    logic in_ready;
    logic [ADDR_W-1:0] in_bucket_id;
    logic [DATA_W-1:0] in_point_x;
    logic [DATA_W-1:0] in_point_y;
    logic last_point;

    logic busy;
    logic done;
    logic [DATA_W-1:0] result_x;
    logic [DATA_W-1:0] result_y;
    logic [DATA_W-1:0] result_z;

    longint unsigned cycle_count;

    initial clk = 1'b0;
    always #5ns clk = ~clk;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cycle_count <= 0;
        else
            cycle_count <= cycle_count + 1;
    end

    pippenger_window_mem_stream_top_param_lanes_sram_macro_v1 #(
        .ADDR_W          (ADDR_W),
        .DATA_W          (DATA_W),
        .DEPTH           (1 << ADDR_W),
        .SRAM_RD_LATENCY (1),
        .GEN_W           (16),
        .LANES           (LANES_VALUE),
        .FIFO_DEPTH      (16),
        .SLOT_COUNT      (16),
        .MIX_CTX_COUNT   (40),
        .MUL_LATENCY     (16)
    ) dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (start),
        .in_valid     (in_valid),
        .in_ready     (in_ready),
        .in_bucket_id (in_bucket_id),
        .in_point_x   (in_point_x),
        .in_point_y   (in_point_y),
        .last_point   (last_point),
        .busy         (busy),
        .done         (done),
        .result_x     (result_x),
        .result_y     (result_y),
        .result_z     (result_z)
    );

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        in_valid = 1'b0;
        in_bucket_id = '0;
        in_point_x = '0;
        in_point_y = '0;
        last_point = 1'b0;

        repeat (8) @(posedge clk);
        rst_n = 1'b1;

        // The top starts in S_TAG_INIT and clears all logical tag addresses.
        wait (busy == 1'b0);
        repeat (4) @(posedge clk);

        $display("");
        $display("============================================================");
        $display("[TB_PARAM_INIT] PASSED LANES=%0d", LANES_VALUE);
        $display("[TB_PARAM_INIT] tag-init cycles=%0d", cycle_count);
        $display("[TB_PARAM_INIT] logical buckets=65536");
        $display("[TB_PARAM_INIT] total physical 8192-entry slices=8");
        $display("[TB_PARAM_INIT] expected total SRAM macros=104");
        $display("============================================================");
        $finish;
    end

    initial begin
        #2ms;
        $fatal(1, "[TB_PARAM_INIT] WATCHDOG LANES=%0d", LANES_VALUE);
    end

endmodule