`timescale 1ns/1ps

// ============================================================================
// File:
//   tb/mem/tb_pippenger_window_param_lanes_functional_v1.sv
//
// Functional equivalence smoke test for LANES=1/2/4/8.
//
// The parameterized SRAM-macro top is compared cycle-independently against
// the already verified fixed 8-lane SRAM-macro top.
//
// Both DUTs receive exactly the same accepted update stream.  The test covers:
//   - global bucket 0 skip
//   - direct writes
//   - repeated-bucket mixed additions
//   - buckets in every 8192-entry physical SRAM slice
//   - low bucket IDs that map to different lanes
//   - highest global bucket
//   - complete reduction
//   - exact final Jacobian Montgomery X/Y/Z equivalence
//
// Override LANES_VALUE with xrun -defparam.
// ============================================================================

module tb_pippenger_window_param_lanes_functional_v1 #(
    parameter int LANES_VALUE = 1
);

    localparam int ADDR_W = 16;
    localparam int DATA_W = 256;
    localparam int NUM_UPDATES = 40;

    localparam logic [255:0] G_AFF_X_M =
        256'h9981e643e9089f48979f48c033fd129c231e295329bc66dbd7362e5a487e2097;

    localparam logic [255:0] G_AFF_Y_M =
        256'hcf3f851fd4a582d670b6b59aac19c1368dfc5d5d1f1dc64db15ea6d2d3dbabe2;

    logic clk;
    logic rst_n;
    logic start;

    logic src_valid;
    logic [ADDR_W-1:0] src_bucket;
    logic [DATA_W-1:0] src_x;
    logic [DATA_W-1:0] src_y;
    logic src_last;

    logic param_in_valid;
    logic param_in_ready;
    logic param_busy;
    logic param_done;
    logic [DATA_W-1:0] param_result_x;
    logic [DATA_W-1:0] param_result_y;
    logic [DATA_W-1:0] param_result_z;

    logic ref_in_valid;
    logic ref_in_ready;
    logic ref_busy;
    logic ref_done;
    logic [DATA_W-1:0] ref_result_x;
    logic [DATA_W-1:0] ref_result_y;
    logic [DATA_W-1:0] ref_result_z;

    logic [ADDR_W-1:0] bucket_vec [0:NUM_UPDATES-1];

    integer send_index;
    integer accepted_count;
    integer input_wait_cycles;
    longint unsigned cycle_count;
    longint unsigned start_cycle;
    longint unsigned param_done_cycle;
    longint unsigned ref_done_cycle;
    logic param_done_seen;
    logic ref_done_seen;

    wire common_ready = param_in_ready && ref_in_ready;
    wire common_fire  = src_valid && common_ready;

    // Each DUT only sees valid when the other DUT is also ready.  Therefore
    // both DUTs accept every update on exactly the same clock edge.
    assign param_in_valid = src_valid && ref_in_ready;
    assign ref_in_valid   = src_valid && param_in_ready;

    initial clk = 1'b0;
    always #5ns clk = ~clk;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cycle_count <= 0;
        else
            cycle_count <= cycle_count + 1;
    end

    // ---------------------------------------------------------------------
    // Bucket pattern.
    //
    // Repeated entries force mixed additions.  Entries separated by 8192
    // exercise each physical SRAM slice in the deep logical-bank wrapper.
    // ---------------------------------------------------------------------
    initial begin
        bucket_vec[0]  = 16'd0;
        bucket_vec[1]  = 16'd1;
        bucket_vec[2]  = 16'd1;
        bucket_vec[3]  = 16'd2;
        bucket_vec[4]  = 16'd2;
        bucket_vec[5]  = 16'd3;
        bucket_vec[6]  = 16'd4;
        bucket_vec[7]  = 16'd5;
        bucket_vec[8]  = 16'd6;
        bucket_vec[9]  = 16'd7;
        bucket_vec[10] = 16'd8;
        bucket_vec[11] = 16'd9;

        bucket_vec[12] = 16'd8192;
        bucket_vec[13] = 16'd8192;
        bucket_vec[14] = 16'd8193;
        bucket_vec[15] = 16'd16384;
        bucket_vec[16] = 16'd16384;
        bucket_vec[17] = 16'd16385;
        bucket_vec[18] = 16'd24576;
        bucket_vec[19] = 16'd24576;
        bucket_vec[20] = 16'd24577;
        bucket_vec[21] = 16'd32768;
        bucket_vec[22] = 16'd32768;
        bucket_vec[23] = 16'd32769;
        bucket_vec[24] = 16'd40960;
        bucket_vec[25] = 16'd40960;
        bucket_vec[26] = 16'd40961;
        bucket_vec[27] = 16'd49152;
        bucket_vec[28] = 16'd49152;
        bucket_vec[29] = 16'd49153;
        bucket_vec[30] = 16'd57344;
        bucket_vec[31] = 16'd57344;
        bucket_vec[32] = 16'd57345;

        bucket_vec[33] = 16'd65535;
        bucket_vec[34] = 16'd65535;
        bucket_vec[35] = 16'd65534;
        bucket_vec[36] = 16'd12345;
        bucket_vec[37] = 16'd12345;
        bucket_vec[38] = 16'd54321;
        bucket_vec[39] = 16'd54321;
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
    ) dut_param (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (start),
        .in_valid     (param_in_valid),
        .in_ready     (param_in_ready),
        .in_bucket_id (src_bucket),
        .in_point_x   (src_x),
        .in_point_y   (src_y),
        .last_point   (src_last),
        .busy         (param_busy),
        .done         (param_done),
        .result_x     (param_result_x),
        .result_y     (param_result_y),
        .result_z     (param_result_z)
    );

    pippenger_window_mem_stream_top_8lane_pipeline_reduce4mul_overlap_sram_macro_v2 #(
        .ADDR_W          (ADDR_W),
        .DATA_W          (DATA_W),
        .DEPTH           (1 << ADDR_W),
        .SRAM_RD_LATENCY (1),
        .GEN_W           (16),
        .FIFO_DEPTH      (16),
        .SLOT_COUNT      (16),
        .MIX_CTX_COUNT   (40),
        .MUL_LATENCY     (16)
    ) dut_ref (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (start),
        .in_valid     (ref_in_valid),
        .in_ready     (ref_in_ready),
        .in_bucket_id (src_bucket),
        .in_point_x   (src_x),
        .in_point_y   (src_y),
        .last_point   (src_last),
        .busy         (ref_busy),
        .done         (ref_done),
        .result_x     (ref_result_x),
        .result_y     (ref_result_y),
        .result_z     (ref_result_z)
    );

    // Stream driver.
    // This is a testbench procedural block rather than always_ff because the
    // test-sequence initial block also performs the one-time stream launch.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            send_index        <= 0;
            accepted_count    <= 0;
            input_wait_cycles <= 0;
            src_valid         <= 1'b0;
            src_bucket        <= '0;
            src_x             <= G_AFF_X_M;
            src_y             <= G_AFF_Y_M;
            src_last          <= 1'b0;
        end else begin
            if (src_valid && !common_ready)
                input_wait_cycles <= input_wait_cycles + 1;

            if (common_fire) begin
                accepted_count <= accepted_count + 1;

                if (send_index == NUM_UPDATES-1) begin
                    src_valid <= 1'b0;
                    src_last  <= 1'b0;
                end else begin
                    send_index <= send_index + 1;
                    src_bucket <= bucket_vec[send_index + 1];
                    src_last   <= (send_index + 1 == NUM_UPDATES-1);
                end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            param_done_cycle <= 0;
            ref_done_cycle   <= 0;
            param_done_seen  <= 1'b0;
            ref_done_seen    <= 1'b0;
        end else begin
            if (param_done) begin
                param_done_cycle <= cycle_count;
                param_done_seen  <= 1'b1;
            end

            if (ref_done) begin
                ref_done_cycle <= cycle_count;
                ref_done_seen  <= 1'b1;
            end
        end
    end

    initial begin : test_sequence
        rst_n = 1'b0;
        start = 1'b0;

        repeat (8) @(posedge clk);
        rst_n = 1'b1;

        // Both designs perform their own tag initialization.  LANES=1 is the
        // slowest, so wait until both are idle before launching the window.
        wait (!param_busy && !ref_busy);
        repeat (4) @(posedge clk);

        @(negedge clk);
        start = 1'b1;
        start_cycle = cycle_count;

        @(negedge clk);
        start = 1'b0;

        // Wait until both tops enter their build-wait state and expose ready.
        wait (param_in_ready && ref_in_ready);

        @(negedge clk);
        send_index = 0;
        src_bucket = bucket_vec[0];
        src_x      = G_AFF_X_M;
        src_y      = G_AFF_Y_M;
        src_last   = (NUM_UPDATES == 1);
        src_valid  = 1'b1;

        // done is a one-cycle pulse, and the parameterized DUT and the
        // fixed 8-lane reference finish at different cycles. Wait for the
        // latched observations rather than requiring simultaneous pulses.
        wait (param_done_seen && ref_done_seen);
        repeat (2) @(posedge clk);

        if (accepted_count != NUM_UPDATES) begin
            $fatal(
                1,
                "[TB_PARAM_FUNC] accepted expected=%0d got=%0d",
                NUM_UPDATES,
                accepted_count
            );
        end

        if (param_result_x !== ref_result_x) begin
            $display("[TB_PARAM_FUNC] PARAM X=%h", param_result_x);
            $display("[TB_PARAM_FUNC] REF   X=%h", ref_result_x);
            $fatal(1, "[TB_PARAM_FUNC] X mismatch LANES=%0d", LANES_VALUE);
        end

        if (param_result_y !== ref_result_y) begin
            $display("[TB_PARAM_FUNC] PARAM Y=%h", param_result_y);
            $display("[TB_PARAM_FUNC] REF   Y=%h", ref_result_y);
            $fatal(1, "[TB_PARAM_FUNC] Y mismatch LANES=%0d", LANES_VALUE);
        end

        if (param_result_z !== ref_result_z) begin
            $display("[TB_PARAM_FUNC] PARAM Z=%h", param_result_z);
            $display("[TB_PARAM_FUNC] REF   Z=%h", ref_result_z);
            $fatal(1, "[TB_PARAM_FUNC] Z mismatch LANES=%0d", LANES_VALUE);
        end

        $display("");
        $display("============================================================");
        $display(
            "[TB_PARAM_FUNC] PARAM-LANE FUNCTIONAL EQUIVALENCE PASSED"
        );
        $display("[TB_PARAM_FUNC] LANES                 = %0d", LANES_VALUE);
        $display("[TB_PARAM_FUNC] updates accepted      = %0d", accepted_count);
        $display("[TB_PARAM_FUNC] input wait cycles     = %0d",
                 input_wait_cycles);
        $display("[TB_PARAM_FUNC] param done cycle      = %0d",
                 param_done_cycle);
        $display("[TB_PARAM_FUNC] reference done cycle  = %0d",
                 ref_done_cycle);
        $display("[TB_PARAM_FUNC] verified bucket0, direct writes,");
        $display("[TB_PARAM_FUNC] mixed adds, all SRAM slices, reduce,");
        $display("[TB_PARAM_FUNC] and exact final X/Y/Z equivalence");
        $display("============================================================");

        $finish;
    end

    initial begin : watchdog
        #2s;
        $fatal(
            1,
            "[TB_PARAM_FUNC] WATCHDOG LANES=%0d accepted=%0d",
            LANES_VALUE,
            accepted_count
        );
    end

endmodule