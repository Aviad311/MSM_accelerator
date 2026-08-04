`timescale 1ns/1ps

// ============================================================================
// File:
//   tb/mem/tb_pippenger_window_param_lanes_build_perf_v1.sv
//
// Build-only performance test for the parameterized 1/2/4/8-lane SRAM-macro
// Pippenger window top.
//
// The test intentionally stops as soon as Bucket Build is complete, before
// the long 65,536-bucket Reduce scan.  This isolates the scaling of:
//   input stream -> scheduler -> per-lane FIFO -> bucket update pipeline
//   -> mixed-add pipeline -> SRAM writeback.
//
// No giant vector file is required. Bucket IDs are generated on the fly.
//
// TEST_MODE:
//   0 = uniform permutation over all 65,536 bucket IDs
//       (bucket zero is remapped to bucket one)
//   1 = balanced hot-LANES: one hot bucket per active lane
//   2 = fixed hot8: buckets 1..8, independent of LANES
//   3 = hot1: all points target bucket 1
//   4 = skew80_hot64: 80% of traffic targets buckets 1..64,
//       while 20% is spread over the complete bucket space
//
// Recommended first comparison:
//   NUM_POINTS_VALUE = 262144
//   TEST_MODE_VALUE  = 0
//   LANES_VALUE      = 1,2,4,8
// ============================================================================

module tb_pippenger_window_param_lanes_build_perf_v2 #(
    parameter int LANES_VALUE      = 8,
    parameter int NUM_POINTS_VALUE = 262144,
    parameter int TEST_MODE_VALUE  = 0
);

    localparam int ADDR_W = 16;
    localparam int DATA_W = 256;

    localparam logic [255:0] G_AFF_X_M =
        256'h9981e643e9089f48979f48c033fd129c231e295329bc66dbd7362e5a487e2097;

    localparam logic [255:0] G_AFF_Y_M =
        256'hcf3f851fd4a582d670b6b59aac19c1368dfc5d5d1f1dc64db15ea6d2d3dbabe2;

    // Three distinct valid secp256k1 affine points in Montgomery domain.
    // Cycling them avoids the artificial case in which every bucket's second
    // update is G+G and must use the shared special-case doubler.
    localparam logic [255:0] G2_AFF_X_M =
        256'hF918623CCBA0EE23CE0B62E1E014040471354AFC88B285A04E0640C981048D2C;

    localparam logic [255:0] G2_AFF_Y_M =
        256'h3C7F7712157B93134B3A0F64BDA2CC6584FD25167DC75CE17D12D622FFACCFBF;

    localparam logic [255:0] G3_AFF_X_M =
        256'h9497730FCDF4C0AD5940D07385985972066CEAFB22EB7BC42379D4BBD5FEA781;

    localparam logic [255:0] G3_AFF_Y_M =
        256'h3EC28DCD9215EC76CC6048BD84885650AC4964CDC5A1F91FAF18B0B0613F55A9;

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
    longint unsigned start_cycle;
    longint unsigned first_accept_cycle;
    longint unsigned last_accept_cycle;
    longint unsigned build_done_cycle;

    longint unsigned offered_cycles;
    longint unsigned accepted_count;
    longint unsigned input_stall_cycles;
    longint unsigned send_index;

    logic first_accept_seen;

    wire input_fire = in_valid && in_ready;

    function automatic logic [ADDR_W-1:0]
        bucket_for_index(input longint unsigned idx);
        longint unsigned raw;
        longint unsigned hot_count;
        begin
            case (TEST_MODE_VALUE)
                0: begin
                    // 40503 is odd, so multiplication modulo 2^16 produces
                    // a full permutation of all 65,536 bucket IDs.
                    raw = ((idx * 40503) + 17) & 16'hffff;

                    // Keep this performance test focused on real updates.
                    // Global bucket zero was already functionally verified.
                    if (raw == 0)
                        raw = 1;

                    bucket_for_index = raw[ADDR_W-1:0];
                end

                1: begin
                    // Buckets 1..LANES map to distinct active lanes for every
                    // supported power-of-two LANES value.
                    hot_count = LANES_VALUE;
                    raw = (idx % hot_count) + 1;
                    bucket_for_index = raw[ADDR_W-1:0];
                end

                2: begin
                    raw = (idx % 8) + 1;
                    bucket_for_index = raw[ADDR_W-1:0];
                end

                4: begin
                    // Deterministic 80/20 skew:
                    // four out of every five points target 64 hot buckets;
                    // the fifth point uses the full-space permutation.
                    if ((idx % 5) != 4) begin
                        raw = (idx % 64) + 1;
                    end else begin
                        raw = ((idx * 40503) + 17) & 16'hffff;
                        if (raw == 0)
                            raw = 1;
                    end
                    bucket_for_index = raw[ADDR_W-1:0];
                end

                default: begin
                    bucket_for_index = 16'd1;
                end
            endcase
        end
    endfunction

    function automatic logic [DATA_W-1:0]
        point_x_for_index(input longint unsigned idx);
        begin
            case (idx % 3)
                0: point_x_for_index = G_AFF_X_M;
                1: point_x_for_index = G2_AFF_X_M;
                default: point_x_for_index = G3_AFF_X_M;
            endcase
        end
    endfunction

    function automatic logic [DATA_W-1:0]
        point_y_for_index(input longint unsigned idx);
        begin
            case (idx % 3)
                0: point_y_for_index = G_AFF_Y_M;
                1: point_y_for_index = G2_AFF_Y_M;
                default: point_y_for_index = G3_AFF_Y_M;
            endcase
        end
    endfunction

    function automatic string mode_name(input int mode);
        begin
            case (mode)
                0: mode_name = "uniform";
                1: mode_name = "balanced_hot_lanes";
                2: mode_name = "fixed_hot8";
                3: mode_name = "hot1";
                4: mode_name = "skew80_hot64";
                default: mode_name = "unknown";
            endcase
        end
    endfunction

    initial clk = 1'b0;
    always #5ns clk = ~clk;

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

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count        <= 0;
            offered_cycles     <= 0;
            accepted_count     <= 0;
            input_stall_cycles <= 0;
            send_index         <= 0;
            first_accept_cycle <= 0;
            last_accept_cycle  <= 0;
            first_accept_seen  <= 1'b0;
            in_valid           <= 1'b0;
            in_bucket_id       <= '0;
            in_point_x         <= G_AFF_X_M;
            in_point_y         <= G_AFF_Y_M;
            last_point         <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;

            if (in_valid) begin
                offered_cycles <= offered_cycles + 1;

                if (!in_ready)
                    input_stall_cycles <= input_stall_cycles + 1;
            end

            if (input_fire) begin
                accepted_count <= accepted_count + 1;

                if (!first_accept_seen) begin
                    first_accept_seen  <= 1'b1;
                    first_accept_cycle <= cycle_count;
                end

                last_accept_cycle <= cycle_count;

                if (send_index == NUM_POINTS_VALUE-1) begin
                    in_valid   <= 1'b0;
                    last_point <= 1'b0;
                end else begin
                    send_index   <= send_index + 1;
                    in_bucket_id <= bucket_for_index(send_index + 1);
                    in_point_x   <= point_x_for_index(send_index + 1);
                    in_point_y   <= point_y_for_index(send_index + 1);
                    last_point   <=
                        (send_index + 1 == NUM_POINTS_VALUE-1);
                end
            end
        end
    end

    initial begin : test_sequence
        real accept_rate;
        real effective_ii;
        real points_per_build_cycle;

        if (!((LANES_VALUE == 1) || (LANES_VALUE == 2) ||
              (LANES_VALUE == 4) || (LANES_VALUE == 8))) begin
            $fatal(1, "LANES_VALUE must be 1, 2, 4, or 8");
        end

        if (NUM_POINTS_VALUE <= 0)
            $fatal(1, "NUM_POINTS_VALUE must be positive");

        rst_n = 1'b0;
        start = 1'b0;

        repeat (8) @(posedge clk);
        rst_n = 1'b1;

        // Wait for one-time tag initialization.
        wait (!busy);
        repeat (4) @(posedge clk);

        @(negedge clk);
        start       = 1'b1;
        start_cycle = cycle_count;

        @(negedge clk);
        start = 1'b0;

        wait (in_ready);

        @(negedge clk);
        send_index   = 0;
        in_bucket_id = bucket_for_index(0);
        in_point_x   = point_x_for_index(0);
        in_point_y   = point_y_for_index(0);
        last_point   = (NUM_POINTS_VALUE == 1);
        in_valid     = 1'b1;

        // Internal build_done becomes true only after the last accepted update
        // has completed and written back. Stop before Reduce consumes time.
        wait (dut.build_done);
        build_done_cycle = cycle_count;

        repeat (2) @(posedge clk);

        if (accepted_count != NUM_POINTS_VALUE) begin
            $fatal(
                1,
                "[BUILD_PERF] accepted expected=%0d got=%0d",
                NUM_POINTS_VALUE,
                accepted_count
            );
        end

        accept_rate =
            (offered_cycles == 0) ? 0.0 :
            real'(accepted_count) / real'(offered_cycles);

        effective_ii =
            (accepted_count == 0) ? 0.0 :
            real'(offered_cycles) / real'(accepted_count);

        points_per_build_cycle =
            (build_done_cycle <= first_accept_cycle) ? 0.0 :
            real'(accepted_count) /
            real'(build_done_cycle - first_accept_cycle + 1);

        $display("");
        $display("============================================================");
        $display("[BUILD_PERF] PASSED");
        $display("[BUILD_PERF] LANES                    = %0d",
                 LANES_VALUE);
        $display("[BUILD_PERF] mode                     = %s",
                 mode_name(TEST_MODE_VALUE));
        $display("[BUILD_PERF] points                   = %0d",
                 NUM_POINTS_VALUE);
        $display("[BUILD_PERF] start_cycle              = %0d",
                 start_cycle);
        $display("[BUILD_PERF] first_accept_cycle       = %0d",
                 first_accept_cycle);
        $display("[BUILD_PERF] last_accept_cycle        = %0d",
                 last_accept_cycle);
        $display("[BUILD_PERF] build_done_cycle         = %0d",
                 build_done_cycle);
        $display("[BUILD_PERF] build_from_first_accept  = %0d",
                 build_done_cycle - first_accept_cycle + 1);
        $display("[BUILD_PERF] offered_cycles           = %0d",
                 offered_cycles);
        $display("[BUILD_PERF] accepted_count           = %0d",
                 accepted_count);
        $display("[BUILD_PERF] input_stall_cycles       = %0d",
                 input_stall_cycles);
        $display("[BUILD_PERF] accept_rate              = %0.9f",
                 accept_rate);
        $display("[BUILD_PERF] effective_input_ii       = %0.9f",
                 effective_ii);
        $display("[BUILD_PERF] points_per_build_cycle   = %0.9f",
                 points_per_build_cycle);
        $display("[BUILD_PERF] scheduler enqueue        = %0d",
                 dut.scheduler_total_enqueue_count);
        $display("[BUILD_PERF] scheduler issue          = %0d",
                 dut.scheduler_total_issue_count);
        $display("[BUILD_PERF] scheduler completed      = %0d",
                 dut.scheduler_total_completed_count);
        $display("[BUILD_PERF] scheduler bypass         = %0d",
                 dut.scheduler_total_bypass_count);
        $display("[BUILD_PERF] scheduler FIFO-full stall= %0d",
                 dut.scheduler_total_fifo_full_stall_count);
        $display("[BUILD_PERF] direct writes            = %0d",
                 dut.scheduler_total_direct_write_count);
        $display("[BUILD_PERF] mixed adds               = %0d",
                 dut.scheduler_total_mixed_add_count);
        $display("============================================================");

        $finish;
    end

    initial begin : watchdog
        #5s;
        $fatal(
            1,
            "[BUILD_PERF] WATCHDOG LANES=%0d mode=%0d accepted=%0d",
            LANES_VALUE,
            TEST_MODE_VALUE,
            accepted_count
        );
    end

endmodule