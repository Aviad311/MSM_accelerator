

`timescale 1ns/1ps

// ============================================================================
// File:
//   tb/mem/tb_pippenger_window_build_steady_state_stress_v2.sv
//
// Purpose:
//   Long build-only steady-state stress test for the frozen parameterized
//   SRAM-macro Pippenger window architecture.
//
// Test structure:
//   1. Start one window.
//   2. Warm up every non-zero bucket exactly once.
//      - Buckets 1..65535 become valid for the current generation.
//      - Warm-up traffic is fully drained before measurement begins.
//   3. Send a long deterministic uniform stream over buckets 1..65535.
//      - Every measured update must therefore use Mixed Add.
//   4. Stop at dut.build_done, before the full Reduce scan.
//   5. Check accounting invariants and print FIFO stability statistics.
//
// Main checks:
//   measured accepted == scheduler enqueue delta
//                     == scheduler issue delta
//                     == scheduler completed delta
//
//   measured direct writes == 0
//   measured mixed adds   == MEASURE_POINTS_VALUE
//
//   all per-lane FIFOs empty after build_done
//   no X values on important control/occupancy signals
//
// Recommended runs:
//   Smoke:
//     MEASURE_POINTS_VALUE=100000
//
//   Main:
//     MEASURE_POINTS_VALUE=4194304
//
//   Final:
//     MEASURE_POINTS_VALUE=16777216
//
// Notes:
//   - No giant SVH vector file is used.
//   - Points are selected from G, 2G, and 3G using an independent hash.
//   - Bucket IDs are generated deterministically at runtime.
//   - The warm-up phase is excluded from all reported steady-state counters.
// ============================================================================

module tb_pippenger_window_build_steady_state_stress_v1 #(
    parameter int LANES_VALUE          = 8,
    parameter int MEASURE_POINTS_VALUE = 4194304,
    parameter int FIFO_DEPTH_VALUE     = 16,
    parameter int CHECKPOINT_INTERVAL  = 0
);

    localparam int ADDR_W = 16;
    localparam int DATA_W = 256;

    localparam int NUM_NONZERO_BUCKETS = (1 << ADDR_W) - 1;
    localparam int WARMUP_POINTS       = NUM_NONZERO_BUCKETS;
    localparam int TOTAL_POINTS        =
        WARMUP_POINTS + MEASURE_POINTS_VALUE;

    localparam int FIFO_OCC_W = $clog2(FIFO_DEPTH_VALUE + 1);
    localparam int LANE_BITS  = (LANES_VALUE <= 1) ? 1 : $clog2(LANES_VALUE);

    localparam logic [255:0] G_AFF_X_M =
        256'h9981e643e9089f48979f48c033fd129c231e295329bc66dbd7362e5a487e2097;

    localparam logic [255:0] G_AFF_Y_M =
        256'hcf3f851fd4a582d670b6b59aac19c1368dfc5d5d1f1dc64db15ea6d2d3dbabe2;

    localparam logic [255:0] G2_AFF_X_M =
        256'hF918623CCBA0EE23CE0B62E1E014040471354AFC88B285A04E0640C981048D2C;

    localparam logic [255:0] G2_AFF_Y_M =
        256'h3C7F7712157B93134B3A0F64BDA2CC6584FD25167DC75CE17D12D622FFACCFBF;

    localparam logic [255:0] G3_AFF_X_M =
        256'h9497730FCDF4C0AD5940D07385985972066CEAFB22EB7BC42379D4BBD5FEA781;

    localparam logic [255:0] G3_AFF_Y_M =
        256'h3EC28DCD9215EC76CC6048BD84885650AC4964CDC5A1F91FAF18B0B0613F55A9;

    typedef enum logic [2:0] {
        PH_IDLE,
        PH_WARMUP,
        PH_WARMUP_DRAIN,
        PH_MEASURE,
        PH_FINAL_DRAIN,
        PH_DONE
    } phase_t;

    logic clk;
    logic rst_n;
    logic start;

    logic                  in_valid;
    logic                  in_ready;
    logic [ADDR_W-1:0]     in_bucket_id;
    logic [DATA_W-1:0]     in_point_x;
    logic [DATA_W-1:0]     in_point_y;
    logic                  last_point;

    logic                  busy;
    logic                  done;
    logic [DATA_W-1:0]     result_x;
    logic [DATA_W-1:0]     result_y;
    logic [DATA_W-1:0]     result_z;

    phase_t phase;

    longint unsigned cycle_count;
    longint unsigned start_cycle;
    longint unsigned warmup_first_accept_cycle;
    longint unsigned warmup_last_accept_cycle;
    longint unsigned warmup_drain_cycle;
    longint unsigned measure_first_accept_cycle;
    longint unsigned measure_last_accept_cycle;
    longint unsigned build_done_cycle;

    longint unsigned warmup_index;
    longint unsigned measure_index;
    longint unsigned warmup_accepted_count;
    longint unsigned measured_accepted_count;

    longint unsigned measure_offered_cycles;
    longint unsigned measure_stall_cycles;
    longint unsigned current_stall_run;
    longint unsigned longest_stall_run;

    longint unsigned checkpoint_next;

    localparam int SEGMENT_COUNT = 5;
    longint unsigned segment_end [0:SEGMENT_COUNT-1];
    longint unsigned segment_start_accept [0:SEGMENT_COUNT-1];
    longint unsigned segment_start_cycle  [0:SEGMENT_COUNT-1];
    longint unsigned segment_start_stall  [0:SEGMENT_COUNT-1];
    logic segment_reported [0:SEGMENT_COUNT-1];

    longint unsigned enqueue_base;
    longint unsigned issue_base;
    longint unsigned completed_base;
    longint unsigned bypass_base;
    longint unsigned fifo_full_stall_base;
    longint unsigned direct_write_base;
    longint unsigned mixed_add_base;

    longint unsigned lane_accept_count [0:LANES_VALUE-1];
    longint unsigned lane_occ_sum      [0:LANES_VALUE-1];
    longint unsigned lane_occ_samples  [0:LANES_VALUE-1];
    longint unsigned lane_occ_max      [0:LANES_VALUE-1];

    longint unsigned lane_occ_sum_first_half  [0:LANES_VALUE-1];
    longint unsigned lane_occ_samples_first   [0:LANES_VALUE-1];
    longint unsigned lane_occ_sum_second_half [0:LANES_VALUE-1];
    longint unsigned lane_occ_samples_second  [0:LANES_VALUE-1];

    integer lane_i;
    integer final_i;

    wire input_fire = in_valid && in_ready;

    // ------------------------------------------------------------------------
    // Deterministic point and bucket generation
    // ------------------------------------------------------------------------

    function automatic logic [ADDR_W-1:0]
        warmup_bucket_for_index(input longint unsigned idx);
        longint unsigned raw;
        begin
            // idx=0..65534 maps exactly to global buckets 1..65535.
            raw = idx + 1;
            warmup_bucket_for_index = raw[ADDR_W-1:0];
        end
    endfunction

    function automatic logic [ADDR_W-1:0]
        measure_bucket_for_index(input longint unsigned idx);
        longint unsigned raw;
        begin
            // 40503 is odd, so multiplication modulo 2^16 produces a full
            // permutation. Bucket zero is remapped to bucket one.
            raw = ((idx * 40503) + 17) & 16'hffff;

            if (raw == 0)
                raw = 1;

            measure_bucket_for_index = raw[ADDR_W-1:0];
        end
    endfunction

    function automatic int point_select_for_index(
        input longint unsigned idx
    );
        longint unsigned mix;
        begin
            // Independent deterministic hash used only for point selection.
            // This breaks correlation between bucket permutation, lane mapping,
            // and the G/2G/3G point selection.
            mix = (idx * 64'h9E3779B97F4A7C15) +
                  64'hD1B54A32D192ED03;
            mix = mix ^ (mix >> 29);
            mix = mix ^ (mix >> 47);

            point_select_for_index = int'(mix % 3);
        end
    endfunction

    function automatic logic [DATA_W-1:0]
        point_x_for_index(input longint unsigned idx);
        begin
            case (point_select_for_index(idx))
                0: point_x_for_index = G_AFF_X_M;
                1: point_x_for_index = G2_AFF_X_M;
                default: point_x_for_index = G3_AFF_X_M;
            endcase
        end
    endfunction

    function automatic logic [DATA_W-1:0]
        point_y_for_index(input longint unsigned idx);
        begin
            case (point_select_for_index(idx))
                0: point_y_for_index = G_AFF_Y_M;
                1: point_y_for_index = G2_AFF_Y_M;
                default: point_y_for_index = G3_AFF_Y_M;
            endcase
        end
    endfunction

    function automatic int lane_for_bucket(
        input logic [ADDR_W-1:0] bucket_id
    );
        begin
            if (LANES_VALUE == 1)
                lane_for_bucket = 0;
            else
                lane_for_bucket =
                    int'(bucket_id[LANE_BITS-1:0]);
        end
    endfunction

    // ------------------------------------------------------------------------
    // Clock and DUT
    // ------------------------------------------------------------------------

    initial clk = 1'b0;
    always #5ns clk = ~clk;

    pippenger_window_mem_stream_top_param_lanes_sram_macro_v1 #(
        .ADDR_W          (ADDR_W),
        .DATA_W          (DATA_W),
        .DEPTH           (1 << ADDR_W),
        .SRAM_RD_LATENCY (1),
        .GEN_W           (16),
        .LANES           (LANES_VALUE),
        .FIFO_DEPTH      (FIFO_DEPTH_VALUE),
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

    // ------------------------------------------------------------------------
    // Driver, counters, occupancy sampling, and phase control
    // ------------------------------------------------------------------------

    always @(posedge clk or negedge rst_n) begin : drive_and_measure
        int accepted_lane;
        longint unsigned occ_value;

        if (!rst_n) begin
            cycle_count               <= 0;
            start_cycle               <= 0;
            warmup_first_accept_cycle <= 0;
            warmup_last_accept_cycle  <= 0;
            warmup_drain_cycle        <= 0;
            measure_first_accept_cycle <= 0;
            measure_last_accept_cycle <= 0;
            build_done_cycle          <= 0;

            warmup_index          <= 0;
            measure_index         <= 0;
            warmup_accepted_count <= 0;
            measured_accepted_count <= 0;

            measure_offered_cycles <= 0;
            measure_stall_cycles   <= 0;
            current_stall_run      <= 0;
            longest_stall_run      <= 0;
            checkpoint_next        <= CHECKPOINT_INTERVAL;

            segment_end[0] <= 100000;
            segment_end[1] <= 500000;
            segment_end[2] <= 1000000;
            segment_end[3] <= 2000000;
            segment_end[4] <= MEASURE_POINTS_VALUE;

            for (lane_i = 0; lane_i < SEGMENT_COUNT; lane_i = lane_i + 1) begin
                segment_start_accept[lane_i] <= 0;
                segment_start_cycle[lane_i]  <= 0;
                segment_start_stall[lane_i]  <= 0;
                segment_reported[lane_i]     <= 1'b0;
            end

            enqueue_base          <= 0;
            issue_base            <= 0;
            completed_base        <= 0;
            bypass_base           <= 0;
            fifo_full_stall_base  <= 0;
            direct_write_base     <= 0;
            mixed_add_base        <= 0;

            phase        <= PH_IDLE;
            in_valid     <= 1'b0;
            in_bucket_id <= '0;
            in_point_x   <= G_AFF_X_M;
            in_point_y   <= G_AFF_Y_M;
            last_point   <= 1'b0;

            for (lane_i = 0; lane_i < LANES_VALUE; lane_i = lane_i + 1) begin
                lane_accept_count[lane_i] <= 0;
                lane_occ_sum[lane_i] <= 0;
                lane_occ_samples[lane_i] <= 0;
                lane_occ_max[lane_i] <= 0;

                lane_occ_sum_first_half[lane_i] <= 0;
                lane_occ_samples_first[lane_i] <= 0;
                lane_occ_sum_second_half[lane_i] <= 0;
                lane_occ_samples_second[lane_i] <= 0;
            end
        end else begin
            cycle_count <= cycle_count + 1;

            // ------------------------------------------------------------
            // Measurement-phase input stall tracking
            // ------------------------------------------------------------
            if ((phase == PH_MEASURE) && in_valid) begin
                measure_offered_cycles <= measure_offered_cycles + 1;

                if (!in_ready) begin
                    measure_stall_cycles <= measure_stall_cycles + 1;
                    current_stall_run <= current_stall_run + 1;

                    if ((current_stall_run + 1) > longest_stall_run)
                        longest_stall_run <= current_stall_run + 1;
                end else begin
                    current_stall_run <= 0;
                end
            end else begin
                current_stall_run <= 0;
            end

            // ------------------------------------------------------------
            // Per-lane FIFO occupancy sampling during measured steady state
            // ------------------------------------------------------------
            if ((phase == PH_MEASURE) || (phase == PH_FINAL_DRAIN)) begin
                for (lane_i = 0;
                     lane_i < LANES_VALUE;
                     lane_i = lane_i + 1) begin

                    if ($isunknown(
                        dut.scheduler_lane_fifo_occupancy[lane_i]
                    )) begin
                        $fatal(
                            1,
                            "[STEADY] X detected on lane %0d FIFO occupancy",
                            lane_i
                        );
                    end

                    occ_value =
                        dut.scheduler_lane_fifo_occupancy[lane_i];

                    lane_occ_sum[lane_i] <=
                        lane_occ_sum[lane_i] + occ_value;

                    lane_occ_samples[lane_i] <=
                        lane_occ_samples[lane_i] + 1;

                    if (occ_value > lane_occ_max[lane_i])
                        lane_occ_max[lane_i] <= occ_value;

                    if (measured_accepted_count <
                        (MEASURE_POINTS_VALUE / 2)) begin
                        lane_occ_sum_first_half[lane_i] <=
                            lane_occ_sum_first_half[lane_i] + occ_value;
                        lane_occ_samples_first[lane_i] <=
                            lane_occ_samples_first[lane_i] + 1;
                    end else begin
                        lane_occ_sum_second_half[lane_i] <=
                            lane_occ_sum_second_half[lane_i] + occ_value;
                        lane_occ_samples_second[lane_i] <=
                            lane_occ_samples_second[lane_i] + 1;
                    end
                end
            end

            // ------------------------------------------------------------
            // Accepted input handling
            // ------------------------------------------------------------
            if (input_fire) begin
                accepted_lane = lane_for_bucket(in_bucket_id);

                case (phase)
                    PH_WARMUP: begin
                        warmup_accepted_count <=
                            warmup_accepted_count + 1;

                        if (warmup_accepted_count == 0)
                            warmup_first_accept_cycle <= cycle_count;

                        warmup_last_accept_cycle <= cycle_count;

                        if (warmup_index == WARMUP_POINTS-1) begin
                            in_valid   <= 1'b0;
                            last_point <= 1'b0;
                            phase      <= PH_WARMUP_DRAIN;
                        end else begin
                            warmup_index <= warmup_index + 1;
                            in_bucket_id <=
                                warmup_bucket_for_index(warmup_index + 1);
                            in_point_x <=
                                point_x_for_index(warmup_index + 1);
                            in_point_y <=
                                point_y_for_index(warmup_index + 1);
                            last_point <= 1'b0;
                        end
                    end

                    PH_MEASURE: begin
                        measured_accepted_count <=
                            measured_accepted_count + 1;

                        lane_accept_count[accepted_lane] <=
                            lane_accept_count[accepted_lane] + 1;

                        if (measured_accepted_count == 0)
                            measure_first_accept_cycle <= cycle_count;

                        measure_last_accept_cycle <= cycle_count;

                        for (lane_i = 0; lane_i < SEGMENT_COUNT; lane_i = lane_i + 1) begin
                            if (!segment_reported[lane_i] &&
                                ((measured_accepted_count + 1) >= segment_end[lane_i])) begin
                                $display("");
                                $display(
                                    "[STEADY_SEGMENT] index=%0d accepted_end=%0d segment_points=%0d segment_cycles=%0d segment_stalls=%0d throughput=%0.9f effective_ii=%0.9f",
                                    lane_i,
                                    measured_accepted_count + 1,
                                    (measured_accepted_count + 1) - segment_start_accept[lane_i],
                                    cycle_count - segment_start_cycle[lane_i] + 1,
                                    measure_stall_cycles - segment_start_stall[lane_i],
                                    real'((measured_accepted_count + 1) - segment_start_accept[lane_i]) /
                                    real'(cycle_count - segment_start_cycle[lane_i] + 1),
                                    real'(cycle_count - segment_start_cycle[lane_i] + 1) /
                                    real'((measured_accepted_count + 1) - segment_start_accept[lane_i])
                                );

                                segment_reported[lane_i] <= 1'b1;

                                if (lane_i + 1 < SEGMENT_COUNT) begin
                                    segment_start_accept[lane_i + 1] <= measured_accepted_count + 1;
                                    segment_start_cycle[lane_i + 1]  <= cycle_count + 1;
                                    segment_start_stall[lane_i + 1]  <= measure_stall_cycles;
                                end
                            end
                        end

                        if ((CHECKPOINT_INTERVAL > 0) &&
                            ((measured_accepted_count + 1) >=
                             checkpoint_next)) begin

                            $display("");
                            $display(
                                "[STEADY_CHECKPOINT] accepted=%0d cycle=%0d stall=%0d longest_stall=%0d",
                                measured_accepted_count + 1,
                                cycle_count,
                                measure_stall_cycles,
                                longest_stall_run
                            );

                            for (lane_i = 0;
                                 lane_i < LANES_VALUE;
                                 lane_i = lane_i + 1) begin
                                $display(
                                    "[STEADY_CHECKPOINT] lane=%0d occ_now=%0d occ_max=%0d accepted=%0d",
                                    lane_i,
                                    dut.scheduler_lane_fifo_occupancy[lane_i],
                                    lane_occ_max[lane_i],
                                    lane_accept_count[lane_i]
                                );
                            end

                            checkpoint_next <=
                                checkpoint_next + CHECKPOINT_INTERVAL;
                        end

                        if (measure_index ==
                            MEASURE_POINTS_VALUE-1) begin
                            in_valid   <= 1'b0;
                            last_point <= 1'b0;
                            phase      <= PH_FINAL_DRAIN;
                        end else begin
                            measure_index <= measure_index + 1;
                            in_bucket_id <=
                                measure_bucket_for_index(measure_index + 1);
                            in_point_x <=
                                point_x_for_index(
                                    WARMUP_POINTS + measure_index + 1
                                );
                            in_point_y <=
                                point_y_for_index(
                                    WARMUP_POINTS + measure_index + 1
                                );
                            last_point <=
                                (measure_index + 1 ==
                                 MEASURE_POINTS_VALUE-1);
                        end
                    end

                    default: begin
                        $fatal(
                            1,
                            "[STEADY] input_fire occurred in illegal phase %0d",
                            phase
                        );
                    end
                endcase
            end

            // ------------------------------------------------------------
            // Warm-up drain completion and measurement start
            // ------------------------------------------------------------
            if ((phase == PH_WARMUP_DRAIN) &&
                (dut.scheduler_total_completed_count ==
                 WARMUP_POINTS)) begin

                warmup_drain_cycle <= cycle_count;

                enqueue_base <= dut.scheduler_total_enqueue_count;
                issue_base <= dut.scheduler_total_issue_count;
                completed_base <= dut.scheduler_total_completed_count;
                bypass_base <= dut.scheduler_total_bypass_count;
                fifo_full_stall_base <=
                    dut.scheduler_total_fifo_full_stall_count;
                direct_write_base <=
                    dut.scheduler_total_direct_write_count;
                mixed_add_base <=
                    dut.scheduler_total_mixed_add_count;

                measure_index <= 0;
                segment_start_accept[0] <= 0;
                segment_start_cycle[0]  <= cycle_count + 1;
                segment_start_stall[0]  <= 0;
                in_bucket_id <= measure_bucket_for_index(0);
                in_point_x <= point_x_for_index(WARMUP_POINTS);
                in_point_y <= point_y_for_index(WARMUP_POINTS);
                last_point <= (MEASURE_POINTS_VALUE == 1);
                in_valid <= 1'b1;
                phase <= PH_MEASURE;

                $display("");
                $display("============================================================");
                $display("[STEADY] Warm-up drained");
                $display("[STEADY] warmup points       = %0d",
                         WARMUP_POINTS);
                $display("[STEADY] warmup completed    = %0d",
                         dut.scheduler_total_completed_count);
                $display("[STEADY] measurement points  = %0d",
                         MEASURE_POINTS_VALUE);
                $display("[STEADY] measurement starts at cycle %0d",
                         cycle_count);
                $display("============================================================");
            end

            if (dut.build_done && (phase == PH_FINAL_DRAIN)) begin
                build_done_cycle <= cycle_count;
                phase <= PH_DONE;
            end
        end
    end

    // ------------------------------------------------------------------------
    // Test sequence and final checks
    // ------------------------------------------------------------------------

    initial begin : test_sequence
        real accept_rate;
        real effective_ii;
        real points_per_build_cycle;
        real lane_avg_occ;
        real lane_first_avg;
        real lane_second_avg;
        real occupancy_growth;

        longint unsigned enqueue_delta;
        longint unsigned issue_delta;
        longint unsigned completed_delta;
        longint unsigned bypass_delta;
        longint unsigned fifo_full_stall_delta;
        longint unsigned direct_write_delta;
        longint unsigned mixed_add_delta;

        if (!((LANES_VALUE == 1) || (LANES_VALUE == 2) ||
              (LANES_VALUE == 4) || (LANES_VALUE == 8))) begin
            $fatal(1, "LANES_VALUE must be 1, 2, 4, or 8");
        end

        if (MEASURE_POINTS_VALUE <= 0)
            $fatal(1, "MEASURE_POINTS_VALUE must be positive");

        if (FIFO_DEPTH_VALUE <= 0)
            $fatal(1, "FIFO_DEPTH_VALUE must be positive");

        $display("[STEADY] POINT_SELECTION = DECORRELATED_HASH_V1");
        $display("[STEADY] SEGMENTED_4M_MODE = ENABLED");

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

        // Begin full-bucket warm-up.
        @(negedge clk);
        phase        = PH_WARMUP;
        warmup_index = 0;
        in_bucket_id = warmup_bucket_for_index(0);
        in_point_x   = point_x_for_index(0);
        in_point_y   = point_y_for_index(0);
        last_point   = 1'b0;
        in_valid     = 1'b1;

        wait (phase == PH_DONE);

        // Allow nonblocking counter updates to settle.
        repeat (2) @(posedge clk);

        enqueue_delta =
            dut.scheduler_total_enqueue_count - enqueue_base;
        issue_delta =
            dut.scheduler_total_issue_count - issue_base;
        completed_delta =
            dut.scheduler_total_completed_count - completed_base;
        bypass_delta =
            dut.scheduler_total_bypass_count - bypass_base;
        fifo_full_stall_delta =
            dut.scheduler_total_fifo_full_stall_count -
            fifo_full_stall_base;
        direct_write_delta =
            dut.scheduler_total_direct_write_count -
            direct_write_base;
        mixed_add_delta =
            dut.scheduler_total_mixed_add_count -
            mixed_add_base;

        // ------------------------------------------------------------
        // Hard accounting checks
        // ------------------------------------------------------------

        if (warmup_accepted_count != WARMUP_POINTS) begin
            $fatal(
                1,
                "[STEADY] warmup accepted expected=%0d got=%0d",
                WARMUP_POINTS,
                warmup_accepted_count
            );
        end

        if (measured_accepted_count != MEASURE_POINTS_VALUE) begin
            $fatal(
                1,
                "[STEADY] measured accepted expected=%0d got=%0d",
                MEASURE_POINTS_VALUE,
                measured_accepted_count
            );
        end

        if (enqueue_delta != MEASURE_POINTS_VALUE) begin
            $fatal(
                1,
                "[STEADY] enqueue delta expected=%0d got=%0d",
                MEASURE_POINTS_VALUE,
                enqueue_delta
            );
        end

        if (issue_delta != MEASURE_POINTS_VALUE) begin
            $fatal(
                1,
                "[STEADY] issue delta expected=%0d got=%0d",
                MEASURE_POINTS_VALUE,
                issue_delta
            );
        end

        if (completed_delta != MEASURE_POINTS_VALUE) begin
            $fatal(
                1,
                "[STEADY] completed delta expected=%0d got=%0d",
                MEASURE_POINTS_VALUE,
                completed_delta
            );
        end

        if (!((measured_accepted_count == enqueue_delta) &&
              (enqueue_delta == issue_delta) &&
              (issue_delta == completed_delta))) begin
            $fatal(
                1,
                "[STEADY] accounting mismatch accepted=%0d enqueue=%0d issue=%0d completed=%0d",
                measured_accepted_count,
                enqueue_delta,
                issue_delta,
                completed_delta
            );
        end

        if (direct_write_delta != 0) begin
            $fatal(
                1,
                "[STEADY] measured direct writes expected=0 got=%0d",
                direct_write_delta
            );
        end

        if (mixed_add_delta != MEASURE_POINTS_VALUE) begin
            $fatal(
                1,
                "[STEADY] measured mixed adds expected=%0d got=%0d",
                MEASURE_POINTS_VALUE,
                mixed_add_delta
            );
        end

        for (final_i = 0;
             final_i < LANES_VALUE;
             final_i = final_i + 1) begin

            if (dut.scheduler_lane_fifo_occupancy[final_i] != 0) begin
                $fatal(
                    1,
                    "[STEADY] lane %0d FIFO not empty at end: %0d",
                    final_i,
                    dut.scheduler_lane_fifo_occupancy[final_i]
                );
            end

            if (lane_occ_max[final_i] > FIFO_DEPTH_VALUE) begin
                $fatal(
                    1,
                    "[STEADY] lane %0d illegal max occupancy=%0d depth=%0d",
                    final_i,
                    lane_occ_max[final_i],
                    FIFO_DEPTH_VALUE
                );
            end
        end

        accept_rate =
            (measure_offered_cycles == 0) ? 0.0 :
            real'(measured_accepted_count) /
            real'(measure_offered_cycles);

        effective_ii =
            (measured_accepted_count == 0) ? 0.0 :
            real'(measure_offered_cycles) /
            real'(measured_accepted_count);

        points_per_build_cycle =
            (build_done_cycle <= measure_first_accept_cycle) ? 0.0 :
            real'(measured_accepted_count) /
            real'(build_done_cycle - measure_first_accept_cycle + 1);

        // ------------------------------------------------------------
        // Final report
        // ------------------------------------------------------------

        $display("");
        $display("============================================================");
        $display("[STEADY] PASSED");
        $display("[STEADY] LANES                         = %0d",
                 LANES_VALUE);
        $display("[STEADY] FIFO_DEPTH                    = %0d",
                 FIFO_DEPTH_VALUE);
        $display("[STEADY] warmup_points                 = %0d",
                 WARMUP_POINTS);
        $display("[STEADY] measured_points               = %0d",
                 MEASURE_POINTS_VALUE);
        $display("[STEADY] total_window_points           = %0d",
                 TOTAL_POINTS);
        $display("[STEADY] start_cycle                   = %0d",
                 start_cycle);
        $display("[STEADY] warmup_first_accept_cycle     = %0d",
                 warmup_first_accept_cycle);
        $display("[STEADY] warmup_last_accept_cycle      = %0d",
                 warmup_last_accept_cycle);
        $display("[STEADY] warmup_drain_cycle            = %0d",
                 warmup_drain_cycle);
        $display("[STEADY] measure_first_accept_cycle    = %0d",
                 measure_first_accept_cycle);
        $display("[STEADY] measure_last_accept_cycle     = %0d",
                 measure_last_accept_cycle);
        $display("[STEADY] build_done_cycle              = %0d",
                 build_done_cycle);
        $display("[STEADY] measured_build_cycles         = %0d",
                 build_done_cycle - measure_first_accept_cycle + 1);
        $display("[STEADY] measure_offered_cycles        = %0d",
                 measure_offered_cycles);
        $display("[STEADY] measured_accepted_count       = %0d",
                 measured_accepted_count);
        $display("[STEADY] measure_stall_cycles          = %0d",
                 measure_stall_cycles);
        $display("[STEADY] longest_continuous_stall      = %0d",
                 longest_stall_run);
        $display("[STEADY] accept_rate                   = %0.9f",
                 accept_rate);
        $display("[STEADY] effective_input_ii            = %0.9f",
                 effective_ii);
        $display("[STEADY] points_per_build_cycle        = %0.9f",
                 points_per_build_cycle);
        $display("[STEADY] scheduler enqueue delta       = %0d",
                 enqueue_delta);
        $display("[STEADY] scheduler issue delta         = %0d",
                 issue_delta);
        $display("[STEADY] scheduler completed delta     = %0d",
                 completed_delta);
        $display("[STEADY] scheduler bypass delta        = %0d",
                 bypass_delta);
        $display("[STEADY] scheduler FIFO-full stall delta= %0d",
                 fifo_full_stall_delta);
        $display("[STEADY] direct-write delta            = %0d",
                 direct_write_delta);
        $display("[STEADY] mixed-add delta               = %0d",
                 mixed_add_delta);
        $display("------------------------------------------------------------");

        for (final_i = 0;
             final_i < LANES_VALUE;
             final_i = final_i + 1) begin

            lane_avg_occ =
                (lane_occ_samples[final_i] == 0) ? 0.0 :
                real'(lane_occ_sum[final_i]) /
                real'(lane_occ_samples[final_i]);

            lane_first_avg =
                (lane_occ_samples_first[final_i] == 0) ? 0.0 :
                real'(lane_occ_sum_first_half[final_i]) /
                real'(lane_occ_samples_first[final_i]);

            lane_second_avg =
                (lane_occ_samples_second[final_i] == 0) ? 0.0 :
                real'(lane_occ_sum_second_half[final_i]) /
                real'(lane_occ_samples_second[final_i]);

            occupancy_growth = lane_second_avg - lane_first_avg;

            $display(
                "[STEADY_LANE] lane=%0d accepted=%0d max_occ=%0d avg_occ=%0.6f first_half_avg=%0.6f second_half_avg=%0.6f growth=%0.6f final_occ=%0d",
                final_i,
                lane_accept_count[final_i],
                lane_occ_max[final_i],
                lane_avg_occ,
                lane_first_avg,
                lane_second_avg,
                occupancy_growth,
                dut.scheduler_lane_fifo_occupancy[final_i]
            );
        end

        $display("============================================================");
        $finish;
    end

    // ------------------------------------------------------------------------
    // Watchdog
    //
    // 10 seconds of simulated time at a 10 ns clock allows one billion cycles,
    // comfortably above the intended 4M/16M-point stress runs.
    // ------------------------------------------------------------------------

    initial begin : watchdog
        #10s;
        $fatal(
            1,
            "[STEADY] WATCHDOG phase=%0d warmup_accepted=%0d measured_accepted=%0d completed=%0d",
            phase,
            warmup_accepted_count,
            measured_accepted_count,
            dut.scheduler_total_completed_count
        );
    end

endmodule