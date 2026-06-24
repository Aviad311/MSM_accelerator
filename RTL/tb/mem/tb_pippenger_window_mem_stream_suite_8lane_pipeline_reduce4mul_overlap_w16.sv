`timescale 1ns/1ps

// ============================================================================
// File:
//   tb/mem/
//   tb_pippenger_window_mem_stream_suite_8lane_pipeline_reduce4mul_overlap_w16.sv
//
// Purpose:
//   Directed W=16 smoke regression for the verified 8-lane build engine and
//   overlapped 4-multiplier reduce engine.
//
// Configuration:
//   Global bucket width : 16 bits
//   Global buckets      : 65,536
//   SRAM banks          : 8
//   Local address width : 13 bits
//   Buckets per bank    : 8,192
//
// Vector file:
//   vectors/window_suite_w16_smoke.svh
//
// The existing W=8 regression is intentionally left unchanged.
// ============================================================================

module tb_pippenger_window_mem_stream_suite_8lane_pipeline_reduce4mul_overlap_w16;

    parameter int ADDR_W          = 16;
    parameter int DATA_W          = 256;
    parameter int DEPTH           = (1 << ADDR_W);
    parameter int SRAM_RD_LATENCY = 3;

    parameter int FIFO_DEPTH      = 16;
    parameter int SLOT_COUNT      = 16;
    parameter int MIX_CTX_COUNT   = 40;
    parameter int MUL_LATENCY     = 16;

    `include "vectors/window_suite_w16_smoke.svh"

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;

    logic                in_valid;
    logic                in_ready;
    logic [ADDR_W-1:0]   in_bucket_id;
    logic [DATA_W-1:0]   in_point_x;
    logic [DATA_W-1:0]   in_point_y;
    logic                last_point;

    logic [DATA_W-1:0] result_x;
    logic [DATA_W-1:0] result_y;
    logic [DATA_W-1:0] result_z;

    longint unsigned cycle_cnt;
    longint unsigned start_cycle;
    longint unsigned latency;

    longint unsigned perf_total_cycles;
    longint unsigned perf_busy_cycles;
    longint unsigned perf_idle_cycles;
    longint unsigned perf_input_valid_cycles;
    longint unsigned perf_input_accept_count;
    longint unsigned perf_input_stall_cycles;
    longint unsigned perf_expected_points;

    logic [255:0]      x_dyn [];
    logic [255:0]      y_dyn [];
    logic [ADDR_W-1:0] b_dyn [];

    pippenger_window_mem_stream_top_8lane_pipeline_reduce4mul_overlap #(
        .ADDR_W          (ADDR_W),
        .DATA_W          (DATA_W),
        .DEPTH           (DEPTH),
        .SRAM_RD_LATENCY (SRAM_RD_LATENCY),
        .GEN_W           (16),
        .FIFO_DEPTH      (FIFO_DEPTH),
        .SLOT_COUNT      (SLOT_COUNT),
        .MIX_CTX_COUNT   (MIX_CTX_COUNT),
        .MUL_LATENCY     (MUL_LATENCY)
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

    initial clk = 1'b0;
    always #5 clk = ~clk;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cycle_cnt <= 0;
        else
            cycle_cnt <= cycle_cnt + 1;
    end

    task automatic load_suite_case_to_dynamic_arrays(
        input int case_idx
    );
        int n;

        begin
            n = suite_num_points[case_idx];

            x_dyn = new[n];
            y_dyn = new[n];
            b_dyn = new[n];

            for (int i = 0; i < n; i = i + 1) begin
                x_dyn[i] = suite_point_x[case_idx][i];
                y_dyn[i] = suite_point_y[case_idx][i];
                b_dyn[i] = suite_bucket_idx[case_idx][i];
            end
        end
    endtask

    task automatic run_case_stream(
        input string test_name,
        input logic [255:0] x_stream[],
        input logic [255:0] y_stream[],
        input logic [ADDR_W-1:0] b_stream[],
        input logic [255:0] exp_x,
        input logic [255:0] exp_y,
        input logic [255:0] exp_z
    );
        int stream_size;

        begin
            stream_size = x_stream.size();

            if (stream_size <= 0) begin
                $fatal(
                    1,
                    "[TB_W16] %s has no input points.",
                    test_name
                );
            end

            @(posedge clk);
            wait (busy === 1'b0);
            @(negedge clk);

            perf_total_cycles       = 0;
            perf_busy_cycles        = 0;
            perf_idle_cycles        = 0;
            perf_input_valid_cycles = 0;
            perf_input_accept_count = 0;
            perf_input_stall_cycles = 0;
            perf_expected_points    = stream_size;

            in_valid     = 1'b0;
            last_point   = 1'b0;
            in_bucket_id = '0;
            in_point_x   = '0;
            in_point_y   = '0;

            start_cycle = cycle_cnt;
            start       = 1'b1;

            @(negedge clk);
            start = 1'b0;

            fork
                begin : PERF_MONITOR
                    while (done !== 1'b1) begin
                        @(posedge clk);

                        perf_total_cycles++;

                        if (busy)
                            perf_busy_cycles++;
                        else
                            perf_idle_cycles++;

                        if (in_valid)
                            perf_input_valid_cycles++;

                        if (in_valid && in_ready)
                            perf_input_accept_count++;

                        if (in_valid && !in_ready)
                            perf_input_stall_cycles++;
                    end
                end
            join_none

            for (int i = 0; i < stream_size; i = i + 1) begin
                @(negedge clk);

                in_point_x   = x_stream[i];
                in_point_y   = y_stream[i];
                in_bucket_id = b_stream[i];
                in_valid     = 1'b1;
                last_point   = (i == stream_size - 1);

                do begin
                    @(posedge clk);
                end while (!in_ready);

                @(negedge clk);

                in_valid   = 1'b0;
                last_point = 1'b0;
            end

            wait (done === 1'b1);
            latency = cycle_cnt - start_cycle;

            if (result_x !== exp_x ||
                result_y !== exp_y ||
                result_z !== exp_z) begin

                $display("");
                $display(
                    "[TB_W16] %s FAILED",
                    test_name
                );

                $display(
                    "[TB_W16] EXPECTED X = %064h",
                    exp_x
                );
                $display(
                    "[TB_W16] GOT      X = %064h",
                    result_x
                );

                $display(
                    "[TB_W16] EXPECTED Y = %064h",
                    exp_y
                );
                $display(
                    "[TB_W16] GOT      Y = %064h",
                    result_y
                );

                $display(
                    "[TB_W16] EXPECTED Z = %064h",
                    exp_z
                );
                $display(
                    "[TB_W16] GOT      Z = %064h",
                    result_z
                );

                $fatal(
                    1,
                    "[TB_W16] W=16 smoke-suite mismatch."
                );
            end else begin
                $display(
                    "[TB_W16] %s PASSED latency=%0d cycles",
                    test_name,
                    latency
                );

                $display(
                    "[PERF_W16] %s expected_points    = %0d",
                    test_name,
                    perf_expected_points
                );

                $display(
                    "[PERF_W16] %s total_cycles       = %0d",
                    test_name,
                    perf_total_cycles
                );

                $display(
                    "[PERF_W16] %s busy_cycles        = %0d",
                    test_name,
                    perf_busy_cycles
                );

                $display(
                    "[PERF_W16] %s idle_cycles        = %0d",
                    test_name,
                    perf_idle_cycles
                );

                $display(
                    "[PERF_W16] %s input_valid_cycles = %0d",
                    test_name,
                    perf_input_valid_cycles
                );

                $display(
                    "[PERF_W16] %s input_accept_count = %0d",
                    test_name,
                    perf_input_accept_count
                );

                $display(
                    "[PERF_W16] %s input_stall_cycles = %0d",
                    test_name,
                    perf_input_stall_cycles
                );

                if (perf_input_accept_count != perf_expected_points) begin
                    $fatal(
                        1,
                        "[TB_W16] %s accepted-count mismatch: expected=%0d got=%0d",
                        test_name,
                        perf_expected_points,
                        perf_input_accept_count
                    );
                end
            end

            @(posedge clk);
            repeat (3) @(posedge clk);
        end
    endtask

    initial begin
        start        = 1'b0;
        in_valid     = 1'b0;
        in_bucket_id = '0;
        in_point_x   = '0;
        in_point_y   = '0;
        last_point   = 1'b0;

        rst_n = 1'b0;
        repeat (5) @(posedge clk);
        rst_n = 1'b1;

        // The DUT performs a one-time generation-tag initialization over
        // all 8,192 local addresses in all eight SRAM banks in parallel.
        wait (busy === 1'b0);
        repeat (2) @(posedge clk);

        $display("");
        $display(
            "[TB_W16] Starting W=16 8-LANE PIPELINE + REDUCE 4MUL OVERLAP smoke suite"
        );

        $display(
            "[TB_W16] Configuration: ADDR_W=%0d DEPTH=%0d SRAM_RD_LATENCY=%0d",
            ADDR_W,
            DEPTH,
            SRAM_RD_LATENCY
        );

        $display(
            "[TB_W16] Bank configuration: NUM_BANKS=8 BANK_ADDR_W=%0d BANK_DEPTH=%0d",
            ADDR_W - 3,
            1 << (ADDR_W - 3)
        );

        $display(
            "[TB_W16] Suite: cases=%0d max_points=%0d suite_addr_w=%0d suite_depth=%0d",
            SUITE_NUM_CASES,
            SUITE_MAX_POINTS,
            SUITE_ADDR_W,
            SUITE_DEPTH
        );

        if (ADDR_W != 16) begin
            $fatal(
                1,
                "[TB_W16] This testbench requires ADDR_W=16, got %0d.",
                ADDR_W
            );
        end

        if (SUITE_ADDR_W != ADDR_W) begin
            $fatal(
                1,
                "[TB_W16] Suite ADDR_W mismatch: suite=%0d tb=%0d",
                SUITE_ADDR_W,
                ADDR_W
            );
        end

        if (SUITE_DEPTH != DEPTH) begin
            $fatal(
                1,
                "[TB_W16] Suite DEPTH mismatch: suite=%0d tb=%0d",
                SUITE_DEPTH,
                DEPTH
            );
        end

        for (
            int case_idx = 0;
            case_idx < SUITE_NUM_CASES;
            case_idx = case_idx + 1
        ) begin
            $display("");
            $display(
                "[TB_W16] =================================================="
            );

            $display(
                "[TB_W16] Running case %0d/%0d: %s",
                case_idx + 1,
                SUITE_NUM_CASES,
                suite_case_name[case_idx]
            );

            $display(
                "[TB_W16] Python stats: num_points=%0d direct_write=%0d mixed_add=%0d skipped_zero=%0d reduce_scan=%0d reduce_active=%0d",
                suite_num_points[case_idx],
                suite_direct_write_count[case_idx],
                suite_mixed_add_count[case_idx],
                suite_skipped_zero_count[case_idx],
                suite_reduce_scan_count[case_idx],
                suite_reduce_active_count[case_idx]
            );

            $display(
                "[TB_W16] =================================================="
            );

            if (suite_reduce_scan_count[case_idx] != DEPTH - 1) begin
                $fatal(
                    1,
                    "[TB_W16] Case %s has invalid reduce_scan_count=%0d expected=%0d",
                    suite_case_name[case_idx],
                    suite_reduce_scan_count[case_idx],
                    DEPTH - 1
                );
            end

            load_suite_case_to_dynamic_arrays(case_idx);

            run_case_stream(
                suite_case_name[case_idx],
                x_dyn,
                y_dyn,
                b_dyn,
                suite_expected_x[case_idx],
                suite_expected_y[case_idx],
                suite_expected_z[case_idx]
            );
        end

        $display("");
        $display(
            "[TB_W16] All W=16 8-lane PIPELINE + REDUCE 4MUL OVERLAP smoke cases PASSED."
        );

        #20;
        $finish;
    end

    // 2 seconds of simulated time at a 10 ns clock gives a generous
    // 200-million-cycle upper bound for the complete directed suite.
    initial begin
        #2000000000;

        $display("");
        $display(
            "[TB_W16] ERROR: W=16 smoke-suite watchdog timeout!"
        );

        $fatal(
            1,
            "[TB_W16] Simulation exceeded the watchdog limit."
        );
    end

endmodule
