`timescale 1ns/1ps

module tb_pippenger_window_mem_stream_suite_8lane_pipeline;

    parameter int ADDR_W          = 8;
    parameter int DATA_W          = 256;
    parameter int DEPTH           = (1 << ADDR_W);
    parameter int SRAM_RD_LATENCY = 3;

    `include "vectors/window_suite_w8.svh"

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

    int unsigned cycle_cnt;
    int unsigned start_cycle;
    int unsigned latency;

    int unsigned perf_total_cycles;
    int unsigned perf_busy_cycles;
    int unsigned perf_idle_cycles;
    int unsigned perf_input_valid_cycles;
    int unsigned perf_input_accept_count;
    int unsigned perf_input_stall_cycles;
    int unsigned perf_expected_points;

    logic [255:0]      x_dyn [];
    logic [255:0]      y_dyn [];
    logic [ADDR_W-1:0] b_dyn [];

    pippenger_window_mem_stream_top_8lane_pipeline #(
        .ADDR_W          (ADDR_W),
        .DATA_W          (DATA_W),
        .DEPTH           (DEPTH),
        .SRAM_RD_LATENCY (SRAM_RD_LATENCY),
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

            for (int i = 0; i < stream_size; i++) begin
                in_point_x   = x_stream[i];
                in_point_y   = y_stream[i];
                in_bucket_id = b_stream[i];
                in_valid     = 1'b1;
                last_point   = (i == stream_size - 1);

                @(posedge clk);

                while (!in_ready)
                    @(posedge clk);

                @(negedge clk);
                in_valid   = 1'b0;
                last_point = 1'b0;
            end

            wait (done === 1'b1);
            latency = cycle_cnt - start_cycle;

            if (result_x !== exp_x ||
                result_y !== exp_y ||
                result_z !== exp_z) begin

                $display("[TB] %s FAILED", test_name);
                $display("[TB] EXPECTED X = %064h", exp_x);
                $display("[TB] GOT      X = %064h", result_x);
                $display("[TB] EXPECTED Y = %064h", exp_y);
                $display("[TB] GOT      Y = %064h", result_y);
                $display("[TB] EXPECTED Z = %064h", exp_z);
                $display("[TB] GOT      Z = %064h", result_z);

                $fatal(1,
                    "[TB] 8-lane pipeline suite mismatch error.");
            end else begin
                $display(
                    "[TB] %s PASSED latency=%0d cycles",
                    test_name,
                    latency
                );

                $display(
                    "[PERF8_PIPE] %s expected_points        = %0d",
                    test_name,
                    perf_expected_points
                );

                $display(
                    "[PERF8_PIPE] %s total_cycles           = %0d",
                    test_name,
                    perf_total_cycles
                );

                $display(
                    "[PERF8_PIPE] %s busy_cycles            = %0d",
                    test_name,
                    perf_busy_cycles
                );

                $display(
                    "[PERF8_PIPE] %s idle_cycles            = %0d",
                    test_name,
                    perf_idle_cycles
                );

                $display(
                    "[PERF8_PIPE] %s input_valid_cycles     = %0d",
                    test_name,
                    perf_input_valid_cycles
                );

                $display(
                    "[PERF8_PIPE] %s input_accept_count     = %0d",
                    test_name,
                    perf_input_accept_count
                );

                $display(
                    "[PERF8_PIPE] %s input_stall_cycles     = %0d",
                    test_name,
                    perf_input_stall_cycles
                );

                if (perf_input_accept_count != perf_expected_points) begin
                    $display(
                        "[PERF8_PIPE_WARN] %s accepted mismatch: expected=%0d got=%0d",
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

        wait (busy === 1'b0);
        repeat (2) @(posedge clk);

        $display(
            "[TB] Starting 8-LANE PIPELINE SUITE validation"
        );

        $display(
            "[TB] Configuration: ADDR_W=%0d DEPTH=%0d SRAM_RD_LATENCY=%0d",
            ADDR_W,
            DEPTH,
            SRAM_RD_LATENCY
        );

        $display(
            "[TB] Suite configuration: SUITE_NUM_CASES=%0d SUITE_MAX_POINTS=%0d SUITE_ADDR_W=%0d SUITE_DEPTH=%0d",
            SUITE_NUM_CASES,
            SUITE_MAX_POINTS,
            SUITE_ADDR_W,
            SUITE_DEPTH
        );

        if (SUITE_ADDR_W != ADDR_W) begin
            $fatal(1,
                "[TB] Suite ADDR_W mismatch: suite=%0d tb=%0d",
                SUITE_ADDR_W,
                ADDR_W);
        end

        if (SUITE_DEPTH != DEPTH) begin
            $fatal(1,
                "[TB] Suite DEPTH mismatch: suite=%0d tb=%0d",
                SUITE_DEPTH,
                DEPTH);
        end

        for (int case_idx = 0;
             case_idx < SUITE_NUM_CASES;
             case_idx = case_idx + 1) begin

            $display("");
            $display(
                "[TB] =================================================="
            );

            $display(
                "[TB] Running 8-lane pipeline suite case %0d/%0d: %s",
                case_idx + 1,
                SUITE_NUM_CASES,
                suite_case_name[case_idx]
            );

            $display(
                "[TB] Python stats: num_points=%0d direct_write=%0d mixed_add=%0d skipped_zero=%0d reduce_scan=%0d reduce_active=%0d",
                suite_num_points[case_idx],
                suite_direct_write_count[case_idx],
                suite_mixed_add_count[case_idx],
                suite_skipped_zero_count[case_idx],
                suite_reduce_scan_count[case_idx],
                suite_reduce_active_count[case_idx]
            );

            $display(
                "[TB] =================================================="
            );

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
            "[TB] All 8-lane PIPELINE suite cases PASSED."
        );

        #20;
        $finish;
    end

    initial begin
        #200000000;

        $display(
            "[TB] ERROR: 8-lane pipeline suite watchdog timeout!"
        );

        $finish;
    end

endmodule