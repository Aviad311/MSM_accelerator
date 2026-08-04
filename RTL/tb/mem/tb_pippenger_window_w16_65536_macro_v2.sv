`timescale 1ns/1ps

// ============================================================================
// W=16 performance suite with separate Build and Reduce timing.
//
// Vector file:
//   vectors/window_suite_w16_perf.svh
//
// Cases:
//   - 1024 uniform
//   - 1024 hot8
//   - 8192 uniform
//
// The testbench uses simulation-only hierarchical observation of the DUT's
// internal reduce_start pulse. No production RTL interface is changed.
// ============================================================================

module tb_pippenger_window_w16_65536_macro_v2;

    parameter int ADDR_W          = 16;
    parameter int DATA_W          = 256;
    parameter int DEPTH           = (1 << ADDR_W);
    parameter int SRAM_RD_LATENCY = 1;

    `include "vectors/window_suite_w16_dense_perf.svh"

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
    longint unsigned reduce_start_cycle;
    longint unsigned done_cycle;

    longint unsigned build_cycles;
    longint unsigned reduce_cycles;
    longint unsigned total_cycles;

    longint unsigned input_valid_cycles;
    longint unsigned input_accept_count;
    longint unsigned input_stall_cycles;

    logic [255:0]      x_dyn [];
    logic [255:0]      y_dyn [];
    logic [ADDR_W-1:0] b_dyn [];

    pippenger_window_mem_stream_top_8lane_pipeline_reduce4mul_overlap_sram_macro_v2 #(
        .ADDR_W          (ADDR_W),
        .DATA_W          (DATA_W),
        .DEPTH           (DEPTH),
        .SRAM_RD_LATENCY (SRAM_RD_LATENCY),
        .GEN_W           (16),
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

    task automatic load_case(
        input int case_idx
    );
        int n;

        begin
            n = suite_num_points[case_idx];

            x_dyn = new[n];
            y_dyn = new[n];
            b_dyn = new[n];

            for (int i = 0; i < n; i++) begin
                x_dyn[i] = suite_point_x[case_idx][i];
                y_dyn[i] = suite_point_y[case_idx][i];
                b_dyn[i] = suite_bucket_idx[case_idx][i];
            end
        end
    endtask

    task automatic run_case(
        input int case_idx
    );
        int stream_size;
        bit reduce_seen;

        begin
            stream_size = x_dyn.size();

            wait (busy === 1'b0);
            @(negedge clk);

            input_valid_cycles = 0;
            input_accept_count = 0;
            input_stall_cycles = 0;

            start_cycle        = cycle_cnt;
            reduce_start_cycle = 0;
            done_cycle         = 0;
            reduce_seen        = 0;

            start = 1'b1;
            @(negedge clk);
            start = 1'b0;

            fork
                begin : PHASE_MONITOR
                    while (done !== 1'b1) begin
                        @(posedge clk);

                        if (!reduce_seen && dut.reduce_start === 1'b1) begin
                            reduce_start_cycle = cycle_cnt;
                            reduce_seen = 1;
                        end

                        if (in_valid)
                            input_valid_cycles++;

                        if (in_valid && in_ready)
                            input_accept_count++;

                        if (in_valid && !in_ready)
                            input_stall_cycles++;
                    end
                end
            join_none

            // Continuous valid/ready streaming:
            // keep in_valid asserted and advance to the next item only
            // after a successful handshake. This removes the artificial
            // one-cycle bubble that existed in the original performance TB.
            begin : CONTINUOUS_STREAM
                int i;

                i = 0;

                @(negedge clk);
                in_valid     = 1'b1;
                in_point_x   = x_dyn[0];
                in_point_y   = y_dyn[0];
                in_bucket_id = b_dyn[0];
                last_point   = (stream_size == 1);

                while (i < stream_size) begin
                    @(posedge clk);

                    if (in_valid && in_ready) begin
                        i++;

                        if (i < stream_size) begin
                            @(negedge clk);
                            in_point_x   = x_dyn[i];
                            in_point_y   = y_dyn[i];
                            in_bucket_id = b_dyn[i];
                            last_point   = (i == stream_size - 1);
                        end else begin
                            @(negedge clk);
                            in_valid   = 1'b0;
                            last_point = 1'b0;
                        end
                    end
                end
            end

            wait (done === 1'b1);
            done_cycle = cycle_cnt;

            if (!reduce_seen) begin
                $fatal(
                    1,
                    "[TB_W16_65536_MACRO_V2] reduce_start was not observed."
                );
            end

            build_cycles  = reduce_start_cycle - start_cycle;
            reduce_cycles = done_cycle - reduce_start_cycle;
            total_cycles  = done_cycle - start_cycle;

            if (result_x !== suite_expected_x[case_idx] ||
                result_y !== suite_expected_y[case_idx] ||
                result_z !== suite_expected_z[case_idx]) begin

                $display(
                    "[TB_W16_65536_MACRO_V2] %s FAILED",
                    suite_case_name[case_idx]
                );
                $display(
                    "[TB_W16_65536_MACRO_V2] EXPECTED X = %064h",
                    suite_expected_x[case_idx]
                );
                $display(
                    "[TB_W16_65536_MACRO_V2] GOT      X = %064h",
                    result_x
                );
                $display(
                    "[TB_W16_65536_MACRO_V2] EXPECTED Y = %064h",
                    suite_expected_y[case_idx]
                );
                $display(
                    "[TB_W16_65536_MACRO_V2] GOT      Y = %064h",
                    result_y
                );
                $display(
                    "[TB_W16_65536_MACRO_V2] EXPECTED Z = %064h",
                    suite_expected_z[case_idx]
                );
                $display(
                    "[TB_W16_65536_MACRO_V2] GOT      Z = %064h",
                    result_z
                );

                $fatal(
                    1,
                    "[TB_W16_65536_MACRO_V2] Result mismatch."
                );
            end

            if (input_accept_count != stream_size) begin
                $fatal(
                    1,
                    "[TB_W16_65536_MACRO_V2] Accepted-count mismatch: expected=%0d got=%0d",
                    stream_size,
                    input_accept_count
                );
            end

            $display("");
            $display(
                "[TB_W16_65536_MACRO_V2] %s PASSED",
                suite_case_name[case_idx]
            );
            $display(
                "[PERF_W16_65536_MACRO_V2] %s expected_points    = %0d",
                suite_case_name[case_idx],
                stream_size
            );
            $display(
                "[PERF_W16_65536_MACRO_V2] %s direct_write       = %0d",
                suite_case_name[case_idx],
                suite_direct_write_count[case_idx]
            );
            $display(
                "[PERF_W16_65536_MACRO_V2] %s mixed_add          = %0d",
                suite_case_name[case_idx],
                suite_mixed_add_count[case_idx]
            );
            $display(
                "[PERF_W16_65536_MACRO_V2] %s reduce_active      = %0d",
                suite_case_name[case_idx],
                suite_reduce_active_count[case_idx]
            );
            $display(
                "[PERF_W16_65536_MACRO_V2] %s build_cycles       = %0d",
                suite_case_name[case_idx],
                build_cycles
            );
            $display(
                "[PERF_W16_65536_MACRO_V2] %s reduce_cycles      = %0d",
                suite_case_name[case_idx],
                reduce_cycles
            );
            $display(
                "[PERF_W16_65536_MACRO_V2] %s total_cycles       = %0d",
                suite_case_name[case_idx],
                total_cycles
            );
            $display(
                "[PERF_W16_65536_MACRO_V2] %s build_percent      = %0.2f",
                suite_case_name[case_idx],
                100.0 * build_cycles / total_cycles
            );
            $display(
                "[PERF_W16_65536_MACRO_V2] %s reduce_percent     = %0.2f",
                suite_case_name[case_idx],
                100.0 * reduce_cycles / total_cycles
            );
            $display(
                "[PERF_W16_65536_MACRO_V2] %s input_valid_cycles = %0d",
                suite_case_name[case_idx],
                input_valid_cycles
            );
            $display(
                "[PERF_W16_65536_MACRO_V2] %s input_accept_count = %0d",
                suite_case_name[case_idx],
                input_accept_count
            );
            $display(
                "[PERF_W16_65536_MACRO_V2] %s input_stall_cycles = %0d",
                suite_case_name[case_idx],
                input_stall_cycles
            );

            $display(
                "[PERF_W16_65536_MACRO_V2] %s accept_rate_points_per_cycle = %0.6f",
                suite_case_name[case_idx],
                (input_valid_cycles == 0)
                    ? 0.0
                    : (1.0 * input_accept_count / input_valid_cycles)
            );
            $display(
                "[PERF_W16_65536_MACRO_V2] %s effective_input_ii = %0.6f",
                suite_case_name[case_idx],
                (input_accept_count == 0)
                    ? 0.0
                    : (1.0 * input_valid_cycles / input_accept_count)
            );

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

        $display("");
        $display(
            "[TB_W16_65536_MACRO_V2] Starting W=16 dense phase-separated performance suite"
        );
        $display(
            "[TB_W16_65536_MACRO_V2] ADDR_W=%0d DEPTH=%0d cases=%0d",
            ADDR_W,
            DEPTH,
            SUITE_NUM_CASES
        );

        if (SUITE_ADDR_W != ADDR_W)
            $fatal(1, "[TB_W16_65536_MACRO_V2] ADDR_W mismatch.");

        if (SUITE_DEPTH != DEPTH)
            $fatal(1, "[TB_W16_65536_MACRO_V2] DEPTH mismatch.");

        // Dense suite case 0 is the 65,536-point uniform workload.
        if (SUITE_NUM_CASES < 1)
            $fatal(1, "[TB_W16_65536_MACRO_V2] Dense suite is empty.");

        $display("");
        $display(
            "[TB_W16_65536_MACRO_V2] Running final macro-v2 case: %s",
            suite_case_name[0]
        );

        load_case(0);
        run_case(0);

        $display("");
        $display(
            "[TB_W16_65536_MACRO_V2] 65,536-point macro-v2 case PASSED."
        );

        #20;
        $finish;
    end

    initial begin
        #(64'd5000000000);

        $fatal(
            1,
            "[TB_W16_65536_MACRO_V2] Watchdog timeout."
        );
    end

endmodule