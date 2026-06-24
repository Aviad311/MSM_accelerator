`timescale 1ns/1ps

// ============================================================================
// W=16 Build-only performance suite with back-to-back input driving.
//
// The production RTL is not modified. Each case is measured until the DUT
// raises its internal reduce_start pulse. The testbench then resets the DUT,
// aborting Reduce and reinitializing the generation tags before the next case.
//
// Vector file:
//   vectors/window_suite_w16_build_only.svh
// ============================================================================

module tb_pippenger_window_mem_stream_suite_8lane_pipeline_w16_build_only_b2b;

    parameter int ADDR_W          = 16;
    parameter int DATA_W          = 256;
    parameter int DEPTH           = (1 << ADDR_W);
    parameter int SRAM_RD_LATENCY = 3;

    `include "vectors/window_suite_w16_build_only.svh"

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
    longint unsigned build_end_cycle;
    longint unsigned build_cycles;

    longint unsigned input_valid_cycles;
    longint unsigned input_accept_count;
    longint unsigned input_stall_cycles;

    logic [255:0]      x_dyn [];
    logic [255:0]      y_dyn [];
    logic [ADDR_W-1:0] b_dyn [];

    pippenger_window_mem_stream_top_8lane_pipeline_reduce4mul_overlap #(
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

    task automatic reset_and_wait_for_idle;
        begin
            @(negedge clk);

            start        = 1'b0;
            in_valid     = 1'b0;
            in_bucket_id = '0;
            in_point_x   = '0;
            in_point_y   = '0;
            last_point   = 1'b0;

            rst_n = 1'b0;
            repeat (5) @(posedge clk);
            @(negedge clk);
            rst_n = 1'b1;

            wait (busy === 1'b0);
            repeat (2) @(posedge clk);
        end
    endtask

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

    task automatic run_build_case(
        input int case_idx
    );
        int stream_size;
        bit reduce_seen;
        real points_per_cycle;
        real cycles_per_point;
        real mixed_adds_per_cycle;

        begin
            stream_size = x_dyn.size();

            input_valid_cycles = 0;
            input_accept_count = 0;
            input_stall_cycles = 0;

            reduce_seen    = 1'b0;
            start_cycle    = cycle_cnt;
            build_end_cycle = 0;

            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;

            fork
                begin : BUILD_MONITOR
                    while (!reduce_seen) begin
                        @(posedge clk);

                        if (in_valid)
                            input_valid_cycles++;

                        if (in_valid && in_ready)
                            input_accept_count++;

                        if (in_valid && !in_ready)
                            input_stall_cycles++;

                        if (dut.reduce_start === 1'b1) begin
                            build_end_cycle = cycle_cnt;
                            reduce_seen = 1'b1;
                        end
                    end
                end
            join_none

            // Back-to-back ready/valid driver:
            // keep in_valid asserted continuously and advance to the next
            // point only on a successful handshake. This removes the
            // artificial one-cycle bubble that existed in the previous TB.
            begin : BACK_TO_BACK_DRIVER
                int i;

                i = 0;

                @(negedge clk);

                in_point_x   = x_dyn[i];
                in_point_y   = y_dyn[i];
                in_bucket_id = b_dyn[i];
                in_valid     = 1'b1;
                last_point   = (i == stream_size - 1);

                while (i < stream_size) begin
                    @(posedge clk);

                    if (in_valid && in_ready) begin
                        i = i + 1;

                        if (i < stream_size) begin
                            @(negedge clk);

                            in_point_x   = x_dyn[i];
                            in_point_y   = y_dyn[i];
                            in_bucket_id = b_dyn[i];
                            in_valid     = 1'b1;
                            last_point   = (i == stream_size - 1);
                        end else begin
                            @(negedge clk);

                            in_valid   = 1'b0;
                            last_point = 1'b0;
                        end
                    end
                end
            end

            wait (reduce_seen == 1'b1);

            build_cycles = build_end_cycle - start_cycle;

            if (input_accept_count != stream_size) begin
                $fatal(
                    1,
                    "[TB_W16_BUILD_B2B] Accepted-count mismatch: expected=%0d got=%0d",
                    stream_size,
                    input_accept_count
                );
            end

            points_per_cycle = real'(stream_size) / real'(build_cycles);
            cycles_per_point = real'(build_cycles) / real'(stream_size);

            if (suite_mixed_add_count[case_idx] != 0)
                mixed_adds_per_cycle =
                    real'(suite_mixed_add_count[case_idx]) /
                    real'(build_cycles);
            else
                mixed_adds_per_cycle = 0.0;

            $display("");
            $display(
                "[TB_W16_BUILD_B2B] %s BUILD COMPLETED",
                suite_case_name[case_idx]
            );
            $display(
                "[PERF_W16_BUILD_B2B] %s expected_points      = %0d",
                suite_case_name[case_idx],
                stream_size
            );
            $display(
                "[PERF_W16_BUILD_B2B] %s active_bucket_count  = %0d",
                suite_case_name[case_idx],
                suite_active_bucket_count[case_idx]
            );
            $display(
                "[PERF_W16_BUILD_B2B] %s direct_write_count   = %0d",
                suite_case_name[case_idx],
                suite_direct_write_count[case_idx]
            );
            $display(
                "[PERF_W16_BUILD_B2B] %s mixed_add_count      = %0d",
                suite_case_name[case_idx],
                suite_mixed_add_count[case_idx]
            );
            $display(
                "[PERF_W16_BUILD_B2B] %s build_cycles         = %0d",
                suite_case_name[case_idx],
                build_cycles
            );
            $display(
                "[PERF_W16_BUILD_B2B] %s points_per_cycle     = %0.6f",
                suite_case_name[case_idx],
                points_per_cycle
            );
            $display(
                "[PERF_W16_BUILD_B2B] %s cycles_per_point     = %0.6f",
                suite_case_name[case_idx],
                cycles_per_point
            );
            $display(
                "[PERF_W16_BUILD_B2B] %s mixed_adds_per_cycle = %0.6f",
                suite_case_name[case_idx],
                mixed_adds_per_cycle
            );
            $display(
                "[PERF_W16_BUILD_B2B] %s input_valid_cycles   = %0d",
                suite_case_name[case_idx],
                input_valid_cycles
            );
            $display(
                "[PERF_W16_BUILD_B2B] %s input_accept_count   = %0d",
                suite_case_name[case_idx],
                input_accept_count
            );
            $display(
                "[PERF_W16_BUILD_B2B] %s input_stall_cycles   = %0d",
                suite_case_name[case_idx],
                input_stall_cycles
            );

            // Abort the Reduce phase and cleanly reinitialize before
            // starting the next Build-only measurement.
            reset_and_wait_for_idle();
        end
    endtask

    initial begin
        start        = 1'b0;
        in_valid     = 1'b0;
        in_bucket_id = '0;
        in_point_x   = '0;
        in_point_y   = '0;
        last_point   = 1'b0;
        rst_n        = 1'b1;

        reset_and_wait_for_idle();

        $display("");
        $display(
            "[TB_W16_BUILD_B2B] Starting W=16 Build-only BACK-TO-BACK dense-workload suite"
        );
        $display(
            "[TB_W16_BUILD_B2B] ADDR_W=%0d DEPTH=%0d cases=%0d",
            ADDR_W,
            DEPTH,
            SUITE_NUM_CASES
        );

        if (SUITE_ADDR_W != ADDR_W)
            $fatal(1, "[TB_W16_BUILD_B2B] ADDR_W mismatch.");

        if (SUITE_DEPTH != DEPTH)
            $fatal(1, "[TB_W16_BUILD_B2B] DEPTH mismatch.");

        for (int case_idx = 0;
             case_idx < SUITE_NUM_CASES;
             case_idx++) begin

            $display("");
            $display(
                "[TB_W16_BUILD_B2B] Running case %0d/%0d: %s",
                case_idx + 1,
                SUITE_NUM_CASES,
                suite_case_name[case_idx]
            );

            load_case(case_idx);
            run_build_case(case_idx);
        end

        $display("");
        $display(
            "[TB_W16_BUILD_B2B] All W=16 Build-only cases COMPLETED."
        );

        #20;
        $finish;
    end

    initial begin
        #1_000_000_000;

        $fatal(
            1,
            "[TB_W16_BUILD_B2B] Watchdog timeout."
        );
    end

endmodule
