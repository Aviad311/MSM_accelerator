`timescale 1ns/1ps

// ============================================================================
// Internal MSM core reset-recovery verification.
//
// Phase A:
//   1. Start Job 2.
//   2. Send 25 accepted points into window 3.
//   3. Assert reset while the MSM core is still busy.
//   4. Verify that busy/done and the input interface return to reset state.
//
// Phase B:
//   1. Without clearing the SRAM contents externally, launch Job 0.
//   2. Run all four windows to completion.
//   3. Compare the final Jacobian result against the Python golden.
//
// This test bypasses AXI, CDC, and the affine-to-Montgomery frontend.
// It directly verifies the internal compute core that is intended for synthesis.
// ============================================================================

module tb_msm_multiwindow_controller_reset_recovery_v1;

    localparam int ADDR_W          = 16;
    localparam int DATA_W          = 256;
    localparam int DEPTH           = (1 << ADDR_W);
    localparam int SRAM_RD_LATENCY = 1;
    localparam int GEN_W           = 16;
    localparam int FIFO_DEPTH      = 16;
    localparam int SLOT_COUNT      = 16;
    localparam int MIX_CTX_COUNT   = 40;
    localparam int MUL_LATENCY     = 16;
    localparam int WINDOW_BITS     = 16;
    localparam int NUM_WINDOWS     = 4;

    localparam int ABORT_JOB            = 2;
    localparam int RECOVERY_JOB         = 0;
    localparam int ABORT_ACCEPTED_POINTS = 25;

    localparam time CLK_PERIOD = 4ns;

    `include "vectors/msm_internal_back_to_back_suite_v1.svh"

    logic clk;
    logic rst_n;
    logic start;

    logic in_valid;
    logic in_ready;
    logic [ADDR_W-1:0] in_bucket_id;
    logic [DATA_W-1:0] in_point_x;
    logic [DATA_W-1:0] in_point_y;
    logic last_point;

    logic [$clog2(NUM_WINDOWS)-1:0] window_index;
    logic busy;
    logic done;

    logic [DATA_W-1:0] result_x;
    logic [DATA_W-1:0] result_y;
    logic [DATA_W-1:0] result_z;

    integer error_count;
    longint unsigned cycle_count;

    initial begin
        clk = 1'b0;
    end

    always #(CLK_PERIOD / 2) begin
        clk = ~clk;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 0;
        end
        else begin
            cycle_count <= cycle_count + 1;
        end
    end

    msm_multiwindow_controller_v1 #(
        .ADDR_W          (ADDR_W),
        .DATA_W          (DATA_W),
        .DEPTH           (DEPTH),
        .SRAM_RD_LATENCY (SRAM_RD_LATENCY),
        .GEN_W           (GEN_W),
        .FIFO_DEPTH      (FIFO_DEPTH),
        .SLOT_COUNT      (SLOT_COUNT),
        .MIX_CTX_COUNT   (MIX_CTX_COUNT),
        .MUL_LATENCY     (MUL_LATENCY),
        .WINDOW_BITS     (WINDOW_BITS),
        .NUM_WINDOWS     (NUM_WINDOWS)
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
        .window_index (window_index),
        .busy         (busy),
        .done         (done),
        .result_x     (result_x),
        .result_y     (result_y),
        .result_z     (result_z)
    );

    task automatic pulse_start;
        begin
            @(negedge clk);
            start = 1'b1;

            @(negedge clk);
            start = 1'b0;
        end
    endtask

    task automatic send_one_point(
        input int job,
        input int window,
        input int point,
        input int point_count
    );
        begin
            in_bucket_id = b2b_bucket[job][window][point];
            in_point_x   = b2b_point_x[job][point];
            in_point_y   = b2b_point_y[job][point];
            last_point   = (point == point_count - 1);
            in_valid     = 1'b1;

            do begin
                @(posedge clk);
            end
            while (!in_ready);

            @(negedge clk);
            in_valid   = 1'b0;
            last_point = 1'b0;
        end
    endtask

    task automatic apply_reset;
        begin
            @(negedge clk);
            in_valid   = 1'b0;
            last_point = 1'b0;
            start      = 1'b0;
            rst_n      = 1'b0;

            repeat (8) begin
                @(posedge clk);
            end

            @(negedge clk);
            rst_n = 1'b1;

            repeat (4) begin
                @(posedge clk);
            end
        end
    endtask

    task automatic run_complete_job(
        input int job
    );
        integer window;
        integer point;
        integer point_count;
        longint unsigned start_cycle;

        begin
            point_count = b2b_job_point_count[job];

            while (busy) begin
                @(posedge clk);
            end

            start_cycle = cycle_count;
            pulse_start();

            $display(
                "[RESET_RECOVERY_TB] RECOVERY START job=%0d points=%0d cycle=%0d",
                job,
                point_count,
                start_cycle
            );

            for (
                window = NUM_WINDOWS - 1;
                window >= 0;
                window = window - 1
            ) begin
                wait (busy && (window_index == window));

                for (
                    point = 0;
                    point < point_count;
                    point = point + 1
                ) begin
                    send_one_point(
                        job,
                        window,
                        point,
                        point_count
                    );
                end

                $display(
                    "[RESET_RECOVERY_TB] job=%0d window=%0d input complete cycle=%0d",
                    job,
                    window,
                    cycle_count
                );

                if (window > 0) begin
                    wait ((window_index != window) || done);
                end
            end

            wait (done);
            @(posedge clk);

            if (
                (result_x !== b2b_expected_x[job])
                || (result_y !== b2b_expected_y[job])
                || (result_z !== b2b_expected_z[job])
            ) begin
                error_count = error_count + 1;

                $display(
                    "[RESET_RECOVERY_TB] ERROR recovery result mismatch"
                );
                $display(
                    "[RESET_RECOVERY_TB] expected X=%064h",
                    b2b_expected_x[job]
                );
                $display(
                    "[RESET_RECOVERY_TB] actual   X=%064h",
                    result_x
                );
                $display(
                    "[RESET_RECOVERY_TB] expected Y=%064h",
                    b2b_expected_y[job]
                );
                $display(
                    "[RESET_RECOVERY_TB] actual   Y=%064h",
                    result_y
                );
                $display(
                    "[RESET_RECOVERY_TB] expected Z=%064h",
                    b2b_expected_z[job]
                );
                $display(
                    "[RESET_RECOVERY_TB] actual   Z=%064h",
                    result_z
                );
            end
            else begin
                $display(
                    "[RESET_RECOVERY_TB] PASS recovery job=%0d latency=%0d cycles",
                    job,
                    cycle_count - start_cycle
                );
            end
        end
    endtask

    initial begin : test_sequence
        integer point;
        integer abort_point_count;

        error_count = 0;

        rst_n        = 1'b0;
        start        = 1'b0;
        in_valid     = 1'b0;
        in_bucket_id = '0;
        in_point_x   = '0;
        in_point_y   = '0;
        last_point   = 1'b0;

        if (B2B_ADDR_W != ADDR_W) begin
            $fatal(1, "[RESET_RECOVERY_TB] vector ADDR_W mismatch");
        end

        if (B2B_WINDOW_BITS != WINDOW_BITS) begin
            $fatal(1, "[RESET_RECOVERY_TB] vector WINDOW_BITS mismatch");
        end

        if (B2B_NUM_WINDOWS != NUM_WINDOWS) begin
            $fatal(1, "[RESET_RECOVERY_TB] vector NUM_WINDOWS mismatch");
        end

        repeat (8) begin
            @(posedge clk);
        end

        rst_n = 1'b1;

        repeat (4) begin
            @(posedge clk);
        end

        abort_point_count = b2b_job_point_count[ABORT_JOB];

        pulse_start();

        $display(
            "[RESET_RECOVERY_TB] ABORT PHASE START job=%0d window=%0d",
            ABORT_JOB,
            NUM_WINDOWS - 1
        );

        wait (busy && (window_index == NUM_WINDOWS - 1));

        for (
            point = 0;
            point < ABORT_ACCEPTED_POINTS;
            point = point + 1
        ) begin
            send_one_point(
                ABORT_JOB,
                NUM_WINDOWS - 1,
                point,
                abort_point_count
            );
        end

        if (!busy) begin
            error_count = error_count + 1;
            $display(
                "[RESET_RECOVERY_TB] ERROR core was not busy before reset"
            );
        end

        $display(
            "[RESET_RECOVERY_TB] asserting reset after %0d accepted points cycle=%0d",
            ABORT_ACCEPTED_POINTS,
            cycle_count
        );

        apply_reset();

        if (busy !== 1'b0) begin
            error_count = error_count + 1;
            $display(
                "[RESET_RECOVERY_TB] ERROR busy did not clear after reset"
            );
        end

        if (done !== 1'b0) begin
            error_count = error_count + 1;
            $display(
                "[RESET_RECOVERY_TB] ERROR done was high after reset"
            );
        end

        if (in_valid !== 1'b0) begin
            error_count = error_count + 1;
            $display(
                "[RESET_RECOVERY_TB] ERROR input valid remained high after reset"
            );
        end

        $display(
            "[RESET_RECOVERY_TB] reset recovery state check complete cycle=%0d",
            cycle_count
        );

        run_complete_job(RECOVERY_JOB);

        if (error_count == 0) begin
            $display(
                "============================================================"
            );
            $display(
                "[RESET_RECOVERY_TB] RESET-DURING-BUILD RECOVERY PASSED"
            );
            $display(
                "[RESET_RECOVERY_TB] aborted_job=%0d accepted_before_reset=%0d recovery_job=%0d",
                ABORT_JOB,
                ABORT_ACCEPTED_POINTS,
                RECOVERY_JOB
            );
            $display(
                "============================================================"
            );
        end
        else begin
            $fatal(
                1,
                "[RESET_RECOVERY_TB] FAILED errors=%0d",
                error_count
            );
        end

        $finish;
    end

    initial begin : watchdog
        #(1s);
        $fatal(1, "[RESET_RECOVERY_TB] WATCHDOG TIMEOUT");
    end

endmodule