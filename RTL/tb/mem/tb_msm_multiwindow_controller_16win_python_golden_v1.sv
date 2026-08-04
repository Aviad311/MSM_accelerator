`timescale 1ns/1ps

module tb_msm_multiwindow_controller_16win_python_golden_v1;

    logic clk;
    logic rst_n;
    logic start;

    logic              in_valid;
    logic              in_ready;
    logic [15:0]       in_bucket_id;
    logic [255:0]      in_point_x;
    logic [255:0]      in_point_y;
    logic              last_point;

    logic [3:0]        window_index;
    logic              busy;
    logic              done;
    logic [255:0]      result_x;
    logic [255:0]      result_y;
    logic [255:0]      result_z;

    longint unsigned cycle_count;
    longint unsigned accepted_points;
    longint unsigned input_wait_cycles;

    `include "vectors/multiwindow_w16_python_golden.svh"

    initial clk = 1'b0;
    always #5 clk = ~clk;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count     <= 0;
            accepted_points <= 0;
        end else begin
            cycle_count <= cycle_count + 1;

            if (in_valid && in_ready)
                accepted_points <= accepted_points + 1;
        end
    end

    msm_multiwindow_controller_v1 #(
        .ADDR_W          (16),
        .DATA_W          (256),
        .DEPTH           (65536),
        .SRAM_RD_LATENCY (1),
        .GEN_W           (16),
        .FIFO_DEPTH      (16),
        .SLOT_COUNT      (16),
        .MIX_CTX_COUNT   (40),
        .MUL_LATENCY     (16),
        .WINDOW_BITS     (16),
        .NUM_WINDOWS     (16)
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

    task automatic send_window(input int window);
        integer point;
        begin
            while (window_index != window[3:0] || !in_ready)
                @(posedge clk);

            for (point = 0; point < MW_POINTS_PER_WINDOW; point++) begin
                @(negedge clk);

                in_valid     = 1'b1;
                in_bucket_id = mw_bucket_idx[window][point];
                in_point_x   = mw_point_x[window][point];
                in_point_y   = mw_point_y[window][point];
                last_point   = (point == MW_POINTS_PER_WINDOW-1);

                while (!in_ready) begin
                    input_wait_cycles = input_wait_cycles + 1;
                    @(posedge clk);
                end

                @(posedge clk);

                if (window_index != window[3:0]) begin
                    $fatal(
                        1,
                        "[TB_MW_PY] Window changed during stream: expected=%0d actual=%0d point=%0d",
                        window,
                        window_index,
                        point
                    );
                end
            end

            @(negedge clk);
            in_valid     = 1'b0;
            in_bucket_id = '0;
            in_point_x   = '0;
            in_point_y   = '0;
            last_point   = 1'b0;

            $display(
                "[TB_MW_PY] Completed input window=%0d points=%0d direct=%0d mixed=%0d zero=%0d active=%0d cycle=%0d",
                window,
                MW_POINTS_PER_WINDOW,
                mw_direct_write_count[window],
                mw_mixed_add_count[window],
                mw_skipped_zero_count[window],
                mw_reduce_active_count[window],
                cycle_count
            );
        end
    endtask

    integer window;

    initial begin
        rst_n            = 1'b0;
        start            = 1'b0;
        in_valid         = 1'b0;
        in_bucket_id     = '0;
        in_point_x       = '0;
        in_point_y       = '0;
        last_point       = 1'b0;
        input_wait_cycles = 0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        for (window = MW_NUM_WINDOWS-1; window >= 0; window = window - 1)
            send_window(window);

        while (!done)
            @(posedge clk);

        #0.1;

        if (accepted_points !== MW_NUM_WINDOWS * MW_POINTS_PER_WINDOW) begin
            $fatal(
                1,
                "[TB_MW_PY] Accepted-point mismatch expected=%0d actual=%0d",
                MW_NUM_WINDOWS * MW_POINTS_PER_WINDOW,
                accepted_points
            );
        end

        if (result_x !== MW_EXPECTED_X ||
            result_y !== MW_EXPECTED_Y ||
            result_z !== MW_EXPECTED_Z) begin

            $display("[TB_MW_PY] FAILED");
            $display("[TB_MW_PY] expected X=%064h", MW_EXPECTED_X);
            $display("[TB_MW_PY] actual   X=%064h", result_x);
            $display("[TB_MW_PY] expected Y=%064h", MW_EXPECTED_Y);
            $display("[TB_MW_PY] actual   Y=%064h", result_y);
            $display("[TB_MW_PY] expected Z=%064h", MW_EXPECTED_Z);
            $display("[TB_MW_PY] actual   Z=%064h", result_z);

            $fatal(1, "[TB_MW_PY] Full Python-golden MSM mismatch.");
        end

        $display("");
        $display("============================================================");
        $display("[TB_MW_PY] FULL 16-WINDOW PYTHON GOLDEN PASSED");
        $display("[TB_MW_PY] windows             = %0d", MW_NUM_WINDOWS);
        $display("[TB_MW_PY] points per window   = %0d", MW_POINTS_PER_WINDOW);
        $display("[TB_MW_PY] accepted points     = %0d", accepted_points);
        $display("[TB_MW_PY] input wait cycles   = %0d", input_wait_cycles);
        $display("[TB_MW_PY] done cycle          = %0d", cycle_count);
        $display("[TB_MW_PY] verified collisions, bucket 0, banks,");
        $display("[TB_MW_PY] generations, ordering, and final Python X/Y/Z");
        $display("============================================================");

        #20;
        $finish;
    end

    initial begin
        #(64'd4000000000);
        $fatal(1, "[TB_MW_PY] Watchdog timeout.");
    end

endmodule