
`timescale 1ns/1ps

module tb_msm_multiwindow_controller_16win_large_macro_v1;

    localparam int NUM_WINDOWS       = 16;
    localparam int POINTS_PER_WINDOW = 65536;

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
    longint unsigned window_start_cycle;
    longint unsigned window_end_cycle;

    `include "vectors/multiwindow_w16_large_macro_golden.svh"

    localparam logic [255:0] G_X_M =
        256'h9981E643E9089F48979F48C033FD129C231E295329BC66DBD7362E5A487E2097;

    localparam logic [255:0] G_Y_M =
        256'hCF3F851FD4A582D670B6B59AAC19C1368DFC5D5D1F1DC64DB15EA6D2D3DBABE2;

    localparam logic [255:0] G2_X_M =
        256'hF918623CCBA0EE23CE0B62E1E014040471354AFC88B285A04E0640C981048D2C;

    localparam logic [255:0] G2_Y_M =
        256'h3C7F7712157B93134B3A0F64BDA2CC6584FD25167DC75CE17D12D622FFACCFBF;

    localparam logic [255:0] G3_X_M =
        256'h9497730FCDF4C0AD5940D07385985972066CEAFB22EB7BC42379D4BBD5FEA781;

    localparam logic [255:0] G3_Y_M =
        256'h3EC28DCD9215EC76CC6048BD84885650AC4964CDC5A1F91FAF18B0B0613F55A9;

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

    function automatic logic [15:0] generate_bucket_id(
        input int window,
        input int point
    );
        longint unsigned value;
        begin
            /*
             * 16,384 active non-zero buckets per window.
             * 65,536 points / 16,384 buckets = four points per bucket:
             * one direct write followed by three Mixed Adds.
             */
            value = (
                (longint'(point)  * 251) +
                (longint'(window) * 4099)
            ) % 16384;

            generate_bucket_id = value[15:0] + 16'd1;
        end
    endfunction

    task automatic generate_point(
        input  int           window,
        input  int           point,
        output logic [255:0] point_x,
        output logic [255:0] point_y
    );
        int selector;
        begin
            selector = (point + window) % 3;

            case (selector)
                0: begin
                    point_x = G_X_M;
                    point_y = G_Y_M;
                end

                1: begin
                    point_x = G2_X_M;
                    point_y = G2_Y_M;
                end

                default: begin
                    point_x = G3_X_M;
                    point_y = G3_Y_M;
                end
            endcase
        end
    endtask

    task automatic send_window(input int window);
        integer point;
        logic [15:0]  generated_bucket;
        logic [255:0] generated_x;
        logic [255:0] generated_y;

        longint unsigned accepted_before;
        longint unsigned waits_before;

        begin
            $display("");
            $display("============================================================");
            $display(
                "[TB_MW_LARGE] Waiting for window=%0d at cycle=%0d",
                window,
                cycle_count
            );
            $display("============================================================");

            while ((window_index != window[3:0]) || !in_ready)
                @(posedge clk);

            window_start_cycle = cycle_count;
            accepted_before    = accepted_points;
            waits_before       = input_wait_cycles;

            for (point = 0; point < POINTS_PER_WINDOW; point++) begin
                generate_point(
                    window,
                    point,
                    generated_x,
                    generated_y
                );

                generated_bucket = generate_bucket_id(window, point);

                @(negedge clk);

                in_valid     = 1'b1;
                in_bucket_id = generated_bucket;
                in_point_x   = generated_x;
                in_point_y   = generated_y;
                last_point   = (point == POINTS_PER_WINDOW-1);

                /*
                 * Wait for exactly one valid/ready handshake.
                 *
                 * The point is accepted on the posedge where in_ready is high.
                 * There must not be an extra posedge afterward while in_valid
                 * still carries the same point.
                 */
                do begin
                    @(posedge clk);

                    if (!in_ready)
                        input_wait_cycles = input_wait_cycles + 1;

                end while (!in_ready);

                if (window_index != window[3:0]) begin
                    $fatal(
                        1,
                        "[TB_MW_LARGE] Window changed during stream: expected=%0d actual=%0d point=%0d",
                        window,
                        window_index,
                        point
                    );
                end

                if ((point != 0) && ((point % 8192) == 0)) begin
                    $display(
                        "[TB_MW_LARGE] window=%0d progress=%0d/%0d accepted_total=%0d waits_total=%0d cycle=%0d",
                        window,
                        point,
                        POINTS_PER_WINDOW,
                        accepted_points,
                        input_wait_cycles,
                        cycle_count
                    );
                end
            end

            @(negedge clk);

            in_valid     = 1'b0;
            in_bucket_id = '0;
            in_point_x   = '0;
            in_point_y   = '0;
            last_point   = 1'b0;

            window_end_cycle = cycle_count;

            $display("");
            $display("[TB_MW_LARGE] Completed input window=%0d", window);
            $display("[TB_MW_LARGE] points sent          = %0d", POINTS_PER_WINDOW);
            $display("[TB_MW_LARGE] accepted this window = %0d",
                     accepted_points - accepted_before);
            $display("[TB_MW_LARGE] waits this window    = %0d",
                     input_wait_cycles - waits_before);
            $display("[TB_MW_LARGE] stream cycles        = %0d",
                     window_end_cycle - window_start_cycle);
            $display("[TB_MW_LARGE] end input cycle      = %0d",
                     cycle_count);
        end
    endtask

    integer window;

    initial begin
        rst_n              = 1'b0;
        start              = 1'b0;
        in_valid           = 1'b0;
        in_bucket_id       = '0;
        in_point_x         = '0;
        in_point_y         = '0;
        last_point         = 1'b0;
        input_wait_cycles  = 0;
        window_start_cycle = 0;
        window_end_cycle   = 0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        start = 1'b1;

        @(negedge clk);
        start = 1'b0;

        for (
            window = NUM_WINDOWS-1;
            window >= 0;
            window = window - 1
        ) begin
            send_window(window);
        end

        $display("");
        $display("[TB_MW_LARGE] All windows sent. Waiting for final done...");

        while (!done)
            @(posedge clk);

        #0.1;

        if (accepted_points !==
            (longint'(NUM_WINDOWS) * POINTS_PER_WINDOW)) begin
            $fatal(
                1,
                "[TB_MW_LARGE] Accepted-point mismatch expected=%0d actual=%0d",
                NUM_WINDOWS * POINTS_PER_WINDOW,
                accepted_points
            );
        end

        if (
            result_x !== MW_LARGE_EXPECTED_X ||
            result_y !== MW_LARGE_EXPECTED_Y ||
            result_z !== MW_LARGE_EXPECTED_Z
        ) begin
            $display("");
            $display("============================================================");
            $display("[TB_MW_LARGE] FAILED");
            $display("[TB_MW_LARGE] expected X=%064h", MW_LARGE_EXPECTED_X);
            $display("[TB_MW_LARGE] actual   X=%064h", result_x);
            $display("[TB_MW_LARGE] expected Y=%064h", MW_LARGE_EXPECTED_Y);
            $display("[TB_MW_LARGE] actual   Y=%064h", result_y);
            $display("[TB_MW_LARGE] expected Z=%064h", MW_LARGE_EXPECTED_Z);
            $display("[TB_MW_LARGE] actual   Z=%064h", result_z);
            $display("============================================================");

            $fatal(
                1,
                "[TB_MW_LARGE] Full 16-window large MSM mismatch."
            );
        end

        $display("");
        $display("============================================================");
        $display("[TB_MW_LARGE] FULL 16-WINDOW LARGE MACRO GOLDEN PASSED");
        $display("[TB_MW_LARGE] windows           = %0d", NUM_WINDOWS);
        $display("[TB_MW_LARGE] points/window     = %0d", POINTS_PER_WINDOW);
        $display("[TB_MW_LARGE] accepted points   = %0d", accepted_points);
        $display("[TB_MW_LARGE] input wait cycles = %0d", input_wait_cycles);
        $display("[TB_MW_LARGE] done cycle        = %0d", cycle_count);
        $display("[TB_MW_LARGE] final X           = %064h", result_x);
        $display("[TB_MW_LARGE] final Y           = %064h", result_y);
        $display("[TB_MW_LARGE] final Z           = %064h", result_z);
        $display("============================================================");

        #20;
        $finish;
    end

    initial begin
        #(64'd8000000000);
        $fatal(
            1,
            "[TB_MW_LARGE] Watchdog timeout at cycle=%0d.",
            cycle_count
        );
    end

endmodule