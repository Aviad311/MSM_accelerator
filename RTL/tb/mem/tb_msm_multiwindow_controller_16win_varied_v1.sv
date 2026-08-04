`timescale 1ns/1ps

module tb_msm_multiwindow_controller_16win_varied_v1;

    localparam int NUM_WINDOWS = 16;

    localparam logic [255:0] GX_M =
        256'h9981E643E9089F48979F48C033FD129C231E295329BC66DBD7362E5A487E2097;

    localparam logic [255:0] GY_M =
        256'hCF3F851FD4A582D670B6B59AAC19C1368DFC5D5D1F1DC64DB15EA6D2D3DBABE2;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

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

    logic              ref_dbl_start;
    logic              ref_dbl_busy;
    logic              ref_dbl_done;
    logic [255:0]      ref_dbl_x;
    logic [255:0]      ref_dbl_y;
    logic [255:0]      ref_dbl_z;

    logic              ref_add_start;
    logic              ref_add_busy;
    logic              ref_add_done;
    logic [255:0]      ref_add_x;
    logic [255:0]      ref_add_y;
    logic [255:0]      ref_add_z;

    logic [255:0]      ref_acc_x;
    logic [255:0]      ref_acc_y;
    logic [255:0]      ref_acc_z;

    logic [255:0]      ref_add_rhs_x;
    logic [255:0]      ref_add_rhs_y;
    logic [255:0]      ref_add_rhs_z;

    logic              reference_complete;

    longint unsigned cycle_count;

    initial clk = 1'b0;
    always #5 clk = ~clk;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cycle_count <= 0;
        else
            cycle_count <= cycle_count + 1;
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

    jacobian_double_seq #(
        .WIDTH(256)
    ) u_ref_double (
        .clk   (clk),
        .rst_n (rst_n),
        .start (ref_dbl_start),
        .X1    (ref_acc_x),
        .Y1    (ref_acc_y),
        .Z1    (ref_acc_z),
        .busy  (ref_dbl_busy),
        .done  (ref_dbl_done),
        .X3    (ref_dbl_x),
        .Y3    (ref_dbl_y),
        .Z3    (ref_dbl_z)
    );

    jacobian_add_4mul_seq #(
        .WIDTH(256)
    ) u_ref_add (
        .clk   (clk),
        .rst_n (rst_n),
        .start (ref_add_start),
        .X1    (ref_acc_x),
        .Y1    (ref_acc_y),
        .Z1    (ref_acc_z),
        .X2    (ref_add_rhs_x),
        .Y2    (ref_add_rhs_y),
        .Z2    (ref_add_rhs_z),
        .busy  (ref_add_busy),
        .done  (ref_add_done),
        .X3    (ref_add_x),
        .Y3    (ref_add_y),
        .Z3    (ref_add_z)
    );

    function automatic int bucket_for_window(input int w);
        begin
            // Deliberately asymmetric pattern to verify ordering and weights.
            case (w)
                15: bucket_for_window = 3;
                14: bucket_for_window = 7;
                13: bucket_for_window = 2;
                12: bucket_for_window = 11;
                11: bucket_for_window = 5;
                10: bucket_for_window = 13;
                 9: bucket_for_window = 4;
                 8: bucket_for_window = 16;
                 7: bucket_for_window = 6;
                 6: bucket_for_window = 9;
                 5: bucket_for_window = 1;
                 4: bucket_for_window = 15;
                 3: bucket_for_window = 8;
                 2: bucket_for_window = 12;
                 1: bucket_for_window = 10;
                 0: bucket_for_window = 14;
                default: bucket_for_window = 1;
            endcase
        end
    endfunction

    task automatic send_single_g(
        input int expected_window,
        input int bucket_id
    );
        begin
            while (!(in_ready && window_index == expected_window[3:0]))
                @(posedge clk);

            @(negedge clk);
            in_valid     = 1'b1;
            in_bucket_id = bucket_id[15:0];
            in_point_x   = GX_M;
            in_point_y   = GY_M;
            last_point   = 1'b1;

            do @(posedge clk); while (!in_ready);

            $display(
                "[TB_16WIN_VAR] Sent G to window=%0d bucket=%0d cycle=%0d",
                expected_window,
                bucket_id,
                cycle_count
            );

            @(negedge clk);
            in_valid     = 1'b0;
            in_bucket_id = '0;
            in_point_x   = '0;
            in_point_y   = '0;
            last_point   = 1'b0;
        end
    endtask

    task automatic ref_double_once;
        begin
            @(negedge clk);
            ref_dbl_start = 1'b1;
            @(negedge clk);
            ref_dbl_start = 1'b0;

            while (!ref_dbl_done)
                @(posedge clk);

            #0.1;
            ref_acc_x = ref_dbl_x;
            ref_acc_y = ref_dbl_y;
            ref_acc_z = ref_dbl_z;
        end
    endtask

    task automatic ref_double_16;
        integer i;
        begin
            for (i = 0; i < 16; i = i + 1)
                ref_double_once();
        end
    endtask

    task automatic ref_add_point(
        input logic [255:0] rhs_x,
        input logic [255:0] rhs_y,
        input logic [255:0] rhs_z
    );
        begin
            ref_add_rhs_x = rhs_x;
            ref_add_rhs_y = rhs_y;
            ref_add_rhs_z = rhs_z;

            @(negedge clk);
            ref_add_start = 1'b1;
            @(negedge clk);
            ref_add_start = 1'b0;

            while (!ref_add_done)
                @(posedge clk);

            #0.1;
            ref_acc_x = ref_add_x;
            ref_acc_y = ref_add_y;
            ref_acc_z = ref_add_z;
        end
    endtask

    task automatic build_bucket_multiple(
        input  int bucket_id,
        output logic [255:0] out_x,
        output logic [255:0] out_y,
        output logic [255:0] out_z
    );
        integer i;
        begin
            // bucket 1 -> G, bucket b -> b*G.
            ref_acc_x = GX_M;
            ref_acc_y = GY_M;
            ref_acc_z = ONE_M;

            for (i = 1; i < bucket_id; i = i + 1)
                ref_add_point(GX_M, GY_M, ONE_M);

            out_x = ref_acc_x;
            out_y = ref_acc_y;
            out_z = ref_acc_z;
        end
    endtask

    task automatic build_reference;
        integer w;
        integer b;
        logic [255:0] win_x;
        logic [255:0] win_y;
        logic [255:0] win_z;
        begin
            reference_complete = 1'b0;

            b = bucket_for_window(NUM_WINDOWS-1);
            build_bucket_multiple(b, win_x, win_y, win_z);

            ref_acc_x = win_x;
            ref_acc_y = win_y;
            ref_acc_z = win_z;

            for (w = NUM_WINDOWS-2; w >= 0; w = w - 1) begin
                ref_double_16();

                b = bucket_for_window(w);
                build_bucket_multiple(b, win_x, win_y, win_z);

                // Restore accumulated value after building b*G.
                // build_bucket_multiple uses ref_acc internally.
                // Save the shifted accumulator before entering it.
                // This save/restore is handled explicitly below.
            end
        end
    endtask

    // Separate reference implementation with explicit saves to avoid
    // overwriting the global accumulator while constructing b*G.
    task automatic build_reference_safe;
        integer w;
        integer b;
        logic [255:0] shifted_x;
        logic [255:0] shifted_y;
        logic [255:0] shifted_z;
        logic [255:0] win_x;
        logic [255:0] win_y;
        logic [255:0] win_z;
        begin
            reference_complete = 1'b0;

            b = bucket_for_window(NUM_WINDOWS-1);
            build_bucket_multiple(b, win_x, win_y, win_z);

            ref_acc_x = win_x;
            ref_acc_y = win_y;
            ref_acc_z = win_z;

            for (w = NUM_WINDOWS-2; w >= 0; w = w - 1) begin
                ref_double_16();

                shifted_x = ref_acc_x;
                shifted_y = ref_acc_y;
                shifted_z = ref_acc_z;

                b = bucket_for_window(w);
                build_bucket_multiple(b, win_x, win_y, win_z);

                ref_acc_x = shifted_x;
                ref_acc_y = shifted_y;
                ref_acc_z = shifted_z;

                ref_add_point(win_x, win_y, win_z);
            end

            reference_complete = 1'b1;
            $display("[TB_16WIN_VAR] Independent varied-window reference complete.");
        end
    endtask

    integer w;

    initial begin
        rst_n              = 1'b0;
        start              = 1'b0;
        in_valid           = 1'b0;
        in_bucket_id       = '0;
        in_point_x         = '0;
        in_point_y         = '0;
        last_point         = 1'b0;
        ref_dbl_start      = 1'b0;
        ref_add_start      = 1'b0;
        ref_acc_x          = '0;
        ref_acc_y          = ONE_M;
        ref_acc_z          = '0;
        ref_add_rhs_x      = '0;
        ref_add_rhs_y      = ONE_M;
        ref_add_rhs_z      = '0;
        reference_complete = 1'b0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;

        fork
            build_reference_safe();
        join_none

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        for (w = NUM_WINDOWS-1; w >= 0; w = w - 1)
            send_single_g(w, bucket_for_window(w));

        while (!done)
            @(posedge clk);

        while (!reference_complete)
            @(posedge clk);

        #0.1;

        if (result_x !== ref_acc_x ||
            result_y !== ref_acc_y ||
            result_z !== ref_acc_z) begin

            $display("[TB_16WIN_VAR] FAILED");
            $display("[TB_16WIN_VAR] expected X=%064h", ref_acc_x);
            $display("[TB_16WIN_VAR] actual   X=%064h", result_x);
            $display("[TB_16WIN_VAR] expected Y=%064h", ref_acc_y);
            $display("[TB_16WIN_VAR] actual   Y=%064h", result_y);
            $display("[TB_16WIN_VAR] expected Z=%064h", ref_acc_z);
            $display("[TB_16WIN_VAR] actual   Z=%064h", result_z);
            $fatal(1, "[TB_16WIN_VAR] Varied-window result mismatch.");
        end

        $display("");
        $display("============================================================");
        $display("[TB_16WIN_VAR] SIXTEEN VARIED WINDOWS PASSED");
        $display("[TB_16WIN_VAR] each window used a different bucket weight");
        $display("[TB_16WIN_VAR] verified ordering, bucket weighting, and 256-bit accumulation");
        $display("[TB_16WIN_VAR] done cycle = %0d", cycle_count);
        $display("============================================================");

        #20;
        $finish;
    end

    initial begin
        #(64'd3000000000);
        $fatal(1, "[TB_16WIN_VAR] Watchdog timeout.");
    end

endmodule