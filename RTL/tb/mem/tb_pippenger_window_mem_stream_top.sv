`timescale 1ns/1ps

module tb_pippenger_window_mem_stream_top;

    parameter int ADDR_W          = 8;
    parameter int DATA_W          = 256;
    parameter int DEPTH           = (1 << ADDR_W);
    parameter int SRAM_RD_LATENCY = 3;

    // Constant Jacobian point representations
    localparam logic [255:0] ZERO =
        256'h0000000000000000000000000000000000000000000000000000000000000000;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    // Affine base point coordinates (secp256k1)
    localparam logic [255:0] GX_M =
        256'h9981E643E9089F48979F48C033FD129C231E295329BC66DBD7362E5A487E2097;

    localparam logic [255:0] GY_M =
        256'hCF3F851FD4A582D670B6B59AAC19C1368DFC5D5D1F1DC64DB15EA6D2D3DBABE2;

    localparam logic [255:0] G2_AFF_X =
        256'hF918623CCBA0EE23CE0B62E1E014040471354AFC88B285A04E0640C981048D2C;

    localparam logic [255:0] G2_AFF_Y =
        256'h3C7F7712157B93134B3A0F64BDA2CC6584FD25167DC75CE17D12D622FFACCFBF;

    localparam logic [255:0] G3_AFF_X =
        256'h9497730FCDF4C0AD5940D07385985972066CEAFB22EB7BC42379D4BBD5FEA781;

    localparam logic [255:0] G3_AFF_Y =
        256'h3EC28DCD9215EC76CC6048BD84885650AC4964CDC5A1F91FAF18B0B0613F55A9;

    // Golden expected window evaluation results
    localparam logic [255:0] EXP_15G_X =
        256'h095BC488048E05A5732C475C3A609EFCC38EC30F0B30A04E778684E3DD149772;
    localparam logic [255:0] EXP_15G_Y =
        256'hA95C53D653DEE15BE8482AD23B040A470B14DC6069A4204C751D6C1C6D8FDC7D;
    localparam logic [255:0] EXP_15G_Z =
        256'h93F59AC795686CC45912CC7F9918DE3F914DBDA84AD1331E5C2FD8DA0EBF9998;

    localparam logic [255:0] EXP_7G_X =
        256'h0F8F394A3B4FC7ADB52F1D939F8201FF15E62BDF67713FD5BC30FA43204F6A82;
    localparam logic [255:0] EXP_7G_Y =
        256'hF516744370F63F9C63D15AEE77889944E82AF7D96A48C78295824FD26501777B;
    localparam logic [255:0] EXP_7G_Z =
        256'hB86E2532092A3ED72E7F1908DED9D4928616B664B361FCFCB85400FA48ECACFE;

    localparam logic [255:0] EXP_13G_X =
        256'h347A9BC7D1280A2EB70B787CDE4718A4D7E0D04076569F027F84A9CD09B5353A;
    localparam logic [255:0] EXP_13G_Y =
        256'hA9CB6A59396652818AADC609EA8880BFC9EAADBCA1E3510EEC72AD13B3086F85;
    localparam logic [255:0] EXP_13G_Z =
        256'h336F3C8D35298A07231FF68CBF3BFFC14C485617328BE37907F3E40FFDDD6C24;

    // Testbench Driving Signals
    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;

    // Streaming interface stimulus lines
    logic                in_valid;
    logic                in_ready;
    logic [ADDR_W-1:0]   in_bucket_id;
    logic [DATA_W-1:0]   in_point_x;
    logic [DATA_W-1:0]   in_point_y;
    logic                last_point;

    // Verification output lines
    logic [DATA_W-1:0] result_x;
    logic [DATA_W-1:0] result_y;
    logic [DATA_W-1:0] result_z;

    // Basic latency metrics
    int unsigned cycle_cnt;
    int unsigned start_cycle;
    int unsigned latency;

    // Performance counters per test case
    int unsigned perf_total_cycles;
    int unsigned perf_busy_cycles;
    int unsigned perf_idle_cycles;
    int unsigned perf_input_valid_cycles;
    int unsigned perf_input_accept_count;
    int unsigned perf_input_stall_cycles;
    int unsigned perf_expected_points;

    // Device Under Test (DUT) Instantiation
    pippenger_window_mem_stream_top #(
        .ADDR_W          (ADDR_W),
        .DATA_W          (DATA_W),
        .DEPTH           (DEPTH),
        .SRAM_RD_LATENCY (SRAM_RD_LATENCY)
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

    // Clock Generator (10ns Period -> 100MHz)
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // Simulation absolute cycle monitor
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_cnt <= 0;
        end else begin
            cycle_cnt <= cycle_cnt + 1;
        end
    end

    // Reusable Streaming Task using dynamic array sizes
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

            // Wait until the DUT is completely idle.
            // This also protects us from starting before the one-time tag init is done.
            @(posedge clk);
            wait (busy === 1'b0);
            @(negedge clk);

            // Reset performance counters for this test case
            perf_total_cycles        = 0;
            perf_busy_cycles         = 0;
            perf_idle_cycles         = 0;
            perf_input_valid_cycles  = 0;
            perf_input_accept_count  = 0;
            perf_input_stall_cycles  = 0;
            perf_expected_points     = stream_size;

            // Assert top-level start for one cycle.
            // The DUT no longer clears X/Y/Z memory per window.
            // It only increments current_gen and uses generation tags.
            start_cycle = cycle_cnt;
            start       = 1'b1;

            @(negedge clk);
            start = 1'b0;

            // Performance monitor for this window.
            fork
                begin : PERF_MONITOR
                    while (done !== 1'b1) begin
                        @(posedge clk);

                        perf_total_cycles++;

                        if (busy) begin
                            perf_busy_cycles++;
                        end else begin
                            perf_idle_cycles++;
                        end

                        if (in_valid) begin
                            perf_input_valid_cycles++;
                        end

                        if (in_valid && in_ready) begin
                            perf_input_accept_count++;
                        end

                        if (in_valid && !in_ready) begin
                            perf_input_stall_cycles++;
                        end
                    end
                end
            join_none

            // Sequentially stream out the elements inside the arrays
            for (int i = 0; i < stream_size; i++) begin
                in_point_x   = x_stream[i];
                in_point_y   = y_stream[i];
                in_bucket_id = b_stream[i];
                in_valid     = 1'b1;

                // Set last_point concurrently with the final packet
                if (i == stream_size - 1) begin
                    last_point = 1'b1;
                end else begin
                    last_point = 1'b0;
                end

                // Stall verification sequence until DUT captures packet via in_ready
                @(posedge clk);
                while (!in_ready) begin
                    @(posedge clk);
                end

                // Clear validation line for single clock isolation
                @(negedge clk);
                in_valid   = 1'b0;
                last_point = 1'b0;
            end

            // Stall thread until core triggers window completion flag
            wait (done === 1'b1);
            latency = cycle_cnt - start_cycle;

            // Verification checker block
            if (result_x !== exp_x || result_y !== exp_y || result_z !== exp_z) begin
                $display("[TB] %s FAILED", test_name);
                $display("[TB] EXPECTED X = %064h", exp_x);
                $display("[TB] GOT      X = %064h", result_x);
                $display("[TB] EXPECTED Y = %064h", exp_y);
                $display("[TB] GOT      Y = %064h", result_y);
                $display("[TB] EXPECTED Z = %064h", exp_z);
                $display("[TB] GOT      Z = %064h", result_z);
                $fatal(1, "[TB] Stream pipeline mismatch error.");
            end else begin
                $display("[TB] %s PASSED latency=%0d cycles", test_name, latency);

                $display("[PERF] %s expected_points        = %0d", test_name, perf_expected_points);
                $display("[PERF] %s total_cycles           = %0d", test_name, perf_total_cycles);
                $display("[PERF] %s busy_cycles            = %0d", test_name, perf_busy_cycles);
                $display("[PERF] %s idle_cycles            = %0d", test_name, perf_idle_cycles);
                $display("[PERF] %s input_valid_cycles     = %0d", test_name, perf_input_valid_cycles);
                $display("[PERF] %s input_accept_count     = %0d", test_name, perf_input_accept_count);
                $display("[PERF] %s input_stall_cycles     = %0d", test_name, perf_input_stall_cycles);

                if (perf_input_accept_count != perf_expected_points) begin
                    $display("[PERF_WARN] %s accepted point count mismatch: expected=%0d got=%0d",
                             test_name, perf_expected_points, perf_input_accept_count);
                end
            end

            @(posedge clk);
            repeat (3) @(posedge clk);
        end
    endtask

    // Stimulus Initialization
    initial begin
        // Force initial reset states
        start        = 1'b0;
        in_valid     = 1'b0;
        in_bucket_id = '0;
        in_point_x   = '0;
        in_point_y   = '0;
        last_point   = 1'b0;

        // System Cold Reset Sequence
        rst_n = 1'b0;
        repeat (5) @(posedge clk);
        rst_n = 1'b1;

        // Wait for one-time tag memory initialization after reset.
        wait (busy === 1'b0);
        repeat (2) @(posedge clk);

        $display("[TB] Starting pippenger_window_mem_stream_top validation sequence with performance counters");
        $display("[TB] Configuration: ADDR_W=%0d DEPTH=%0d SRAM_RD_LATENCY=%0d",
                 ADDR_W, DEPTH, SRAM_RD_LATENCY);

        // Test Case 1: Standard Multi-Bucket Mix Ingestion
        run_case_stream(
            "original_15G_stream",
            '{GX_M, G2_AFF_X, G3_AFF_X, GX_M},
            '{GY_M, G2_AFF_Y, G3_AFF_Y, GY_M},
            '{8'd1, 8'd2, 8'd3, 8'd1},
            EXP_15G_X, EXP_15G_Y, EXP_15G_Z
        );

        // Test Case 2: Zero Address Bypass Control
        run_case_stream(
            "all_zero_buckets_stream",
            '{GX_M, G2_AFF_X, G3_AFF_X, GX_M},
            '{GY_M, G2_AFF_Y, G3_AFF_Y, GY_M},
            '{8'd0, 8'd0, 8'd0, 8'd0},
            ZERO, ONE_M, ZERO
        );

        // Test Case 3: Lone Bucket Accumulation
        run_case_stream(
            "single_G_bucket1_stream",
            '{GX_M},
            '{GY_M},
            '{8'd1},
            GX_M, GY_M, ONE_M
        );

        // Test Case 4: Consecutive Same-Bucket Hammering
        run_case_stream(
            "all_points_bucket1_7G_stream",
            '{GX_M, G2_AFF_X, G3_AFF_X, GX_M},
            '{GY_M, G2_AFF_Y, G3_AFF_Y, GY_M},
            '{8'd1, 8'd1, 8'd1, 8'd1},
            EXP_7G_X, EXP_7G_Y, EXP_7G_Z
        );

        // Test Case 5: Partial Address Spectrum Load
        // Only buckets 2 and 3 are active.
        // Result = 2*G2 + 3*G3 = 2*(2G) + 3*(3G) = 13G
        run_case_stream(
            "bucket2_bucket3_only_13G_stream",
            '{G2_AFF_X, G3_AFF_X},
            '{G2_AFF_Y, G3_AFF_Y},
            '{8'd2, 8'd3},
            EXP_13G_X, EXP_13G_Y, EXP_13G_Z
        );

        $display("[TB] All stream verification cases PASSED.");
        #20;
        $finish;
    end

    // Simple Timeout Guard Window
    initial begin
        #500000;
        $display("[TB] ERROR: Simulation hit maximum watchdog limit timeout!");
        $finish;
    end

endmodule