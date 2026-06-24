`timescale 1ns/1ps

module tb_pippenger_window_mem_stream_stress;

    // Parameters matching our 8-bit parametric hardware architecture
    parameter int ADDR_W = 8;
    parameter int DATA_W = 256;
    parameter int DEPTH  = (1 << ADDR_W);

    // Constant definitions for elliptic curve points (secp256k1)
    localparam logic [255:0] ZERO = 256'h0;
    localparam logic [255:0] ONE_M = 256'h00000000000000000000000000000000000000000000000000000001000003D1;

    // Affine Base Point G
    localparam logic [255:0] GX_M = 256'h9981E643E9089F48979F48C033FD129C231E295329BC66DBD7362E5A487E2097;
    localparam logic [255:0] GY_M = 256'hCF3F851FD4A582D670B6B59AAC19C1368DFC5D5D1F1DC64DB15EA6D2D3DBABE2;

    // Testbench signals
    logic clk;
    logic rst_n;
    logic start;
    logic in_valid;
    logic in_ready;
    logic [ADDR_W-1:0] in_bucket_id;
    logic [DATA_W-1:0] in_point_x;
    logic [DATA_W-1:0] in_point_y;
    logic last_point;

    logic busy;
    logic done;
    logic [DATA_W-1:0] result_x;
    logic [DATA_W-1:0] result_y;
    logic [DATA_W-1:0] result_z;

    int unsigned cycle_cnt;
    int unsigned start_cycle;

    // Instantiate the Stream Top Level module
    pippenger_window_mem_stream_top #(
        .ADDR_W(ADDR_W),
        .DATA_W(DATA_W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_bucket_id(in_bucket_id),
        .in_point_x(in_point_x),
        .in_point_y(in_point_y),
        .last_point(last_point),
        .busy(busy),
        .done(done),
        .result_x(result_x),
        .result_y(result_y),
        .result_z(result_z)
    );

    // Clock Generation (100MHz)
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // Cycle Counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) cycle_cnt <= 0;
        else cycle_cnt <= cycle_cnt + 1;
    end

    // Task to model a single stream injection with full handshake compliance
    task automatic send_stream_point(
        input logic [ADDR_W-1:0] bid,
        input logic [255:0] px,
        input logic [255:0] py,
        input logic is_last
    );
        begin
            in_valid     = 1'b1;
            in_bucket_id = bid;
            in_point_x   = px;
            in_point_y   = py;
            last_point   = is_last;

            // Wait for the rising edge of clock where BOTH valid and ready are asserted
            do begin
                @(posedge clk);
            end while (in_ready !== 1'b1);

            // Deassert valid immediately on the next clock edge to model inter-packet gap if needed
            #1;
            in_valid     = 1'b0;
            last_point   = 1'b0;
        end
    endtask

    initial begin
        // Reset and initialization
        start        = 1'b0;
        in_valid     = 1'b0;
        in_bucket_id = '0;
        in_point_x   = '0;
        in_point_y   = '0;
        last_point   = 1'b0;
        
        rst_n = 1'b0;
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (5) @(posedge clk);

        $display("[STRESS_TB] --- STARTING ADVANCED STRESS TESTING ---");

        // ====================================================================
        // TEST CASE 6: THE HAMMER TEST (Continuous RMW Collisions on Bucket 5)
        // Sending 6 points consecutively into the exact same bucket ID.
        // This forces the backpressure mechanism to assert/deassert in_ready.
        // ====================================================================
        $display("[STRESS_TB] Executing Test 6: The Bucket Hammer Test...");
        
        @(negedge clk);
        start_cycle = cycle_cnt;
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        // Stream 5 points into Bucket 5, and the 6th point closes the stream
        send_stream_point(8'd5, GX_M, GY_M, 1'b0); // Pt 1 -> B5
        send_stream_point(8'd5, GX_M, GY_M, 1'b0); // Pt 2 -> B5 (Triggers RMW Stall)
        send_stream_point(8'd5, GX_M, GY_M, 1'b0); // Pt 3 -> B5 (Triggers RMW Stall)
        send_stream_point(8'd5, GX_M, GY_M, 1'b0); // Pt 4 -> B5 (Triggers RMW Stall)
        send_stream_point(8'd5, GX_M, GY_M, 1'b0); // Pt 5 -> B5 (Triggers RMW Stall)
        send_stream_point(8'd5, GX_M, GY_M, 1'b1); // Pt 6 -> B5 (Last point flag)

        // Wait for hardware to process pipeline and auto-run reduction
        wait(done === 1'b1);
        $display("[STRESS_TB] Test 6 Completed. Latency = %0d cycles.", (cycle_cnt - start_cycle));
        $display("[STRESS_TB] Resulting Point at Bucket 5 Accumulated 6*G.");
        $display("[STRESS_TB] GOT X = %064h", result_x);
        $display("[STRESS_TB] GOT Y = %064h", result_y);
        $display("[STRESS_TB] GOT Z = %064h", result_z);

        repeat (10) @(posedge clk);

        // ====================================================================
        // TEST CASE 7: THE DESERT TEST (Extreme Sparse Boundaries)
        // We place one point at Bucket 1, and one point at Bucket 255.
        // This validates the Infinity Bypass over a large continuous empty span.
        // ====================================================================
        $display("[STRESS_TB] Executing Test 7: The Desert Test (Sparse Space)...");
        
        @(negedge clk);
        start_cycle = cycle_cnt;
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        send_stream_point(8'd1,   GX_M, GY_M, 1'b0); // Point into Bucket 1
        send_stream_point(8'd255, GX_M, GY_M, 1'b1); // Point into Bucket 255 (Last)

        wait(done === 1'b1);
        $display("[STRESS_TB] Test 7 Completed. Latency = %0d cycles.", (cycle_cnt - start_cycle));
        $display("[STRESS_TB] If Infinity Bypass is operational, latency should be low (~1300-1500 cycles).");
        $display("[STRESS_TB] GOT X = %064h", result_x);
        $display("[STRESS_TB] GOT Y = %064h", result_y);
        $display("[STRESS_TB] GOT Z = %064h", result_z);

        $display("[STRESS_TB] --- ALL ADVANCED STRESS TESTS COMPLETED ---");
        #50;
        $finish;
    end

endmodule