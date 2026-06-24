`timescale 1ns/1ps

module tb_pippenger_window_seq_4;

    localparam int WIDTH = 256;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    // -------------------------------------------------------------
    // Affine Montgomery points
    // -------------------------------------------------------------

    // G affine, Montgomery domain
    localparam logic [255:0] G1_X =
        256'h9981E643E9089F48979F48C033FD129C231E295329BC66DBD7362E5A487E2097;

    localparam logic [255:0] G1_Y =
        256'hCF3F851FD4A582D670B6B59AAC19C1368DFC5D5D1F1DC64DB15EA6D2D3DBABE2;

    // 2G affine, Montgomery domain
    localparam logic [255:0] G2_AFF_X =
        256'hF918623CCBA0EE23CE0B62E1E014040471354AFC88B285A04E0640C981048D2C;

    localparam logic [255:0] G2_AFF_Y =
        256'h3C7F7712157B93134B3A0F64BDA2CC6584FD25167DC75CE17D12D622FFACCFBF;

    // 3G affine, Montgomery domain
    localparam logic [255:0] G3_AFF_X =
        256'h9497730FCDF4C0AD5940D07385985972066CEAFB22EB7BC42379D4BBD5FEA781;

    localparam logic [255:0] G3_AFF_Y =
        256'h3EC28DCD9215EC76CC6048BD84885650AC4964CDC5A1F91FAF18B0B0613F55A9;

    // -------------------------------------------------------------
    // Expected results
    // -------------------------------------------------------------

    // INF = (0, ONE_M, 0)
    localparam logic [255:0] INF_X =
        256'h0;

    localparam logic [255:0] INF_Y =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    localparam logic [255:0] INF_Z =
        256'h0;

    // Expected original full window result:
    //
    // Points / bucket IDs:
    //   P0 = G   -> bucket1
    //   P1 = 2G  -> bucket2
    //   P2 = 3G  -> bucket3
    //   P3 = G   -> bucket1
    //
    // Buckets after build:
    //   bucket1 = G + G = 2G
    //   bucket2 = 2G
    //   bucket3 = 3G
    //
    // Reduction:
    //   result = 1*bucket1 + 2*bucket2 + 3*bucket3
    //          = 1*(2G) + 2*(2G) + 3*(3G)
    //          = 2G + 4G + 9G
    //          = 15G
    //
    // Expected below is in Jacobian Montgomery coordinates.
    localparam logic [255:0] EXP_15G_X =
        256'h095BC488048E05A5732C475C3A609EFCC38EC30F0B30A04E778684E3DD149772;

    localparam logic [255:0] EXP_15G_Y =
        256'hA95C53D653DEE15BE8482AD23B040A470B14DC6069A4204C751D6C1C6D8FDC7D;

    localparam logic [255:0] EXP_15G_Z =
        256'h93F59AC795686CC45912CC7F9918DE3F914DBDA84AD1331E5C2FD8DA0EBF9998;

    // Expected all_points_bucket1 result:
    //
    //   P0 = G   -> bucket1
    //   P1 = 2G  -> bucket1
    //   P2 = 3G  -> bucket1
    //   P3 = G   -> bucket1
    //
    // bucket1 = G + 2G + 3G + G = 7G
    // reduction result = 1*bucket1 = 7G
    //
    // This is Jacobian Montgomery coordinates after the same mixed-add flow.
    localparam logic [255:0] EXP_7G_X =
        256'h0F8F394A3B4FC7ADB52F1D939F8201FF15E62BDF67713FD5BC30FA43204F6A82;

    localparam logic [255:0] EXP_7G_Y =
        256'hF516744370F63F9C63D15AEE77889944E82AF7D96A48C78295824FD26501777B;

    localparam logic [255:0] EXP_7G_Z =
        256'hB86E2532092A3ED72E7F1908DED9D4928616B664B361FCFCB85400FA48ECACFE;

    // Expected bucket2_bucket3_only result:
    //
    //   P0 = G   -> ignored
    //   P1 = 2G  -> bucket2
    //   P2 = 3G  -> bucket3
    //   P3 = G   -> ignored
    //
    // bucket1 = INF
    // bucket2 = 2G
    // bucket3 = 3G
    //
    // Reduction:
    //   result = 1*bucket1 + 2*bucket2 + 3*bucket3
    //          = 0 + 4G + 9G
    //          = 13G
    //
    // This is Jacobian Montgomery coordinates for the reduction flow.
    localparam logic [255:0] EXP_13G_X =
        256'h347A9BC7D1280A2EB70B787CDE4718A4D7E0D04076569F027F84A9CD09B5353A;

    localparam logic [255:0] EXP_13G_Y =
        256'hA9CB6A59396652818AADC609EA8880BFC9EAADBCA1E3510EEC72AD13B3086F85;

    localparam logic [255:0] EXP_13G_Z =
        256'h336F3C8D35298A07231FF68CBF3BFFC14C485617328BE37907F3E40FFDDD6C24;

    // -------------------------------------------------------------
    // DUT signals
    // -------------------------------------------------------------

    logic clk = 1'b0;
    logic rst_n;

    logic start;
    logic busy;
    logic done;

    logic [WIDTH-1:0] p0_x;
    logic [WIDTH-1:0] p0_y;
    logic [1:0]       p0_bid;

    logic [WIDTH-1:0] p1_x;
    logic [WIDTH-1:0] p1_y;
    logic [1:0]       p1_bid;

    logic [WIDTH-1:0] p2_x;
    logic [WIDTH-1:0] p2_y;
    logic [1:0]       p2_bid;

    logic [WIDTH-1:0] p3_x;
    logic [WIDTH-1:0] p3_y;
    logic [1:0]       p3_bid;

    logic [WIDTH-1:0] result_x;
    logic [WIDTH-1:0] result_y;
    logic [WIDTH-1:0] result_z;

    always #5 clk = ~clk;

    pippenger_window_seq_4 #(
        .WIDTH(WIDTH)
    ) dut (
        .clk   (clk),
        .rst_n (rst_n),

        .start (start),

        .p0_x   (p0_x),
        .p0_y   (p0_y),
        .p0_bid (p0_bid),

        .p1_x   (p1_x),
        .p1_y   (p1_y),
        .p1_bid (p1_bid),

        .p2_x   (p2_x),
        .p2_y   (p2_y),
        .p2_bid (p2_bid),

        .p3_x   (p3_x),
        .p3_y   (p3_y),
        .p3_bid (p3_bid),

        .busy (busy),
        .done (done),

        .result_x (result_x),
        .result_y (result_y),
        .result_z (result_z)
    );

    // -------------------------------------------------------------
    // Generic test task
    // -------------------------------------------------------------
    task automatic run_window_check(
        input logic [WIDTH-1:0] in_p0_x,
        input logic [WIDTH-1:0] in_p0_y,
        input logic [1:0]       in_p0_bid,

        input logic [WIDTH-1:0] in_p1_x,
        input logic [WIDTH-1:0] in_p1_y,
        input logic [1:0]       in_p1_bid,

        input logic [WIDTH-1:0] in_p2_x,
        input logic [WIDTH-1:0] in_p2_y,
        input logic [1:0]       in_p2_bid,

        input logic [WIDTH-1:0] in_p3_x,
        input logic [WIDTH-1:0] in_p3_y,
        input logic [1:0]       in_p3_bid,

        input logic [WIDTH-1:0] exp_x,
        input logic [WIDTH-1:0] exp_y,
        input logic [WIDTH-1:0] exp_z,

        input string            test_name
    );
        int cycles;

        begin
            @(posedge clk);

            if (busy) begin
                $fatal(1, "[%s] Tried to start while busy", test_name);
            end

            p0_x   <= in_p0_x;
            p0_y   <= in_p0_y;
            p0_bid <= in_p0_bid;

            p1_x   <= in_p1_x;
            p1_y   <= in_p1_y;
            p1_bid <= in_p1_bid;

            p2_x   <= in_p2_x;
            p2_y   <= in_p2_y;
            p2_bid <= in_p2_bid;

            p3_x   <= in_p3_x;
            p3_y   <= in_p3_y;
            p3_bid <= in_p3_bid;

            start <= 1'b1;

            @(posedge clk);
            start <= 1'b0;

            p0_x   <= '0;
            p0_y   <= '0;
            p0_bid <= 2'd0;

            p1_x   <= '0;
            p1_y   <= '0;
            p1_bid <= 2'd0;

            p2_x   <= '0;
            p2_y   <= '0;
            p2_bid <= 2'd0;

            p3_x   <= '0;
            p3_y   <= '0;
            p3_bid <= 2'd0;

            cycles = 0;
            while (!done) begin
                @(posedge clk);
                cycles++;

                if (cycles > 8000) begin
                    $fatal(1, "[%s] TIMEOUT waiting for done", test_name);
                end
            end

            $display("[%s] latency = %0d cycles", test_name, cycles);
            $display("[%s] X = %h", test_name, result_x);
            $display("[%s] Y = %h", test_name, result_y);
            $display("[%s] Z = %h", test_name, result_z);

            if (result_x !== exp_x || result_y !== exp_y || result_z !== exp_z) begin
                $display("[%s] EXPECTED X = %h", test_name, exp_x);
                $display("[%s] GOT      X = %h", test_name, result_x);
                $display("[%s] EXPECTED Y = %h", test_name, exp_y);
                $display("[%s] GOT      Y = %h", test_name, result_y);
                $display("[%s] EXPECTED Z = %h", test_name, exp_z);
                $display("[%s] GOT      Z = %h", test_name, result_z);
                $fatal(1, "[%s] pippenger_window_seq_4 mismatch", test_name);
            end

            $display("[%s] PASSED", test_name);

            // Allow FSM to return cleanly to IDLE
            @(posedge clk);
        end
    endtask

    // -------------------------------------------------------------
    // Main test sequence
    // -------------------------------------------------------------
    initial begin
        rst_n = 1'b0;
        start = 1'b0;

        p0_x   = '0;
        p0_y   = '0;
        p0_bid = 2'd0;

        p1_x   = '0;
        p1_y   = '0;
        p1_bid = 2'd0;

        p2_x   = '0;
        p2_y   = '0;
        p2_bid = 2'd0;

        p3_x   = '0;
        p3_y   = '0;
        p3_bid = 2'd0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        $display("==============================================");
        $display(" tb_pippenger_window_seq_4 START");
        $display("==============================================");

        // ---------------------------------------------------------
        // Test 1:
        // Original scenario:
        //   P0 = G   -> bucket1
        //   P1 = 2G  -> bucket2
        //   P2 = 3G  -> bucket3
        //   P3 = G   -> bucket1
        //
        // Expected:
        //   15G
        // ---------------------------------------------------------
        run_window_check(
            G1_X,      G1_Y,      2'd1,
            G2_AFF_X,  G2_AFF_Y,  2'd2,
            G3_AFF_X,  G3_AFF_Y,  2'd3,
            G1_X,      G1_Y,      2'd1,

            EXP_15G_X,
            EXP_15G_Y,
            EXP_15G_Z,

            "original_15G"
        );

        // ---------------------------------------------------------
        // Test 2:
        // All bucket IDs are zero.
        //
        // Expected:
        //   no points are added, result = INF.
        // ---------------------------------------------------------
        run_window_check(
            G1_X,      G1_Y,      2'd0,
            G2_AFF_X,  G2_AFF_Y,  2'd0,
            G3_AFF_X,  G3_AFF_Y,  2'd0,
            G1_X,      G1_Y,      2'd0,

            INF_X,
            INF_Y,
            INF_Z,

            "all_zero_buckets"
        );

        // ---------------------------------------------------------
        // Test 3:
        // Single active bucket:
        //   P0 = G -> bucket1
        //   all other points ignored.
        //
        // Expected:
        //   result = 1*bucket1 = G.
        //   Since the bucket received one affine point into empty bucket,
        //   result should be G represented as Jacobian with Z=ONE_M.
        // ---------------------------------------------------------
        run_window_check(
            G1_X,      G1_Y,      2'd1,
            G2_AFF_X,  G2_AFF_Y,  2'd0,
            G3_AFF_X,  G3_AFF_Y,  2'd0,
            G1_X,      G1_Y,      2'd0,

            G1_X,
            G1_Y,
            ONE_M,

            "single_G_bucket1"
        );

        // ---------------------------------------------------------
        // Test 4:
        // All points go to bucket1:
        //   G + 2G + 3G + G = 7G
        //
        // Since only bucket1 is active:
        //   reduction result = 1*bucket1 = 7G
        // ---------------------------------------------------------
        run_window_check(
            G1_X,      G1_Y,      2'd1,
            G2_AFF_X,  G2_AFF_Y,  2'd1,
            G3_AFF_X,  G3_AFF_Y,  2'd1,
            G1_X,      G1_Y,      2'd1,

            EXP_7G_X,
            EXP_7G_Y,
            EXP_7G_Z,

            "all_points_bucket1_7G"
        );

        // ---------------------------------------------------------
        // Test 5:
        // Only bucket2 and bucket3 are active:
        //   P1 = 2G -> bucket2
        //   P2 = 3G -> bucket3
        //
        // Reduction:
        //   result = 2*bucket2 + 3*bucket3
        //          = 2*(2G) + 3*(3G)
        //          = 4G + 9G
        //          = 13G
        // ---------------------------------------------------------
        run_window_check(
            G1_X,      G1_Y,      2'd0,
            G2_AFF_X,  G2_AFF_Y,  2'd2,
            G3_AFF_X,  G3_AFF_Y,  2'd3,
            G1_X,      G1_Y,      2'd0,

            EXP_13G_X,
            EXP_13G_Y,
            EXP_13G_Z,

            "bucket2_bucket3_only_13G"
        );

        $display("==============================================");
        $display(" tb_pippenger_window_seq_4 PASSED");
        $display("==============================================");

        $finish;
    end

endmodule