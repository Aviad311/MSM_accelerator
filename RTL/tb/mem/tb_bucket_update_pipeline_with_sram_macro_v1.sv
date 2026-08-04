`timescale 1ns/1ps

module tb_bucket_update_pipeline_with_sram_macro_v1;

    localparam int ADDR_W        = 13;
    localparam int DATA_W        = 256;
    localparam int DEPTH         = 8192;
    localparam int GEN_W         = 16;
    localparam int SLOT_COUNT    = 16;
    localparam int MIX_CTX_COUNT = 40;
    localparam int MUL_LATENCY   = 16;

    localparam logic [GEN_W-1:0] TEST_GEN = 16'h002A;
    localparam logic [ADDR_W-1:0] TEST_BUCKET = 13'd7;

    localparam logic [255:0] ZERO =
        256'h0000000000000000000000000000000000000000000000000000000000000000;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    localparam logic [255:0] GX_M =
        256'h9981E643E9089F48979F48C033FD129C231E295329BC66DBD7362E5A487E2097;

    localparam logic [255:0] GY_M =
        256'hCF3F851FD4A582D670B6B59AAC19C1368DFC5D5D1F1DC64DB15EA6D2D3DBABE2;

    localparam logic [255:0] EXP_2G_X =
        256'h7C75DD9524177D593C03889B8DCD9B1CB05FB7D2A3DA7FE8BA9F29B104E7DB13;

    localparam logic [255:0] EXP_2G_Y =
        256'h55DEBB381F4AD034CC27CB48A46449AAA87D43FDB563384B1CD20838E6FDDC9F;

    localparam logic [255:0] EXP_2G_Z =
        256'h9E7F0A3FA94B05ACE16D6B355833826D1BF8BABA3E3B8C9B62BD4DA6A7B75B95;

    logic clk;
    logic rst_n;

    logic                  in_valid;
    logic                  in_ready;
    logic [GEN_W-1:0]      current_gen;
    logic [ADDR_W-1:0]     in_bucket_id;
    logic [DATA_W-1:0]     in_point_x;
    logic [DATA_W-1:0]     in_point_y;

    logic                  out_valid;
    logic                  out_ready;
    logic [ADDR_W-1:0]     out_bucket_id;
    logic                  out_skipped;
    logic                  out_direct_write;
    logic                  out_mixed_add;
    logic [DATA_W-1:0]     out_x;
    logic [DATA_W-1:0]     out_y;
    logic [DATA_W-1:0]     out_z;

    logic [$clog2(SLOT_COUNT+1)-1:0] active_slots;
    logic [63:0] accepted_count;
    logic [63:0] completed_count;
    logic [63:0] same_bucket_stall_count;
    logic [63:0] direct_write_count;
    logic [63:0] mixed_add_count;

    int unsigned cycle_count;
    int unsigned pass_count;

    bucket_update_pipeline_with_sram_macro_v1 #(
        .ADDR_W           (ADDR_W),
        .DATA_W           (DATA_W),
        .DEPTH            (DEPTH),
        .GEN_W            (GEN_W),
        .SLOT_COUNT       (SLOT_COUNT),
        .MIX_CTX_COUNT    (MIX_CTX_COUNT),
        .MUL_LATENCY      (MUL_LATENCY),
        .SKIP_ZERO_BUCKET (1'b1)
    ) dut (
        .clk                     (clk),
        .rst_n                   (rst_n),

        .in_valid                (in_valid),
        .in_ready                (in_ready),
        .current_gen             (current_gen),
        .in_bucket_id            (in_bucket_id),
        .in_point_x              (in_point_x),
        .in_point_y              (in_point_y),

        .out_valid               (out_valid),
        .out_ready               (out_ready),
        .out_bucket_id           (out_bucket_id),
        .out_skipped             (out_skipped),
        .out_direct_write        (out_direct_write),
        .out_mixed_add           (out_mixed_add),
        .out_x                   (out_x),
        .out_y                   (out_y),
        .out_z                   (out_z),

        .active_slots            (active_slots),
        .accepted_count          (accepted_count),
        .completed_count         (completed_count),
        .same_bucket_stall_count (same_bucket_stall_count),
        .direct_write_count      (direct_write_count),
        .mixed_add_count         (mixed_add_count)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cycle_count <= 0;
        else
            cycle_count <= cycle_count + 1;
    end

    task automatic send_update(
        input logic [ADDR_W-1:0] bucket_id,
        input logic [DATA_W-1:0] point_x,
        input logic [DATA_W-1:0] point_y
    );
        begin
            @(negedge clk);
            in_valid     = 1'b1;
            in_bucket_id = bucket_id;
            in_point_x   = point_x;
            in_point_y   = point_y;

            while (1) begin
                @(posedge clk);
                if (in_ready)
                    break;
            end

            @(negedge clk);
            in_valid     = 1'b0;
            in_bucket_id = '0;
            in_point_x   = '0;
            in_point_y   = '0;

            $display(
                "[TB] INPUT ACCEPTED cycle=%0d bucket=%0d",
                cycle_count,
                bucket_id
            );
        end
    endtask

    task automatic wait_and_check_output(
        input string             test_name,
        input logic [ADDR_W-1:0] expected_bucket,
        input logic              expected_skipped,
        input logic              expected_direct,
        input logic              expected_mixed,
        input logic [DATA_W-1:0] expected_x,
        input logic [DATA_W-1:0] expected_y,
        input logic [DATA_W-1:0] expected_z
    );
        begin
            while (1) begin
                @(posedge clk);
                if (out_valid && out_ready)
                    break;
            end

            if (out_bucket_id !== expected_bucket ||
                out_skipped !== expected_skipped ||
                out_direct_write !== expected_direct ||
                out_mixed_add !== expected_mixed ||
                out_x !== expected_x ||
                out_y !== expected_y ||
                out_z !== expected_z) begin

                $display("[TB] %s FAILED", test_name);
                $display("[TB] bucket expected=%0d got=%0d",
                         expected_bucket, out_bucket_id);
                $display("[TB] flags expected skip/direct/mixed=%b/%b/%b",
                         expected_skipped, expected_direct, expected_mixed);
                $display("[TB] flags got      skip/direct/mixed=%b/%b/%b",
                         out_skipped, out_direct_write, out_mixed_add);
                $display("[TB] expected X=%064h", expected_x);
                $display("[TB] got      X=%064h", out_x);
                $display("[TB] expected Y=%064h", expected_y);
                $display("[TB] got      Y=%064h", out_y);
                $display("[TB] expected Z=%064h", expected_z);
                $display("[TB] got      Z=%064h", out_z);

                $fatal(1, "[TB] Integration output mismatch");
            end

            pass_count++;

            $display(
                "[TB] %s PASSED cycle=%0d bucket=%0d",
                test_name,
                cycle_count,
                out_bucket_id
            );

            @(posedge clk);
        end
    endtask

    initial begin
        pass_count   = 0;

        rst_n        = 1'b0;
        in_valid     = 1'b0;
        current_gen  = TEST_GEN;
        in_bucket_id = '0;
        in_point_x   = '0;
        in_point_y   = '0;
        out_ready    = 1'b1;

        repeat (5) @(posedge clk);

        @(negedge clk);
        rst_n = 1'b1;

        repeat (3) @(posedge clk);

        $display("");
        $display("============================================================");
        $display("[TB] Starting pipeline + SRAM macro integration test");
        $display("============================================================");

        // First G enters an empty-generation bucket and must be written directly.
        send_update(TEST_BUCKET, GX_M, GY_M);

        wait_and_check_output(
            "direct_write_G",
            TEST_BUCKET,
            1'b0,
            1'b1,
            1'b0,
            GX_M,
            GY_M,
            ONE_M
        );

        // Second G reads the stored G and must execute G + G = 2G.
        send_update(TEST_BUCKET, GX_M, GY_M);

        wait_and_check_output(
            "mixed_add_G_plus_G",
            TEST_BUCKET,
            1'b0,
            1'b0,
            1'b1,
            EXP_2G_X,
            EXP_2G_Y,
            EXP_2G_Z
        );

        // Bucket zero must be skipped and must not touch SRAM.
        send_update(13'd0, GX_M, GY_M);

        wait_and_check_output(
            "skip_bucket_zero",
            13'd0,
            1'b1,
            1'b0,
            1'b0,
            ZERO,
            ONE_M,
            ZERO
        );

        repeat (2) @(posedge clk);

        if (accepted_count !== 64'd3)
            $fatal(1, "[TB] accepted_count expected 3 got %0d", accepted_count);

        if (completed_count !== 64'd3)
            $fatal(1, "[TB] completed_count expected 3 got %0d", completed_count);

        if (direct_write_count !== 64'd1)
            $fatal(1, "[TB] direct_write_count expected 1 got %0d",
                   direct_write_count);

        if (mixed_add_count !== 64'd1)
            $fatal(1, "[TB] mixed_add_count expected 1 got %0d",
                   mixed_add_count);

        if (active_slots !== '0)
            $fatal(1, "[TB] active_slots expected 0 got %0d", active_slots);

        $display("");
        $display("============================================================");
        $display("[TB] pipeline + SRAM macro integration PASSED");
        $display("[TB] Successful checks        = %0d", pass_count);
        $display("[TB] Accepted                 = %0d", accepted_count);
        $display("[TB] Completed                = %0d", completed_count);
        $display("[TB] Direct writes            = %0d", direct_write_count);
        $display("[TB] Mixed adds               = %0d", mixed_add_count);
        $display("[TB] Same-bucket stall cycles = %0d",
                 same_bucket_stall_count);
        $display("[TB] Total cycles             = %0d", cycle_count);
        $display("============================================================");

        #20;
        $finish;
    end

    initial begin
        #20000;
        $fatal(1, "[TB] WATCHDOG TIMEOUT");
    end

endmodule