`timescale 1ns/1ps

module tb_bucket_update_pipeline_with_sram_macro_stress_v1;

    localparam int ADDR_W        = 13;
    localparam int DATA_W        = 256;
    localparam int DEPTH         = 8192;
    localparam int GEN_W         = 16;
    localparam int SLOT_COUNT    = 16;
    localparam int MIX_CTX_COUNT = 40;
    localparam int MUL_LATENCY   = 16;

    localparam logic [GEN_W-1:0] TEST_GEN = 16'h0042;

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
    int unsigned total_outputs;
    int unsigned bucket7_seen;
    int unsigned bucket8_seen;
    int unsigned bucket9_seen;
    int unsigned bucket0_seen;

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
        int unsigned wait_cycles;
        begin
            wait_cycles = 0;

            @(negedge clk);
            in_valid     = 1'b1;
            in_bucket_id = bucket_id;
            in_point_x   = point_x;
            in_point_y   = point_y;

            while (1) begin
                @(posedge clk);
                if (in_ready)
                    break;
                wait_cycles++;
            end

            $display(
                "[TB] INPUT ACCEPTED cycle=%0d bucket=%0d wait_cycles=%0d",
                cycle_count,
                bucket_id,
                wait_cycles
            );

            @(negedge clk);
            in_valid     = 1'b0;
            in_bucket_id = '0;
            in_point_x   = '0;
            in_point_y   = '0;
        end
    endtask

    task automatic check_direct_g;
        begin
            if (out_skipped !== 1'b0 ||
                out_direct_write !== 1'b1 ||
                out_mixed_add !== 1'b0 ||
                out_x !== GX_M ||
                out_y !== GY_M ||
                out_z !== ONE_M) begin
                $fatal(1, "[TB] Direct-write output mismatch");
            end
        end
    endtask

    task automatic check_mixed_2g;
        begin
            if (out_skipped !== 1'b0 ||
                out_direct_write !== 1'b0 ||
                out_mixed_add !== 1'b1 ||
                out_x !== EXP_2G_X ||
                out_y !== EXP_2G_Y ||
                out_z !== EXP_2G_Z) begin
                $display("[TB] expected X=%064h", EXP_2G_X);
                $display("[TB] got      X=%064h", out_x);
                $display("[TB] expected Y=%064h", EXP_2G_Y);
                $display("[TB] got      Y=%064h", out_y);
                $display("[TB] expected Z=%064h", EXP_2G_Z);
                $display("[TB] got      Z=%064h", out_z);
                $fatal(1, "[TB] Mixed-add output mismatch");
            end
        end
    endtask

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            total_outputs <= 0;
            bucket7_seen  <= 0;
            bucket8_seen  <= 0;
            bucket9_seen  <= 0;
            bucket0_seen  <= 0;
        end else if (out_valid && out_ready) begin
            case (out_bucket_id)
                13'd7: begin
                    if (bucket7_seen == 0)
                        check_direct_g();
                    else if (bucket7_seen == 1)
                        check_mixed_2g();
                    else
                        $fatal(1, "[TB] Unexpected extra output for bucket 7");
                    bucket7_seen <= bucket7_seen + 1;
                end

                13'd8: begin
                    if (bucket8_seen == 0)
                        check_direct_g();
                    else if (bucket8_seen == 1)
                        check_mixed_2g();
                    else
                        $fatal(1, "[TB] Unexpected extra output for bucket 8");
                    bucket8_seen <= bucket8_seen + 1;
                end

                13'd9: begin
                    if (bucket9_seen != 0)
                        $fatal(1, "[TB] Unexpected extra output for bucket 9");
                    check_direct_g();
                    bucket9_seen <= bucket9_seen + 1;
                end

                13'd0: begin
                    if (bucket0_seen != 0)
                        $fatal(1, "[TB] Unexpected extra output for bucket 0");

                    if (out_skipped !== 1'b1 ||
                        out_direct_write !== 1'b0 ||
                        out_mixed_add !== 1'b0 ||
                        out_x !== ZERO ||
                        out_y !== ONE_M ||
                        out_z !== ZERO) begin
                        $fatal(1, "[TB] Bucket-zero skip output mismatch");
                    end
                    bucket0_seen <= bucket0_seen + 1;
                end

                default:
                    $fatal(1, "[TB] Unexpected output bucket=%0d", out_bucket_id);
            endcase

            total_outputs <= total_outputs + 1;

            $display(
                "[TB] OUTPUT PASSED cycle=%0d bucket=%0d direct=%b mixed=%b skipped=%b",
                cycle_count,
                out_bucket_id,
                out_direct_write,
                out_mixed_add,
                out_skipped
            );
        end
    end

    initial begin
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
        $display("[TB] Starting pipeline + SRAM macro stress test");
        $display("============================================================");
        // First bucket-7 request enters the pipeline.
        send_update(13'd7, GX_M, GY_M);

// This request is presented while bucket 7 is still busy.
// send_update holds in_valid until in_ready rises, so this creates
// measurable same-bucket stall cycles.
        send_update(13'd7, GX_M, GY_M);

// Independent buckets continue afterward.
        send_update(13'd8, GX_M, GY_M);
        send_update(13'd9, GX_M, GY_M);

// Second update to bucket 8 must execute MixedAdd.
        send_update(13'd8, GX_M, GY_M);

// Bucket zero is skipped.
        send_update(13'd0, GX_M, GY_M);
        

        wait (total_outputs == 6);
        repeat (3) @(posedge clk);

        if (bucket7_seen !== 2)
            $fatal(1, "[TB] bucket7_seen expected 2 got %0d", bucket7_seen);

        if (bucket8_seen !== 2)
            $fatal(1, "[TB] bucket8_seen expected 2 got %0d", bucket8_seen);

        if (bucket9_seen !== 1)
            $fatal(1, "[TB] bucket9_seen expected 1 got %0d", bucket9_seen);

        if (bucket0_seen !== 1)
            $fatal(1, "[TB] bucket0_seen expected 1 got %0d", bucket0_seen);

        if (accepted_count !== 64'd6)
            $fatal(1, "[TB] accepted_count expected 6 got %0d", accepted_count);

        if (completed_count !== 64'd6)
            $fatal(1, "[TB] completed_count expected 6 got %0d", completed_count);

        if (direct_write_count !== 64'd3)
            $fatal(1, "[TB] direct_write_count expected 3 got %0d",
                   direct_write_count);

        if (mixed_add_count !== 64'd2)
            $fatal(1, "[TB] mixed_add_count expected 2 got %0d",
                   mixed_add_count);

        if (same_bucket_stall_count == 0)
            $fatal(1, "[TB] Expected at least one same-bucket stall cycle");

        if (active_slots !== '0)
            $fatal(1, "[TB] active_slots expected 0 got %0d", active_slots);

        $display("");
        $display("============================================================");
        $display("[TB] pipeline + SRAM macro stress test PASSED");
        $display("[TB] Outputs                  = %0d", total_outputs);
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
        #50000;
        $fatal(1, "[TB] WATCHDOG TIMEOUT");
    end

endmodule