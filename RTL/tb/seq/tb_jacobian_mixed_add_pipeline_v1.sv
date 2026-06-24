`timescale 1ns/1ps

module tb_jacobian_mixed_add_pipeline_v1;

    localparam int WIDTH     = 256;
    localparam int TAG_W     = 16;
    localparam int CTX_COUNT = 40;
    localparam int NUM_OPS   = 256;

    localparam logic [255:0] GX_M =
        256'h9981E643E9089F48979F48C033FD129C231E295329BC66DBD7362E5A487E2097;

    localparam logic [255:0] GY_M =
        256'hCF3F851FD4A582D670B6B59AAC19C1368DFC5D5D1F1DC64DB15EA6D2D3DBABE2;

    localparam logic [255:0] G2_X =
        256'h7C75DD9524177D593C03889B8DCD9B1CB05FB7D2A3DA7FE8BA9F29B104E7DB13;

    localparam logic [255:0] G2_Y =
        256'h55DEBB381F4AD034CC27CB48A46449AAA87D43FDB563384B1CD20838E6FDDC9F;

    localparam logic [255:0] G2_Z =
        256'h9E7F0A3FA94B05ACE16D6B355833826D1BF8BABA3E3B8C9B62BD4DA6A7B75B95;

    localparam logic [255:0] EXP_3G_X =
        256'h019FA59F6F459FC6748FA0A875006844FC39BED026E15B2769CD0E0931000A12;

    localparam logic [255:0] EXP_3G_Y =
        256'hF03F524E8729A2D670F5F5BE0A33EEDC2FC8D898B67B2802B68EF68395ABD131;

    localparam logic [255:0] EXP_3G_Z =
        256'hC2C26ED3E5BE9201DB856E0C5E96B76D5D182C134369ED8ECD3F6A303370697B;

    logic clk;
    logic rst_n;

    logic in_valid;
    logic in_ready;
    logic [TAG_W-1:0] in_tag;
    logic [WIDTH-1:0] in_X1;
    logic [WIDTH-1:0] in_Y1;
    logic [WIDTH-1:0] in_Z1;
    logic [WIDTH-1:0] in_X2;
    logic [WIDTH-1:0] in_Y2;

    logic out_valid;
    logic out_ready;
    logic [TAG_W-1:0] out_tag;
    logic [WIDTH-1:0] out_X3;
    logic [WIDTH-1:0] out_Y3;
    logic [WIDTH-1:0] out_Z3;
    logic out_special;

    logic [$clog2(CTX_COUNT+1)-1:0] active_contexts;

    int cycle_count;
    int send_count;
    int recv_count;
    int first_accept_cycle;
    int last_accept_cycle;
    int first_output_cycle;
    int last_output_cycle;
    int prev_output_cycle;
    int max_output_gap;

    bit seen_tag [0:NUM_OPS-1];

    always #5 clk = ~clk;

    jacobian_mixed_add_pipeline_v1 #(
        .WIDTH(WIDTH),
        .TAG_W(TAG_W),
        .CTX_COUNT(CTX_COUNT),
        .MUL_LATENCY(16)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),

        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_tag(in_tag),

        .in_X1(in_X1),
        .in_Y1(in_Y1),
        .in_Z1(in_Z1),
        .in_X2(in_X2),
        .in_Y2(in_Y2),

        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_tag(out_tag),
        .out_X3(out_X3),
        .out_Y3(out_Y3),
        .out_Z3(out_Z3),
        .out_special(out_special),

        .active_contexts(active_contexts)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 0;
            recv_count <= 0;
            first_output_cycle <= -1;
            last_output_cycle <= -1;
            prev_output_cycle <= -1;
            max_output_gap <= 0;

            for (int k = 0; k < NUM_OPS; k++)
                seen_tag[k] <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;

            if (out_valid && out_ready) begin
                $display("[PIPE_RECV] cycle=%0d tag=%0d active=%0d",
                         cycle_count, out_tag, active_contexts);

                if (out_tag >= NUM_OPS)
                    $fatal(1, "Output tag out of range: %0d", out_tag);

                if (seen_tag[out_tag])
                    $fatal(1, "Duplicate output tag: %0d", out_tag);

                seen_tag[out_tag] <= 1'b1;

                if (out_special)
                    $fatal(1, "Unexpected special-path indication for tag %0d", out_tag);

                if (out_X3 !== EXP_3G_X ||
                    out_Y3 !== EXP_3G_Y ||
                    out_Z3 !== EXP_3G_Z) begin
                    $display("[FAIL] tag=%0d", out_tag);
                    $display("Expected X=%064h", EXP_3G_X);
                    $display("Got      X=%064h", out_X3);
                    $display("Expected Y=%064h", EXP_3G_Y);
                    $display("Got      Y=%064h", out_Y3);
                    $display("Expected Z=%064h", EXP_3G_Z);
                    $display("Got      Z=%064h", out_Z3);
                    $fatal(1, "Pipeline result mismatch");
                end

                if (first_output_cycle < 0)
                    first_output_cycle <= cycle_count;

                if (prev_output_cycle >= 0 &&
                    (cycle_count - prev_output_cycle) > max_output_gap)
                    max_output_gap <= cycle_count - prev_output_cycle;

                prev_output_cycle <= cycle_count;
                last_output_cycle <= cycle_count;
                recv_count <= recv_count + 1;
            end
        end
    end

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;

        in_valid = 1'b0;
        in_tag = '0;
        in_X1 = '0;
        in_Y1 = '0;
        in_Z1 = '0;
        in_X2 = '0;
        in_Y2 = '0;

        out_ready = 1'b1;

        send_count = 0;
        first_accept_cycle = -1;
        last_accept_cycle = -1;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        $display("====================================================");
        $display(" tb_jacobian_mixed_add_pipeline_v1 START");
        $display(" Operations=%0d Contexts=%0d", NUM_OPS, CTX_COUNT);
        $display(" Input vector: 2G + G = 3G, normal path");
        $display("====================================================");

        while (send_count < NUM_OPS) begin
            @(negedge clk);

            in_valid = 1'b1;
            in_tag   = send_count[TAG_W-1:0];
            in_X1    = G2_X;
            in_Y1    = G2_Y;
            in_Z1    = G2_Z;
            in_X2    = GX_M;
            in_Y2    = GY_M;

            @(posedge clk);

            if (in_ready) begin
                $display("[PIPE_SEND] cycle=%0d tag=%0d active=%0d",
                         cycle_count, send_count, active_contexts);

                if (first_accept_cycle < 0)
                    first_accept_cycle = cycle_count;

                last_accept_cycle = cycle_count;
                send_count++;
            end
        end

        @(negedge clk);
        in_valid = 1'b0;
        in_tag = '0;
        in_X1 = '0;
        in_Y1 = '0;
        in_Z1 = '0;
        in_X2 = '0;
        in_Y2 = '0;

        wait (recv_count == NUM_OPS);
        repeat (5) @(posedge clk);

        for (int k = 0; k < NUM_OPS; k++) begin
            if (!seen_tag[k])
                $fatal(1, "Missing output tag %0d", k);
        end

        $display("====================================================");
        $display(" PIPELINE V1 PASSED");
        $display(" accepted=%0d received=%0d", send_count, recv_count);
        $display(" first_accept_cycle=%0d", first_accept_cycle);
        $display(" last_accept_cycle =%0d", last_accept_cycle);
        $display(" first_output_cycle=%0d", first_output_cycle);
        $display(" last_output_cycle =%0d", last_output_cycle);
        $display(" max_output_gap    =%0d", max_output_gap);

        if (NUM_OPS > 1) begin
            $display(" average_accept_interval = %f",
                     real'(last_accept_cycle-first_accept_cycle) /
                     real'(NUM_OPS-1));

            $display(" average_output_interval = %f",
                     real'(last_output_cycle-first_output_cycle) /
                     real'(NUM_OPS-1));
        end

        $display(" Theoretical steady-state lower bound: about 6 cycles/result");
        $display("====================================================");

        $finish;
    end

    initial begin
        #2000000;
        $fatal(1, "Timeout in tb_jacobian_mixed_add_pipeline_v1");
    end

endmodule