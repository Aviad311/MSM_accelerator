`timescale 1ns/1ps

module tb_jacobian_mixed_add_pipeline_v2;

    localparam int WIDTH     = 256;
    localparam int TAG_W     = 16;
    localparam int CTX_COUNT = 40;
    localparam int NUM_OPS   = 16;

    localparam logic [255:0] P =
        256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F;

    localparam logic [255:0] ZERO =
        256'h0;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    localparam logic [255:0] GX_M =
        256'h9981E643E9089F48979F48C033FD129C231E295329BC66DBD7362E5A487E2097;

    localparam logic [255:0] GY_M =
        256'hCF3F851FD4A582D670B6B59AAC19C1368DFC5D5D1F1DC64DB15EA6D2D3DBABE2;

    localparam logic [255:0] NEG_GY_M = P - GY_M;

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

    typedef enum logic [1:0] {
        K_NORMAL,
        K_INF_SHORTCUT,
        K_DOUBLE,
        K_INVERSE
    } kind_t;

    logic clk;
    logic rst_n;

    logic in_valid;
    logic in_ready;
    logic [TAG_W-1:0] in_tag;
    logic [WIDTH-1:0] in_X1, in_Y1, in_Z1, in_X2, in_Y2;

    logic out_valid;
    logic out_ready;
    logic [TAG_W-1:0] out_tag;
    logic [WIDTH-1:0] out_X3, out_Y3, out_Z3;
    logic out_special;
    logic [$clog2(CTX_COUNT+1)-1:0] active_contexts;

    kind_t expected_kind [0:NUM_OPS-1];
    bit seen [0:NUM_OPS-1];

    int cycle_count;
    int send_count;
    int recv_count;
    int stall_cycles;

    initial clk = 1'b0;
    always #5 clk = ~clk;

    jacobian_mixed_add_pipeline_v2 #(
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

    task automatic set_case(input int idx);
        begin
            in_tag = idx[TAG_W-1:0];

            case (idx % 4)
                0: begin
                    expected_kind[idx] = K_NORMAL;
                    in_X1 = G2_X; in_Y1 = G2_Y; in_Z1 = G2_Z;
                    in_X2 = GX_M; in_Y2 = GY_M;
                end

                1: begin
                    expected_kind[idx] = K_INF_SHORTCUT;
                    in_X1 = ZERO; in_Y1 = ONE_M; in_Z1 = ZERO;
                    in_X2 = GX_M; in_Y2 = GY_M;
                end

                2: begin
                    expected_kind[idx] = K_DOUBLE;
                    in_X1 = GX_M; in_Y1 = GY_M; in_Z1 = ONE_M;
                    in_X2 = GX_M; in_Y2 = GY_M;
                end

                default: begin
                    expected_kind[idx] = K_INVERSE;
                    in_X1 = GX_M; in_Y1 = GY_M; in_Z1 = ONE_M;
                    in_X2 = GX_M; in_Y2 = NEG_GY_M;
                end
            endcase
        end
    endtask

    always_comb begin
        if (!rst_n)
            out_ready = 1'b0;
        else
            out_ready = ((cycle_count % 11) < 8);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 0;
            recv_count   <= 0;
            stall_cycles <= 0;
            for (int k = 0; k < NUM_OPS; k++)
                seen[k] <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;

            if (out_valid && !out_ready)
                stall_cycles <= stall_cycles + 1;

            if (out_valid && out_ready) begin
                if (out_tag >= NUM_OPS)
                    $fatal(1, "Tag out of range: %0d", out_tag);
                if (seen[out_tag])
                    $fatal(1, "Duplicate tag: %0d", out_tag);
                if (out_special)
                    $fatal(1, "v2 returned unexpected out_special for tag %0d",
                           out_tag);

                seen[out_tag] <= 1'b1;

                case (expected_kind[out_tag])
                    K_NORMAL: begin
                        if (out_X3 !== EXP_3G_X ||
                            out_Y3 !== EXP_3G_Y ||
                            out_Z3 !== EXP_3G_Z)
                            $fatal(1, "Normal mismatch tag=%0d", out_tag);
                    end

                    K_INF_SHORTCUT: begin
                        if (out_X3 !== GX_M ||
                            out_Y3 !== GY_M ||
                            out_Z3 !== ONE_M)
                            $fatal(1, "Infinity shortcut mismatch tag=%0d",
                                   out_tag);
                    end

                    K_DOUBLE: begin
                        if (out_X3 !== G2_X ||
                            out_Y3 !== G2_Y ||
                            out_Z3 !== G2_Z) begin
                            $display("DOUBLE GOT X=%064h", out_X3);
                            $display("DOUBLE GOT Y=%064h", out_Y3);
                            $display("DOUBLE GOT Z=%064h", out_Z3);
                            $fatal(1, "Doubling mismatch tag=%0d", out_tag);
                        end
                    end

                    K_INVERSE: begin
                        if (out_X3 !== ZERO ||
                            out_Y3 !== ONE_M ||
                            out_Z3 !== ZERO)
                            $fatal(1, "Inverse mismatch tag=%0d", out_tag);
                    end
                endcase

                $display("[V2_RECV] cycle=%0d tag=%0d kind=%0d active=%0d",
                         cycle_count, out_tag, expected_kind[out_tag],
                         active_contexts);

                recv_count <= recv_count + 1;
            end
        end
    end

    initial begin
        rst_n = 1'b0;
        in_valid = 1'b0;
        in_tag = '0;
        in_X1 = '0; in_Y1 = '0; in_Z1 = '0;
        in_X2 = '0; in_Y2 = '0;
        send_count = 0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        $display("====================================================");
        $display(" tb_jacobian_mixed_add_pipeline_v2 START");
        $display(" normal + Z1=0 + true doubling + inverse + backpressure");
        $display("====================================================");

        while (send_count < NUM_OPS) begin
            @(negedge clk);
            in_valid = 1'b1;
            set_case(send_count);

            @(posedge clk);
            if (in_ready) begin
                $display("[V2_SEND] cycle=%0d tag=%0d kind=%0d active=%0d",
                         cycle_count, send_count, expected_kind[send_count],
                         active_contexts);
                send_count++;
            end
        end

        @(negedge clk);
        in_valid = 1'b0;

        wait (recv_count == NUM_OPS);
        repeat (5) @(posedge clk);

        for (int k = 0; k < NUM_OPS; k++) begin
            if (!seen[k])
                $fatal(1, "Missing tag %0d", k);
        end

        $display("====================================================");
        $display(" PIPELINE V2 PASSED");
        $display(" accepted=%0d received=%0d", send_count, recv_count);
        $display(" output_stall_cycles=%0d", stall_cycles);
        $display("====================================================");
        $finish;
    end

    initial begin
        #4000000;
        $fatal(1, "Timeout in tb_jacobian_mixed_add_pipeline_v2");
    end

endmodule