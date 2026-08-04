`timescale 1ns/1ps

module tb_point_to_montgomery_stream;

    localparam int DATA_W      = 256;
    localparam int BUCKET_W    = 16;
    localparam int FIFO_DEPTH  = 8;
    localparam int NUM_TESTS   = 6;

    logic clk;
    logic rst_n;

    logic                in_valid;
    logic                in_ready;
    logic [DATA_W-1:0]   in_point_x;
    logic [DATA_W-1:0]   in_point_y;
    logic [BUCKET_W-1:0] in_bucket_id;
    logic                in_last_point;

    logic                out_valid;
    logic                out_ready;
    logic [DATA_W-1:0]   out_point_x_m;
    logic [DATA_W-1:0]   out_point_y_m;
    logic [BUCKET_W-1:0] out_bucket_id;
    logic                out_last_point;

    logic                busy;
    logic [$clog2(FIFO_DEPTH+1)-1:0] pending_count_dbg;
    logic [$clog2(FIFO_DEPTH+1)-1:0] result_count_dbg;

    int sent_count;
    int recv_count;
    int cycle_count;

    logic [255:0] src_x [0:NUM_TESTS-1];
    logic [255:0] src_y [0:NUM_TESTS-1];
    logic [15:0]  src_b [0:NUM_TESTS-1];
    logic         src_l [0:NUM_TESTS-1];

    logic [255:0] exp_x [0:NUM_TESTS-1];
    logic [255:0] exp_y [0:NUM_TESTS-1];

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    localparam logic [255:0] GX =
        256'h79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798;
    localparam logic [255:0] GY =
        256'h483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8;

    localparam logic [255:0] GX_M =
        256'h9981E643E9089F48979F48C033FD129C231E295329BC66DBD7362E5A487E2097;
    localparam logic [255:0] GY_M =
        256'hCF3F851FD4A582D670B6B59AAC19C1368DFC5D5D1F1DC64DB15EA6D2D3DBABE2;

    point_to_montgomery_stream #(
        .DATA_W      (DATA_W),
        .BUCKET_W    (BUCKET_W),
        .MUL_LATENCY (16),
        .FIFO_DEPTH  (FIFO_DEPTH)
    ) dut (
        .clk               (clk),
        .rst_n             (rst_n),

        .in_valid          (in_valid),
        .in_ready          (in_ready),
        .in_point_x        (in_point_x),
        .in_point_y        (in_point_y),
        .in_bucket_id      (in_bucket_id),
        .in_last_point     (in_last_point),

        .out_valid         (out_valid),
        .out_ready         (out_ready),
        .out_point_x_m     (out_point_x_m),
        .out_point_y_m     (out_point_y_m),
        .out_bucket_id     (out_bucket_id),
        .out_last_point    (out_last_point),

        .busy              (busy),
        .pending_count_dbg (pending_count_dbg),
        .result_count_dbg  (result_count_dbg)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    always @(posedge clk) begin
        if (!rst_n)
            cycle_count <= 0;
        else
            cycle_count <= cycle_count + 1;
    end

    task automatic drive_one(input int idx);
        begin
            in_valid      <= 1'b1;
            in_point_x    <= src_x[idx];
            in_point_y    <= src_y[idx];
            in_bucket_id  <= src_b[idx];
            in_last_point <= src_l[idx];

            do @(posedge clk); while (!in_ready);

            sent_count <= sent_count + 1;

            // Keep protocol clean after the accepted edge.
            in_valid      <= 1'b0;
            in_point_x    <= '0;
            in_point_y    <= '0;
            in_bucket_id  <= '0;
            in_last_point <= 1'b0;
        end
    endtask

    task automatic check_output(input int idx);
        begin
            if (out_point_x_m !== exp_x[idx]) begin
                $display("[FAIL] idx=%0d X expected=%064h got=%064h",
                         idx, exp_x[idx], out_point_x_m);
                $fatal(1);
            end

            if (out_point_y_m !== exp_y[idx]) begin
                $display("[FAIL] idx=%0d Y expected=%064h got=%064h",
                         idx, exp_y[idx], out_point_y_m);
                $fatal(1);
            end

            if (out_bucket_id !== src_b[idx]) begin
                $display("[FAIL] idx=%0d bucket expected=%0d got=%0d",
                         idx, src_b[idx], out_bucket_id);
                $fatal(1);
            end

            if (out_last_point !== src_l[idx]) begin
                $display("[FAIL] idx=%0d last expected=%0b got=%0b",
                         idx, src_l[idx], out_last_point);
                $fatal(1);
            end

            $display("[PASS] idx=%0d bucket=%0d last=%0b cycle=%0d",
                     idx, out_bucket_id, out_last_point, cycle_count);
        end
    endtask

    // Scoreboard: outputs must remain ordered.
    always @(posedge clk) begin
        if (!rst_n) begin
            recv_count <= 0;
        end else if (out_valid && out_ready) begin
            check_output(recv_count);
            recv_count <= recv_count + 1;
        end
    end

    initial begin
        rst_n         = 1'b0;
        in_valid      = 1'b0;
        in_point_x    = '0;
        in_point_y    = '0;
        in_bucket_id  = '0;
        in_last_point = 1'b0;
        out_ready     = 1'b0;
        sent_count    = 0;
        recv_count    = 0;
        cycle_count   = 0;

        // 0 converts to 0.
        src_x[0] = 256'd0;
        src_y[0] = 256'd0;
        src_b[0] = 16'd1;
        src_l[0] = 1'b0;
        exp_x[0] = 256'd0;
        exp_y[0] = 256'd0;

        // Normal-domain 1 converts to Montgomery ONE_M.
        src_x[1] = 256'd1;
        src_y[1] = 256'd1;
        src_b[1] = 16'd2;
        src_l[1] = 1'b0;
        exp_x[1] = ONE_M;
        exp_y[1] = ONE_M;

        // secp256k1 generator, normal affine -> Montgomery affine.
        src_x[2] = GX;
        src_y[2] = GY;
        src_b[2] = 16'd3;
        src_l[2] = 1'b0;
        exp_x[2] = GX_M;
        exp_y[2] = GY_M;

        // Repeated values verify ordering and metadata alignment.
        src_x[3] = 256'd1;
        src_y[3] = 256'd0;
        src_b[3] = 16'd17;
        src_l[3] = 1'b0;
        exp_x[3] = ONE_M;
        exp_y[3] = 256'd0;

        src_x[4] = GX;
        src_y[4] = 256'd1;
        src_b[4] = 16'd65535;
        src_l[4] = 1'b0;
        exp_x[4] = GX_M;
        exp_y[4] = ONE_M;

        src_x[5] = GX;
        src_y[5] = GY;
        src_b[5] = 16'd9;
        src_l[5] = 1'b1;
        exp_x[5] = GX_M;
        exp_y[5] = GY_M;

        repeat (5) @(posedge clk);
        rst_n <= 1'b1;
        repeat (2) @(posedge clk);

        // Initially block the output. This forces completed conversions
        // to accumulate in the result FIFO.
        out_ready <= 1'b0;

        // Send all inputs as quickly as the converter allows.
        for (int i = 0; i < NUM_TESTS; i++) begin
            drive_one(i);
        end

        // Keep backpressure asserted after multiplier completion.
        repeat (25) @(posedge clk);

        // Output payload must remain stable while valid && !ready.
        if (out_valid) begin
            logic [255:0] held_x;
            logic [255:0] held_y;
            logic [15:0]  held_b;
            logic         held_l;

            held_x = out_point_x_m;
            held_y = out_point_y_m;
            held_b = out_bucket_id;
            held_l = out_last_point;

            repeat (4) begin
                @(posedge clk);
                if (!out_valid ||
                    out_point_x_m !== held_x ||
                    out_point_y_m !== held_y ||
                    out_bucket_id !== held_b ||
                    out_last_point !== held_l) begin
                    $fatal(1, "[FAIL] Output changed while stalled");
                end
            end
            $display("[PASS] output stability under backpressure");
        end else begin
            $fatal(1, "[FAIL] Expected queued output during backpressure");
        end

        out_ready <= 1'b1;

        fork
            begin : timeout_block
                repeat (500) @(posedge clk);
                $fatal(1, "[TIMEOUT] recv=%0d sent=%0d pending=%0d result=%0d",
                       recv_count, sent_count, pending_count_dbg, result_count_dbg);
            end
            begin : completion_block
                wait (recv_count == NUM_TESTS);
                @(posedge clk);

                if (busy !== 1'b0)
                    $fatal(1, "[FAIL] busy remained high after all outputs drained");

                $display("==================================================");
                $display("[TB] tb_point_to_montgomery_stream PASSED");
                $display("[TB] sent=%0d received=%0d", sent_count, recv_count);
                $display("==================================================");
                disable timeout_block;
                $finish;
            end
        join
    end

endmodule