`timescale 1ns/1ps

module tb_pippenger_window_8lane_sram_macro_smoke_v2;

    localparam int ADDR_W          = 16;
    localparam int DATA_W          = 256;
    localparam int DEPTH           = 65536;
    localparam int SRAM_RD_LATENCY = 1;
    localparam int GEN_W           = 16;

    localparam int FIFO_DEPTH      = 16;
    localparam int SLOT_COUNT      = 16;
    localparam int MIX_CTX_COUNT   = 40;
    localparam int MUL_LATENCY     = 16;

    localparam logic [255:0] GX_M =
        256'h9981E643E9089F48979F48C033FD129C231E295329BC66DBD7362E5A487E2097;

    localparam logic [255:0] GY_M =
        256'hCF3F851FD4A582D670B6B59AAC19C1368DFC5D5D1F1DC64DB15EA6D2D3DBABE2;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    logic clk;
    logic rst_n;

    logic start;

    logic                in_valid;
    logic                in_ready;
    logic [ADDR_W-1:0]   in_bucket_id;
    logic [DATA_W-1:0]   in_point_x;
    logic [DATA_W-1:0]   in_point_y;
    logic                last_point;

    logic                busy;
    logic                done;

    logic [DATA_W-1:0]   result_x;
    logic [DATA_W-1:0]   result_y;
    logic [DATA_W-1:0]   result_z;

    longint unsigned cycle_count;
    longint unsigned start_cycle;
    longint unsigned input_cycle;
    longint unsigned done_cycle;

    pippenger_window_mem_stream_top_8lane_pipeline_reduce4mul_overlap_sram_macro_v2 #(
        .ADDR_W          (ADDR_W),
        .DATA_W          (DATA_W),
        .DEPTH           (DEPTH),
        .SRAM_RD_LATENCY (SRAM_RD_LATENCY),
        .GEN_W           (GEN_W),
        .FIFO_DEPTH      (FIFO_DEPTH),
        .SLOT_COUNT      (SLOT_COUNT),
        .MIX_CTX_COUNT   (MIX_CTX_COUNT),
        .MUL_LATENCY     (MUL_LATENCY)
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

    initial clk = 1'b0;
    always #5 clk = ~clk;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cycle_count <= 0;
        else
            cycle_count <= cycle_count + 1;
    end

    task automatic pulse_start;
        begin
            @(negedge clk);
            start = 1'b1;

            @(negedge clk);
            start = 1'b0;

            start_cycle = cycle_count;

            $display(
                "[TB] start accepted near cycle=%0d",
                start_cycle
            );
        end
    endtask

    task automatic send_single_point;
        begin
            wait (in_ready === 1'b1);

            @(negedge clk);
            in_valid     = 1'b1;
            in_bucket_id = 16'd1;
            in_point_x   = GX_M;
            in_point_y   = GY_M;
            last_point   = 1'b1;

            do begin
                @(posedge clk);
            end while (!(in_valid && in_ready));

            input_cycle = cycle_count;

            $display(
                "[TB] single G accepted cycle=%0d bucket=1 lane=1 local_addr=0",
                input_cycle
            );

            @(negedge clk);
            in_valid     = 1'b0;
            in_bucket_id = '0;
            in_point_x   = '0;
            in_point_y   = '0;
            last_point   = 1'b0;
        end
    endtask

    initial begin
        rst_n        = 1'b0;
        start        = 1'b0;
        in_valid     = 1'b0;
        in_bucket_id = '0;
        in_point_x   = '0;
        in_point_y   = '0;
        last_point   = 1'b0;

        start_cycle = 0;
        input_cycle = 0;
        done_cycle  = 0;

        repeat (5) @(posedge clk);

        @(negedge clk);
        rst_n = 1'b1;

        $display("");
        $display("============================================================");
        $display("[TB] Waiting for 8-bank SRAM tag initialization");
        $display("============================================================");

        wait (busy === 1'b0);

        $display(
            "[TB] Tag initialization complete cycle=%0d",
            cycle_count
        );

        pulse_start();
        send_single_point();

        wait (done === 1'b1);
        #1;

        done_cycle = cycle_count;

        if (result_x !== GX_M ||
            result_y !== GY_M ||
            result_z !== ONE_M) begin

            $display("[TB] RESULT MISMATCH");
            $display("[TB] EXPECTED X = %064h", GX_M);
            $display("[TB] GOT      X = %064h", result_x);
            $display("[TB] EXPECTED Y = %064h", GY_M);
            $display("[TB] GOT      Y = %064h", result_y);
            $display("[TB] EXPECTED Z = %064h", ONE_M);
            $display("[TB] GOT      Z = %064h", result_z);

            $fatal(1, "[TB] Full 8-lane fully macro-backed SRAM smoke test failed");
        end

        $display("");
        $display("============================================================");
        $display("[TB] 8-lane fully macro-backed SRAM full-window smoke test PASSED");
        $display("[TB] Input case       = one G in global bucket 1");
        $display("[TB] Expected result  = G");
        $display("[TB] Start cycle      = %0d", start_cycle);
        $display("[TB] Input cycle      = %0d", input_cycle);
        $display("[TB] Done cycle       = %0d", done_cycle);
        $display("[TB] Window latency   = %0d cycles",
                 done_cycle - start_cycle);
        $display("[TB] Total cycles     = %0d", cycle_count);
        $display("============================================================");

        #20;
        $finish;
    end

    initial begin
        #10000000;
        $fatal(1, "[TB] WATCHDOG TIMEOUT");
    end

endmodule