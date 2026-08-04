`timescale 1ns/1ps

module tb_msm_affine_frontend_4win_v1;

    localparam int NUM_WINDOWS = 4;

    localparam logic [255:0] GX =
        256'h79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798;

    localparam logic [255:0] GY =
        256'h483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8;

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

    logic [1:0]        window_index;
    logic              busy;
    logic              done;
    logic [255:0]      result_x;
    logic [255:0]      result_y;
    logic [255:0]      result_z;

    logic              converter_busy;
    logic [5:0]        converter_pending_count;
    logic [5:0]        converter_result_count;

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

    longint unsigned cycle_count;

    initial clk = 1'b0;
    always #5 clk = ~clk;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cycle_count <= 0;
        else
            cycle_count <= cycle_count + 1;
    end

    msm_affine_frontend_top #(
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
        .NUM_WINDOWS     (NUM_WINDOWS),
        .CONV_FIFO_DEPTH (32)
    ) dut (
        .clk                     (clk),
        .rst_n                   (rst_n),
        .start                   (start),
        .in_valid                (in_valid),
        .in_ready                (in_ready),
        .in_bucket_id            (in_bucket_id),
        .in_point_x              (in_point_x),
        .in_point_y              (in_point_y),
        .last_point              (last_point),
        .window_index            (window_index),
        .busy                    (busy),
        .done                    (done),
        .result_x                (result_x),
        .result_y                (result_y),
        .result_z                (result_z),
        .converter_busy          (converter_busy),
        .converter_pending_count (converter_pending_count),
        .converter_result_count  (converter_result_count)
    );

    jacobian_double_seq #(.WIDTH(256)) u_ref_double (
        .clk(clk), .rst_n(rst_n), .start(ref_dbl_start),
        .X1(ref_acc_x), .Y1(ref_acc_y), .Z1(ref_acc_z),
        .busy(ref_dbl_busy), .done(ref_dbl_done),
        .X3(ref_dbl_x), .Y3(ref_dbl_y), .Z3(ref_dbl_z)
    );

    jacobian_add_4mul_seq #(.WIDTH(256)) u_ref_add (
        .clk(clk), .rst_n(rst_n), .start(ref_add_start),
        .X1(ref_acc_x), .Y1(ref_acc_y), .Z1(ref_acc_z),
        .X2(GX_M), .Y2(GY_M), .Z2(ONE_M),
        .busy(ref_add_busy), .done(ref_add_done),
        .X3(ref_add_x), .Y3(ref_add_y), .Z3(ref_add_z)
    );

    task automatic send_single_g(input int expected_window);
        begin
            while (!(in_ready && window_index == expected_window[1:0]))
                @(posedge clk);

            @(negedge clk);
            in_valid     = 1'b1;
            in_bucket_id = 16'd1;
            in_point_x   = GX;
            in_point_y   = GY;
            last_point   = 1'b1;

            do @(posedge clk); while (!in_ready);

            $display("[TB_AFFINE_4WIN] Sent normal-domain G to window=%0d at cycle=%0d",
                     expected_window, cycle_count);

            @(negedge clk);
            in_valid     = 1'b0;
            in_bucket_id = '0;
            in_point_x   = '0;
            in_point_y   = '0;
            last_point   = 1'b0;
        end
    endtask

    task automatic ref_double_16;
        integer i;
        begin
            for (i = 0; i < 16; i = i + 1) begin
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
        end
    endtask

    task automatic ref_add_g;
        begin
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

    task automatic build_reference;
        integer w;
        begin
            ref_acc_x = GX_M;
            ref_acc_y = GY_M;
            ref_acc_z = ONE_M;
            for (w = NUM_WINDOWS-2; w >= 0; w = w - 1) begin
                ref_double_16();
                ref_add_g();
            end
            $display("[TB_AFFINE_4WIN] Independent 4-window reference complete.");
        end
    endtask

    initial begin
        rst_n         = 1'b0;
        start         = 1'b0;
        in_valid      = 1'b0;
        in_bucket_id  = '0;
        in_point_x    = '0;
        in_point_y    = '0;
        last_point    = 1'b0;
        ref_dbl_start = 1'b0;
        ref_add_start = 1'b0;
        ref_acc_x     = '0;
        ref_acc_y     = ONE_M;
        ref_acc_z     = '0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;

        fork
            build_reference();
        join_none

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        send_single_g(3);
        send_single_g(2);
        send_single_g(1);
        send_single_g(0);

        while (!done)
            @(posedge clk);

        #0.1;

        while (ref_acc_z == '0)
            @(posedge clk);

        if (converter_busy !== 1'b0) begin
            $display("[TB_AFFINE_4WIN] converter pending=%0d result=%0d",
                     converter_pending_count, converter_result_count);
            $fatal(1, "[TB_AFFINE_4WIN] Converter still busy after MSM done.");
        end

        if (result_x !== ref_acc_x ||
            result_y !== ref_acc_y ||
            result_z !== ref_acc_z) begin
            $display("[TB_AFFINE_4WIN] FAILED");
            $display("[TB_AFFINE_4WIN] expected X=%064h", ref_acc_x);
            $display("[TB_AFFINE_4WIN] actual   X=%064h", result_x);
            $display("[TB_AFFINE_4WIN] expected Y=%064h", ref_acc_y);
            $display("[TB_AFFINE_4WIN] actual   Y=%064h", result_y);
            $display("[TB_AFFINE_4WIN] expected Z=%064h", ref_acc_z);
            $display("[TB_AFFINE_4WIN] actual   Z=%064h", result_z);
            $fatal(1, "[TB_AFFINE_4WIN] Four-window affine frontend result mismatch.");
        end

        $display("");
        $display("============================================================");
        $display("[TB_AFFINE_4WIN] AFFINE FRONTEND FOUR-WINDOW PASSED");
        $display("[TB_AFFINE_4WIN] input points were normal-domain G");
        $display("[TB_AFFINE_4WIN] converter produced Montgomery points internally");
        $display("[TB_AFFINE_4WIN] verified = G*2^48 + G*2^32 + G*2^16 + G");
        $display("[TB_AFFINE_4WIN] done cycle = %0d", cycle_count);
        $display("============================================================");

        #20;
        $finish;
    end

    initial begin
        #(64'd400000000);
        $fatal(1, "[TB_AFFINE_4WIN] Watchdog timeout.");
    end

endmodule