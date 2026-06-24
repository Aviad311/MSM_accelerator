`timescale 1ns/1ps

module tb_secp256k1_montgomery_mul_stream;

    localparam int WIDTH = 256;
    localparam int NUM_TESTS = 8;
    localparam int LATENCY = 16;

    logic clk;
    logic rst_n;

    logic in_valid;
    logic [WIDTH-1:0] op_a;
    logic [WIDTH-1:0] op_b;

    logic out_valid;
    logic [WIDTH-1:0] result;
    logic ready;

    logic [WIDTH-1:0] exp_result [0:NUM_TESTS-1];
    int send_count;
    int recv_count;
    int cycle_count;

    // Montgomery constants / known values
    localparam logic [255:0] ZERO =
        256'h0000000000000000000000000000000000000000000000000000000000000000;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    localparam logic [255:0] GX_M =
        256'h9981E643E9089F48979F48C033FD129C231E295329BC66DBD7362E5A487E2097;

    localparam logic [255:0] GY_M =
        256'hCF3F851FD4A582D670B6B59AAC19C1368DFC5D5D1F1DC64DB15EA6D2D3DBABE2;

    secp256k1_montgomery_mul #(
        .WIDTH(WIDTH)
    ) dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (in_valid),
        .op_a      (op_a),
        .op_b      (op_b),
        .out_valid (out_valid),
        .result    (result),
        .ready     (ready)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    task automatic drive_one(
        input logic [WIDTH-1:0] a,
        input logic [WIDTH-1:0] b,
        input logic [WIDTH-1:0] expected,
        input string name
    );
        begin
            @(negedge clk);

            if (!ready) begin
                $fatal(1, "[%s] DUT not ready", name);
            end

            op_a = a;
            op_b = b;
            in_valid = 1'b1;

            exp_result[send_count] = expected;

            $display("[SEND] idx=%0d cycle=%0d name=%s", send_count, cycle_count, name);

            send_count++;
        end
    endtask

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 0;
            recv_count  <= 0;
        end else begin
            cycle_count <= cycle_count + 1;

            if (out_valid) begin
                $display("[RECV] idx=%0d cycle=%0d result=%064h",
                         recv_count, cycle_count, result);

                if (recv_count >= NUM_TESTS) begin
                    $fatal(1, "Received more outputs than expected");
                end

                if (result !== exp_result[recv_count]) begin
                    $display("[FAIL] idx=%0d", recv_count);
                    $display("[FAIL] expected=%064h", exp_result[recv_count]);
                    $display("[FAIL] got     =%064h", result);
                    $fatal(1, "Streaming multiplier result mismatch");
                end

                recv_count <= recv_count + 1;
            end
        end
    end

    initial begin
        rst_n     = 1'b0;
        in_valid  = 1'b0;
        op_a      = '0;
        op_b      = '0;
        send_count = 0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        $display("====================================================");
        $display(" tb_secp256k1_montgomery_mul_stream START");
        $display(" Goal: prove initiation interval = 1 cycle");
        $display("====================================================");

        // Back-to-back inputs, one new multiplication every cycle.
        drive_one(ZERO,  ZERO,  ZERO,  "zero_times_zero");
        drive_one(ZERO,  ONE_M, ZERO,  "zero_times_one");
        drive_one(ONE_M, ZERO,  ZERO,  "one_times_zero");
        drive_one(ONE_M, ONE_M, ONE_M, "one_times_one");
        drive_one(GX_M,  ONE_M, GX_M,  "gx_times_one");
        drive_one(ONE_M, GX_M,  GX_M,  "one_times_gx");
        drive_one(GY_M,  ONE_M, GY_M,  "gy_times_one");
        drive_one(ONE_M, GY_M,  GY_M,  "one_times_gy");

        @(negedge clk);
        in_valid = 1'b0;
        op_a     = '0;
        op_b     = '0;

        wait (recv_count == NUM_TESTS);
        repeat (2) @(posedge clk);

        $display("====================================================");
        $display(" STREAMING MULTIPLIER PASSED");
        $display(" Accepted %0d back-to-back inputs", NUM_TESTS);
        $display(" Received %0d ordered outputs", NUM_TESTS);
        $display("====================================================");

        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "Timeout in tb_secp256k1_montgomery_mul_stream");
    end

endmodule