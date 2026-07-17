`timescale 1ns/1ps

module point_to_montgomery_stream #(
    parameter int DATA_W       = 256,
    parameter int BUCKET_W     = 16,
    parameter int MUL_LATENCY  = 16,
    parameter int FIFO_DEPTH   = 32
) (
    input  logic                 clk,
    input  logic                 rst_n,

    // Normal-domain affine input stream
    input  logic                 in_valid,
    output logic                 in_ready,
    input  logic [DATA_W-1:0]    in_point_x,
    input  logic [DATA_W-1:0]    in_point_y,
    input  logic [BUCKET_W-1:0]  in_bucket_id,
    input  logic                 in_last_point,

    // Montgomery-domain affine output stream
    output logic                 out_valid,
    input  logic                 out_ready,
    output logic [DATA_W-1:0]    out_point_x_m,
    output logic [DATA_W-1:0]    out_point_y_m,
    output logic [BUCKET_W-1:0]  out_bucket_id,
    output logic                 out_last_point,

    // Optional status/debug
    output logic                 busy,
    output logic [$clog2(FIFO_DEPTH+1)-1:0] pending_count_dbg,
    output logic [$clog2(FIFO_DEPTH+1)-1:0] result_count_dbg
);

    localparam int PTR_W   = (FIFO_DEPTH <= 1) ? 1 : $clog2(FIFO_DEPTH);
    localparam int COUNT_W = $clog2(FIFO_DEPTH + 1);

    // secp256k1 R^2 mod p.
    // MontMul(a, R2) = a * R^2 * R^-1 = a * R mod p.
    localparam logic [DATA_W-1:0] R2 =
        256'h000000000000000000000000000000000000000000000001000007A2000E90A1;

    // Metadata FIFO for conversions currently inside the multiplier pipelines.
    logic [BUCKET_W-1:0] pending_bucket [0:FIFO_DEPTH-1];
    logic                pending_last   [0:FIFO_DEPTH-1];
    logic [PTR_W-1:0]    pending_wr_ptr;
    logic [PTR_W-1:0]    pending_rd_ptr;
    logic [COUNT_W-1:0]  pending_count;

    // Result FIFO decouples the non-stallable multiplier outputs from
    // downstream ready/valid backpressure.
    logic [DATA_W-1:0]   result_x      [0:FIFO_DEPTH-1];
    logic [DATA_W-1:0]   result_y      [0:FIFO_DEPTH-1];
    logic [BUCKET_W-1:0] result_bucket [0:FIFO_DEPTH-1];
    logic                result_last   [0:FIFO_DEPTH-1];
    logic [PTR_W-1:0]    result_wr_ptr;
    logic [PTR_W-1:0]    result_rd_ptr;
    logic [COUNT_W-1:0]  result_count;

    logic                mul_x_ready;
    logic                mul_y_ready;
    logic                mul_x_out_valid;
    logic                mul_y_out_valid;
    logic [DATA_W-1:0]   mul_x_result;
    logic [DATA_W-1:0]   mul_y_result;

    logic                accept_in;
    logic                accept_out;
    logic                conversion_done;
    logic [COUNT_W:0]    outstanding_total;

    function automatic logic [PTR_W-1:0] ptr_inc(
        input logic [PTR_W-1:0] ptr
    );
        if (ptr == FIFO_DEPTH-1)
            ptr_inc = '0;
        else
            ptr_inc = ptr + 1'b1;
    endfunction

    assign outstanding_total = pending_count + result_count;

    // Reserve one result-FIFO slot for every accepted point.
    // The conservative rule below deliberately does not use a same-cycle
    // output pop to accept one additional input.
    assign in_ready = mul_x_ready &&
                      mul_y_ready &&
                      (outstanding_total < FIFO_DEPTH);

    assign accept_in      = in_valid && in_ready;
    assign out_valid      = (result_count != 0);
    assign accept_out     = out_valid && out_ready;
    assign conversion_done = mul_x_out_valid && mul_y_out_valid;

    assign out_point_x_m  = result_x[result_rd_ptr];
    assign out_point_y_m  = result_y[result_rd_ptr];
    assign out_bucket_id  = result_bucket[result_rd_ptr];
    assign out_last_point = result_last[result_rd_ptr];

    assign busy = (outstanding_total != 0);

    assign pending_count_dbg = pending_count;
    assign result_count_dbg  = result_count;

    secp256k1_montgomery_mul u_mul_x (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (accept_in),
        .op_a      (in_point_x),
        .op_b      (R2),
        .out_valid (mul_x_out_valid),
        .result    (mul_x_result),
        .ready     (mul_x_ready)
    );

    secp256k1_montgomery_mul u_mul_y (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (accept_in),
        .op_a      (in_point_y),
        .op_b      (R2),
        .out_valid (mul_y_out_valid),
        .result    (mul_y_result),
        .ready     (mul_y_ready)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pending_wr_ptr <= '0;
            pending_rd_ptr <= '0;
            pending_count  <= '0;

            result_wr_ptr   <= '0;
            result_rd_ptr   <= '0;
            result_count    <= '0;
        end else begin
            // Both multiplier instances are identical and receive the same
            // valid pulse, so their out_valid pulses must remain aligned.
            if (mul_x_out_valid != mul_y_out_valid) begin
                $fatal(1, "[MONT_CONV] X/Y multiplier out_valid misalignment");
            end

            if (conversion_done && (pending_count == 0)) begin
                $fatal(1, "[MONT_CONV] conversion completed without metadata");
            end

            // Accept a new point and queue its metadata.
            if (accept_in) begin
                pending_bucket[pending_wr_ptr] <= in_bucket_id;
                pending_last[pending_wr_ptr]   <= in_last_point;
                pending_wr_ptr                 <= ptr_inc(pending_wr_ptr);
            end

            // Completed X/Y conversion: move result + oldest metadata
            // into the downstream result FIFO.
            if (conversion_done) begin
                result_x[result_wr_ptr]      <= mul_x_result;
                result_y[result_wr_ptr]      <= mul_y_result;
                result_bucket[result_wr_ptr] <= pending_bucket[pending_rd_ptr];
                result_last[result_wr_ptr]   <= pending_last[pending_rd_ptr];

                result_wr_ptr  <= ptr_inc(result_wr_ptr);
                pending_rd_ptr <= ptr_inc(pending_rd_ptr);
            end

            // Downstream consumed one converted point.
            if (accept_out) begin
                result_rd_ptr <= ptr_inc(result_rd_ptr);
            end

            unique case ({accept_in, conversion_done})
                2'b10: pending_count <= pending_count + 1'b1;
                2'b01: pending_count <= pending_count - 1'b1;
                default: pending_count <= pending_count;
            endcase

            unique case ({conversion_done, accept_out})
                2'b10: result_count <= result_count + 1'b1;
                2'b01: result_count <= result_count - 1'b1;
                default: result_count <= result_count;
            endcase
        end
    end

    // MUL_LATENCY is retained as an architectural/documentation parameter.
    // Metadata alignment is driven by the multiplier's actual out_valid,
    // so correctness does not depend on manually shifting by this value.
    initial begin
        if (DATA_W != 256)
            $fatal(1, "[MONT_CONV] Current R2 constant requires DATA_W=256");
        if (FIFO_DEPTH < 2)
            $fatal(1, "[MONT_CONV] FIFO_DEPTH must be at least 2");
        if (MUL_LATENCY < 1)
            $fatal(1, "[MONT_CONV] MUL_LATENCY must be positive");
    end

endmodule