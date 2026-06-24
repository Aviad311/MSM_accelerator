// =================================================================
// File: RTL/montgomery/field_mul_seq.sv
// =================================================================
// Sequential field multiplication wrapper for secp256k1 Montgomery
// multiplier.
//
// This wrapper converts the multiplier interface:
//
//   in_valid / out_valid
//
// into a simple FSM-friendly interface:
//
//   start / busy / done
//
// Usage:
//   1. Pulse start for one cycle while busy == 0.
//   2. Wrapper captures a,b.
//   3. Wrapper sends one-cycle in_valid to secp256k1_montgomery_mul.
//   4. Wait until done == 1.
//   5. result is valid when done == 1.
//
// For first bring-up, we issue only one multiplication at a time.
// =================================================================

`timescale 1ns/1ps

module field_mul_seq #(
    parameter int WIDTH = 256
) (
    input  logic             clk,
    input  logic             rst_n,

    input  logic             start,
    input  logic [WIDTH-1:0] a,
    input  logic [WIDTH-1:0] b,

    output logic             busy,
    output logic             done,
    output logic [WIDTH-1:0] result
);

    typedef enum logic [1:0] {
        S_IDLE,
        S_ISSUE,
        S_WAIT,
        S_DONE
    } state_t;

    state_t state, next_state;

    logic [WIDTH-1:0] a_reg;
    logic [WIDTH-1:0] b_reg;
    logic [WIDTH-1:0] result_reg;

    logic             mul_in_valid;
    logic [WIDTH-1:0] mul_op_a;
    logic [WIDTH-1:0] mul_op_b;
    logic             mul_out_valid;
    logic [WIDTH-1:0] mul_result;
    logic             mul_ready;

    assign result = result_reg;

    // -------------------------------------------------------------
    // State and data registers
    // -------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            a_reg      <= '0;
            b_reg      <= '0;
            result_reg <= '0;
        end else begin
            state <= next_state;

            // Capture input operands only when accepting a new operation.
            if (state == S_IDLE && start) begin
                a_reg <= a;
                b_reg <= b;
            end

            // Capture multiplier result.
            if (state == S_WAIT && mul_out_valid) begin
                result_reg <= mul_result;
            end
        end
    end

    // -------------------------------------------------------------
    // FSM combinational logic
    // -------------------------------------------------------------
    always_comb begin
        next_state = state;

        busy = 1'b0;
        done = 1'b0;

        mul_in_valid = 1'b0;
        mul_op_a     = a_reg;
        mul_op_b     = b_reg;

        case (state)
            S_IDLE: begin
                busy = 1'b0;

                if (start) begin
                    next_state = S_ISSUE;
                end
            end

            S_ISSUE: begin
                busy         = 1'b1;
                mul_in_valid = 1'b1;
                mul_op_a     = a_reg;
                mul_op_b     = b_reg;

                next_state = S_WAIT;
            end

            S_WAIT: begin
                busy = 1'b1;

                if (mul_out_valid) begin
                    next_state = S_DONE;
                end
            end

            S_DONE: begin
                busy = 1'b0;
                done = 1'b1;

                next_state = S_IDLE;
            end

            default: begin
                next_state = S_IDLE;
            end
        endcase
    end

    // -------------------------------------------------------------
    // Underlying secp256k1 Montgomery multiplier
    // -------------------------------------------------------------
    secp256k1_montgomery_mul #(
        .WIDTH(WIDTH)
    ) u_mul (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (mul_in_valid),
        .op_a      (mul_op_a),
        .op_b      (mul_op_b),
        .out_valid (mul_out_valid),
        .result    (mul_result),
        .ready     (mul_ready)
    );

endmodule