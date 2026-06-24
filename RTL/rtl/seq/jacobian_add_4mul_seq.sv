// =================================================================
// File: RTL/seq/jacobian_add_4mul_seq.sv
// =================================================================
// Four-multiplier sequential Jacobian point addition for secp256k1.
//
// This module preserves the start/busy/done interface of
// jacobian_add_seq, but replaces the single shared field multiplier
// with four field_mul_seq instances.
//
// It is intentionally NOT a multi-operation streaming pipeline.
// The Reduce stage has a true dependency between consecutive running
// sums, so the first optimization is to shorten the latency of one
// Jacobian Add by executing independent multiplications in parallel.
//
// Normal-path multiplication schedule:
//
//   Round 1:
//     Z1Z1 = Z1^2
//     Z2Z2 = Z2^2
//
//   Round 2:
//     U1  = X1 * Z2Z2
//     U2  = X2 * Z1Z1
//     Z1C = Z1 * Z1Z1
//     Z2C = Z2 * Z2Z2
//
//   Round 3:
//     S1 = Y1 * Z2C
//     S2 = Y2 * Z1C
//
//   Special-case check:
//     Z1 == 0         -> Q
//     Z2 == 0         -> P
//     H == 0,Rr == 0  -> double(P)
//     H == 0,Rr != 0  -> infinity
//
//   Round 4:
//     HH  = H^2
//     RR  = Rr^2
//     Z12 = Z1 * Z2
//
//   Round 5:
//     HHH = H * HH
//     V   = U1 * HH
//     Z3  = Z12 * H
//
//     X3 is then computed combinationally:
//
//       X3 = RR - HHH - 2*V
//
//   Round 6:
//     YA = Rr * (V - X3)
//     YB = S1 * HHH
//
//     Y3 = YA - YB
//
// All field values are in Montgomery representation.
// =================================================================

`timescale 1ns/1ps

module jacobian_add_4mul_seq #(
    parameter int WIDTH = 256
) (
    input  logic             clk,
    input  logic             rst_n,

    input  logic             start,

    input  logic [WIDTH-1:0] X1,
    input  logic [WIDTH-1:0] Y1,
    input  logic [WIDTH-1:0] Z1,

    input  logic [WIDTH-1:0] X2,
    input  logic [WIDTH-1:0] Y2,
    input  logic [WIDTH-1:0] Z2,

    output logic             busy,
    output logic             done,

    output logic [WIDTH-1:0] X3,
    output logic [WIDTH-1:0] Y3,
    output logic [WIDTH-1:0] Z3
);

    localparam logic [255:0] P =
        256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    function automatic logic [255:0] field_add_mod(
        input logic [255:0] a,
        input logic [255:0] b
    );
        logic [256:0] sum;
        logic [256:0] reduced;
        begin
            sum = {1'b0, a} + {1'b0, b};

            if (sum >= {1'b0, P}) begin
                reduced = sum - {1'b0, P};
                field_add_mod = reduced[255:0];
            end else begin
                field_add_mod = sum[255:0];
            end
        end
    endfunction

    function automatic logic [255:0] field_sub_mod(
        input logic [255:0] a,
        input logic [255:0] b
    );
        logic [256:0] diff;
        begin
            if (a >= b) begin
                field_sub_mod = a - b;
            end else begin
                diff = {1'b0, P} + {1'b0, a} - {1'b0, b};
                field_sub_mod = diff[255:0];
            end
        end
    endfunction

    function automatic logic [255:0] field_double_mod(
        input logic [255:0] a
    );
        begin
            field_double_mod = field_add_mod(a, a);
        end
    endfunction

    typedef enum logic [4:0] {
        S_IDLE,

        S_R1_START,
        S_R1_WAIT,

        S_R2_START,
        S_R2_WAIT,

        S_R3_START,
        S_R3_WAIT,

        S_CHECK_SPECIAL,

        S_DOUBLE_START,
        S_DOUBLE_WAIT,

        S_R4_START,
        S_R4_WAIT,

        S_R5_START,
        S_R5_WAIT,

        S_R6_START,
        S_R6_WAIT,

        S_DONE
    } state_t;

    state_t state;
    state_t next_state;

    logic [255:0] X1_reg;
    logic [255:0] Y1_reg;
    logic [255:0] Z1_reg;

    logic [255:0] X2_reg;
    logic [255:0] Y2_reg;
    logic [255:0] Z2_reg;

    logic [255:0] X3_reg;
    logic [255:0] Y3_reg;
    logic [255:0] Z3_reg;

    logic [255:0] Z1Z1;
    logic [255:0] Z2Z2;

    logic [255:0] U1;
    logic [255:0] U2;

    logic [255:0] Z1C;
    logic [255:0] Z2C;

    logic [255:0] S1;
    logic [255:0] S2;

    logic [255:0] H;
    logic [255:0] Rr;

    logic [255:0] HH;
    logic [255:0] HHH;
    logic [255:0] V;
    logic [255:0] RR;
    logic [255:0] Z12;

    logic [3:0]   mul_start;
    logic [255:0] mul_a [0:3];
    logic [255:0] mul_b [0:3];
    logic [3:0]   mul_busy;
    logic [3:0]   mul_done;
    logic [255:0] mul_result [0:3];

    logic         dbl_start;
    logic         dbl_busy;
    logic         dbl_done;
    logic [255:0] dbl_X3;
    logic [255:0] dbl_Y3;
    logic [255:0] dbl_Z3;

    assign X3 = X3_reg;
    assign Y3 = Y3_reg;
    assign Z3 = Z3_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;

            X1_reg <= '0;
            Y1_reg <= '0;
            Z1_reg <= '0;

            X2_reg <= '0;
            Y2_reg <= '0;
            Z2_reg <= '0;

            X3_reg <= '0;
            Y3_reg <= ONE_M;
            Z3_reg <= '0;

            Z1Z1 <= '0;
            Z2Z2 <= '0;

            U1 <= '0;
            U2 <= '0;

            Z1C <= '0;
            Z2C <= '0;

            S1 <= '0;
            S2 <= '0;

            H  <= '0;
            Rr <= '0;

            HH  <= '0;
            HHH <= '0;
            V   <= '0;
            RR  <= '0;
            Z12 <= '0;
        end else begin
            state <= next_state;

            if (state == S_IDLE && start) begin
                X1_reg <= X1;
                Y1_reg <= Y1;
                Z1_reg <= Z1;

                X2_reg <= X2;
                Y2_reg <= Y2;
                Z2_reg <= Z2;

                X3_reg <= '0;
                Y3_reg <= ONE_M;
                Z3_reg <= '0;

                Z1Z1 <= '0;
                Z2Z2 <= '0;
                U1   <= '0;
                U2   <= '0;
                Z1C  <= '0;
                Z2C  <= '0;
                S1   <= '0;
                S2   <= '0;
                H    <= '0;
                Rr   <= '0;
                HH   <= '0;
                HHH  <= '0;
                V    <= '0;
                RR   <= '0;
                Z12  <= '0;

                if (Z1 == '0) begin
                    X3_reg <= X2;
                    Y3_reg <= Y2;
                    Z3_reg <= Z2;
                end else if (Z2 == '0) begin
                    X3_reg <= X1;
                    Y3_reg <= Y1;
                    Z3_reg <= Z1;
                end
            end

            // Round 1:
            //   m0 = Z1Z1
            //   m1 = Z2Z2
            if (state == S_R1_WAIT &&
                mul_done[0] &&
                mul_done[1]) begin

                Z1Z1 <= mul_result[0];
                Z2Z2 <= mul_result[1];
            end

            // Round 2:
            //   m0 = U1
            //   m1 = U2
            //   m2 = Z1C
            //   m3 = Z2C
            if (state == S_R2_WAIT &&
                mul_done[0] &&
                mul_done[1] &&
                mul_done[2] &&
                mul_done[3]) begin

                U1  <= mul_result[0];
                U2  <= mul_result[1];
                Z1C <= mul_result[2];
                Z2C <= mul_result[3];
            end

            // Round 3:
            //   m0 = S1
            //   m1 = S2
            // Then derive H and Rr.
            if (state == S_R3_WAIT &&
                mul_done[0] &&
                mul_done[1]) begin

                S1 <= mul_result[0];
                S2 <= mul_result[1];

                H  <= field_sub_mod(U2, U1);
                Rr <= field_sub_mod(
                          mul_result[1],
                          mul_result[0]
                      );
            end

            // Opposite points produce infinity.
            if (state == S_CHECK_SPECIAL &&
                H == '0 &&
                Rr != '0) begin

                X3_reg <= '0;
                Y3_reg <= ONE_M;
                Z3_reg <= '0;
            end

            if (state == S_DOUBLE_WAIT && dbl_done) begin
                X3_reg <= dbl_X3;
                Y3_reg <= dbl_Y3;
                Z3_reg <= dbl_Z3;
            end

            // Round 4:
            //   m0 = HH
            //   m1 = RR
            //   m2 = Z12
            if (state == S_R4_WAIT &&
                mul_done[0] &&
                mul_done[1] &&
                mul_done[2]) begin

                HH  <= mul_result[0];
                RR  <= mul_result[1];
                Z12 <= mul_result[2];
            end

            // Round 5:
            //   m0 = HHH
            //   m1 = V
            //   m2 = Z3
            // Then compute X3.
            if (state == S_R5_WAIT &&
                mul_done[0] &&
                mul_done[1] &&
                mul_done[2]) begin

                HHH <= mul_result[0];
                V   <= mul_result[1];

                Z3_reg <= mul_result[2];

                X3_reg <= field_sub_mod(
                              field_sub_mod(
                                  RR,
                                  mul_result[0]
                              ),
                              field_double_mod(
                                  mul_result[1]
                              )
                          );
            end

            // Round 6:
            //   m0 = YA = Rr * (V - X3)
            //   m1 = YB = S1 * HHH
            // Then compute Y3.
            if (state == S_R6_WAIT &&
                mul_done[0] &&
                mul_done[1]) begin

                Y3_reg <= field_sub_mod(
                              mul_result[0],
                              mul_result[1]
                          );
            end
        end
    end

    always_comb begin
        next_state = state;

        busy = 1'b0;
        done = 1'b0;

        mul_start = '0;

        for (int i = 0; i < 4; i = i + 1) begin
            mul_a[i] = '0;
            mul_b[i] = '0;
        end

        dbl_start = 1'b0;

        unique case (state)

            S_IDLE: begin
                if (start) begin
                    if (Z1 == '0 || Z2 == '0) begin
                        next_state = S_DONE;
                    end else begin
                        next_state = S_R1_START;
                    end
                end
            end

            S_R1_START: begin
                busy = 1'b1;

                mul_start[0] = 1'b1;
                mul_a[0]     = Z1_reg;
                mul_b[0]     = Z1_reg;

                mul_start[1] = 1'b1;
                mul_a[1]     = Z2_reg;
                mul_b[1]     = Z2_reg;

                next_state = S_R1_WAIT;
            end

            S_R1_WAIT: begin
                busy = 1'b1;

                if (mul_done[0] && mul_done[1])
                    next_state = S_R2_START;
            end

            S_R2_START: begin
                busy = 1'b1;

                mul_start[0] = 1'b1;
                mul_a[0]     = X1_reg;
                mul_b[0]     = Z2Z2;

                mul_start[1] = 1'b1;
                mul_a[1]     = X2_reg;
                mul_b[1]     = Z1Z1;

                mul_start[2] = 1'b1;
                mul_a[2]     = Z1_reg;
                mul_b[2]     = Z1Z1;

                mul_start[3] = 1'b1;
                mul_a[3]     = Z2_reg;
                mul_b[3]     = Z2Z2;

                next_state = S_R2_WAIT;
            end

            S_R2_WAIT: begin
                busy = 1'b1;

                if (mul_done[0] &&
                    mul_done[1] &&
                    mul_done[2] &&
                    mul_done[3]) begin

                    next_state = S_R3_START;
                end
            end

            S_R3_START: begin
                busy = 1'b1;

                mul_start[0] = 1'b1;
                mul_a[0]     = Y1_reg;
                mul_b[0]     = Z2C;

                mul_start[1] = 1'b1;
                mul_a[1]     = Y2_reg;
                mul_b[1]     = Z1C;

                next_state = S_R3_WAIT;
            end

            S_R3_WAIT: begin
                busy = 1'b1;

                if (mul_done[0] && mul_done[1])
                    next_state = S_CHECK_SPECIAL;
            end

            S_CHECK_SPECIAL: begin
                busy = 1'b1;

                if (H == '0 && Rr == '0) begin
                    next_state = S_DOUBLE_START;
                end else if (H == '0) begin
                    next_state = S_DONE;
                end else begin
                    next_state = S_R4_START;
                end
            end

            S_DOUBLE_START: begin
                busy      = 1'b1;
                dbl_start = 1'b1;
                next_state = S_DOUBLE_WAIT;
            end

            S_DOUBLE_WAIT: begin
                busy = 1'b1;

                if (dbl_done)
                    next_state = S_DONE;
            end

            S_R4_START: begin
                busy = 1'b1;

                mul_start[0] = 1'b1;
                mul_a[0]     = H;
                mul_b[0]     = H;

                mul_start[1] = 1'b1;
                mul_a[1]     = Rr;
                mul_b[1]     = Rr;

                mul_start[2] = 1'b1;
                mul_a[2]     = Z1_reg;
                mul_b[2]     = Z2_reg;

                next_state = S_R4_WAIT;
            end

            S_R4_WAIT: begin
                busy = 1'b1;

                if (mul_done[0] &&
                    mul_done[1] &&
                    mul_done[2]) begin

                    next_state = S_R5_START;
                end
            end

            S_R5_START: begin
                busy = 1'b1;

                mul_start[0] = 1'b1;
                mul_a[0]     = H;
                mul_b[0]     = HH;

                mul_start[1] = 1'b1;
                mul_a[1]     = U1;
                mul_b[1]     = HH;

                mul_start[2] = 1'b1;
                mul_a[2]     = Z12;
                mul_b[2]     = H;

                next_state = S_R5_WAIT;
            end

            S_R5_WAIT: begin
                busy = 1'b1;

                if (mul_done[0] &&
                    mul_done[1] &&
                    mul_done[2]) begin

                    next_state = S_R6_START;
                end
            end

            S_R6_START: begin
                busy = 1'b1;

                mul_start[0] = 1'b1;
                mul_a[0]     = Rr;
                mul_b[0]     = field_sub_mod(V, X3_reg);

                mul_start[1] = 1'b1;
                mul_a[1]     = S1;
                mul_b[1]     = HHH;

                next_state = S_R6_WAIT;
            end

            S_R6_WAIT: begin
                busy = 1'b1;

                if (mul_done[0] && mul_done[1])
                    next_state = S_DONE;
            end

            S_DONE: begin
                done       = 1'b1;
                next_state = S_IDLE;
            end

            default: begin
                next_state = S_IDLE;
            end

        endcase
    end

    genvar mi;
    generate
        for (mi = 0; mi < 4; mi = mi + 1) begin : g_mul
            field_mul_seq #(
                .WIDTH(WIDTH)
            ) u_field_mul (
                .clk    (clk),
                .rst_n  (rst_n),
                .start  (mul_start[mi]),
                .a      (mul_a[mi]),
                .b      (mul_b[mi]),
                .busy   (mul_busy[mi]),
                .done   (mul_done[mi]),
                .result (mul_result[mi])
            );
        end
    endgenerate

    jacobian_double_seq #(
        .WIDTH(WIDTH)
    ) u_double (
        .clk   (clk),
        .rst_n (rst_n),

        .start (dbl_start),

        .X1    (X1_reg),
        .Y1    (Y1_reg),
        .Z1    (Z1_reg),

        .busy  (dbl_busy),
        .done  (dbl_done),

        .X3    (dbl_X3),
        .Y3    (dbl_Y3),
        .Z3    (dbl_Z3)
    );

endmodule