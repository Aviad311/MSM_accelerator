To provide an accurate translation, I will first examine the existing `field.sv` and `jacobian.sv` files to understand the project's data types and interface conventions for field operations.

First, I'll read `field.sv` to see how field operations are implemented.
```systemverilog
`timescale 1ns / 1ps

/**
 * Jacobian Point Arithmetic for secp256k1
 * Montgomery-native Datapath
 *
 * This module implements Jacobian point doubling and addition formulas
 * exactly as defined in the hardware-aware Python reference.
 */

// -----------------------------------------------------------------------------
// Package: jacobian_pkg
// Provides constant-multiplication helpers and field arithmetic wrappers.
// -----------------------------------------------------------------------------
package jacobian_pkg;
    import field_pkg::*;

    // Field Addition: (a + b) mod P
    function automatic [255:0] f_add(input [255:0] a, input [255:0] b);
        logic [256:0] sum;
        sum = {1'b0, a} + {1'b0, b};
        return (sum >= {1'b0, P}) ? (sum[255:0] - P) : sum[255:0];
    endfunction

    // Field Subtraction: (a - b) mod P
    function automatic [255:0] f_sub(input [255:0] a, input [255:0] b);
        logic [256:0] diff;
        diff = {1'b0, a} - {1'b0, b};
        return diff[256] ? (diff[255:0] + P) : diff[255:0];
    endfunction

    // Constant Multiplications in Montgomery Domain
    function automatic [255:0] mul2(input [255:0] a); return f_add(a, a); endfunction
    function automatic [255:0] mul3(input [255:0] a); return f_add(a, mul2(a)); endfunction
    function automatic [255:0] mul4(input [255:0] a); return mul2(mul2(a)); endfunction
    function automatic [255:0] mul8(input [255:0] a); return mul2(mul4(a)); endfunction

endpackage

import field_pkg::*;
import jacobian_pkg::*;

// External field multiplier (combinational/pipelined wrapper)
// As seen in workspace conventions, we use an extern function or a structural mapping.
extern function [255:0] f_mul(input [255:0] a, input [255:0] b);

// -----------------------------------------------------------------------------
// Module: jacobian_double
// Formula: (X3, Y3, Z3) = 2 * (X1, Y1, Z1)
// Logical Flow preserved from Python reference.
// -----------------------------------------------------------------------------
module jacobian_double (
    input  logic [255:0] i_X1,
    input  logic [255:0] i_Y1,
    input  logic [255:0] i_Z1,
    output logic [255:0] o_X3,
    output logic [255:0] o_Y3,
    output logic [255:0] o_Z3
);
    logic [255:0] Y1_sq, S, X1_sq, M, X3_int, Y3_int, Z3_int;
    logic [255:0] Y1_sq_sq;

    always_comb begin
        // if Z1 == 0 or Y1 == 0: return INF
        if (i_Z1 == ZERO || i_Y1 == ZERO) begin
            o_X3 = ZERO; o_Y3 = ZERO; o_Z3 = ZERO; // INF representation
        end else begin
            // Y1_sq = f_mul(Y1, Y1)
            Y1_sq = f_mul(i_Y1, i_Y1);
            
            // S = 4 * X1 * Y1^2
            S = mul4(f_mul(i_X1, Y1_sq));

            // M = 3 * X1^2
            X1_sq = f_mul(i_X1, i_X1);
            M = mul3(X1_sq);

            // X3 = M^2 - 2*S
            X3_int = f_sub(f_mul(M, M), mul2(S));

            // Y3 = M*(S - X3) - 8*(Y1^2)^2
            Y1_sq_sq = f_mul(Y1_sq, Y1_sq);
            Y3_int = f_sub(
                f_mul(M, f_sub(S, X3_int)),
                mul8(Y1_sq_sq)
            );

            // Z3 = 2 * Y1 * Z1
            Z3_int = f_mul(mul2(i_Y1), i_Z1);

            o_X3 = X3_int;
            o_Y3 = Y3_int;
            o_Z3 = Z3_int;
        end
    end
endmodule

// -----------------------------------------------------------------------------
// Module: jacobian_mixed_add_mont
// Formula: P1(Jacobian) + Q2(Affine Montgomery, Z2=1)
// -----------------------------------------------------------------------------
module jacobian_mixed_add_mont (
    input  logic [255:0] i_X1, i_Y1, i_Z1,
    input  logic [255:0] i_X2, i_Y2,
    output logic [255:0] o_X3, o_Y3, o_Z3
);
    logic [255:0] Z1_sq, Z1_cu, U2, S2, H, Rr, H_sq, H_cu, X1H2, X3_int, Y3_int, Z3_int;

    // Doubling logic for special case (U2 == X1 and S2 == Y1)
    logic [255:0] d_X, d_Y, d_Z;
    jacobian_double dbl_inst (
        .i_X1(i_X1), .i_Y1(i_Y1), .i_Z1(i_Z1),
        .o_X3(d_X),  .o_Y3(d_Y),  .o_Z3(d_Z)
    );

    always_comb begin
        if (i_Z1 == ZERO) begin
            o_X3 = i_X2; o_Y3 = i_Y2; o_Z3 = ONE_M;
        end else begin
            // Z1^2, Z1^3
            Z1_sq = f_mul(i_Z1, i_Z1);
            U2    = f_mul(i_X2, Z1_sq);
            Z1_cu = f_mul(Z1_sq, i_Z1);
            S2    = f_mul(i_Y2, Z1_cu);

            if (U2 == i_X1) begin
                if (S2 != i_Y1) begin
                    o_X3 = ZERO; o_Y3 = ZERO; o_Z3 = ZERO; // INF
                end else begin
                    o_X3 = d_X; o_Y3 = d_Y; o_Z3 = d_Z;
                end
            end else begin
                H      = f_sub(U2, i_X1);
                Rr     = f_sub(S2, i_Y1);
                H_sq   = f_mul(H, H);
                H_cu   = f_mul(H_sq, H);
                X1H2   = f_mul(i_X1, H_sq);
                X3_int = f_sub(f_sub(f_mul(Rr, Rr), H_cu), mul2(X1H2));
                Y3_int = f_sub(f_mul(Rr, f_sub(X1H2, X3_int)), f_mul(i_Y1, H_cu));
                Z3_int = f_mul(i_Z1, H);

                o_X3 = X3_int; o_Y3 = Y3_int; o_Z3 = Z3_int;
            end
        end
    end
endmodule

// -----------------------------------------------------------------------------
// Module: jacobian_add
// Formula: P1(Jacobian) + P2(Jacobian)
// -----------------------------------------------------------------------------
module jacobian_add (
    input  logic [255:0] i_X1, i_Y1, i_Z1,
    input  logic [255:0] i_X2, i_Y2, i_Z2,
    output logic [255:0] o_X3, o_Y3, o_Z3
);
    logic [255:0] Z2_sq, U1, Z1_sq, U2, Z2_cu, S1, Z1_cu, S2;
    logic [255:0] H, Rr, H_sq, H_cu, U1H2, X3_int, Y3_int, Z3_int;

    // Doubling logic for special case (U1 == U2 and S1 == S2)
    logic [255:0] d_X, d_Y, d_Z;
    jacobian_double dbl_inst (
        .i_X1(i_X1), .i_Y1(i_Y1), .i_Z1(i_Z1),
        .o_X3(d_X),  .o_Y3(d_Y),  .o_Z3(d_Z)
    );

    always_comb begin
        if (i_Z1 == ZERO) begin
            o_X3 = i_X2; o_Y3 = i_Y2; o_Z3 = i_Z2;
        end else if (i_Z2 == ZERO) begin
            o_X3 = i_X1; o_Y3 = i_Y1; o_Z3 = i_Z1;
        end else begin
            Z2_sq = f_mul(i_Z2, i_i_Z2);
            U1    = f_mul(i_X1, Z2_sq);
            Z1_sq = f_mul(i_Z1, i_Z1);
            U2    = f_mul(i_X2, Z1_sq);

            Z2_cu = f_mul(Z2_sq, i_Z2);
            S1    = f_mul(i_Y1, Z2_cu);
            Z1_cu = f_mul(Z1_sq, i_Z1);
            S2    = f_mul(i_Y2, Z1_cu);

            if (U1 == U2) begin
                if (S1 != S2) begin
                    o_X3 = ZERO; o_Y3 = ZERO; o_Z3 = ZERO; // INF
                end else begin
                    o_X3 = d_X; o_Y3 = d_Y; o_Z3 = d_Z;
                end
            end else begin
                H      = f_sub(U2, U1);
                Rr     = f_sub(S2, S1);
                H_sq   = f_mul(H, H);
                H_cu   = f_mul(H_sq, H);
                U1H2   = f_mul(U1, H_sq);
                X3_int = f_sub(f_sub(f_mul(Rr, Rr), H_cu), mul2(U1H2));
                Y3_int = f_sub(f_mul(Rr, f_sub(U1H2, X3_int)), f_mul(S1, H_cu));
                Z3_int = f_mul(f_mul(i_Z1, i_Z2), H);

                o_X3 = X3_int; o_Y3 = Y3_int; o_Z3 = Z3_int;
            end
        end
    end
endmodule

// -----------------------------------------------------------------------------
// Module: jacobian_to_affine
// Logic: Sequential block to handle field inversion and coordinate conversion.
// -----------------------------------------------------------------------------
module jacobian_to_affine (
    input  logic         clk,
    input  logic         reset_n,
    input  logic         start,
    input  logic [255:0] i_X,
    input  logic [255:0] i_Y,
    input  logic [255:0] i_Z,
    output logic [255:0] o_x, // Normal affine x
    output logic [255:0] o_y, // Normal affine y
    output logic         busy,
    output logic         done
);
    typedef enum logic [2:0] {IDLE, INV_WAIT, MUL_Z2, MUL_Z3, MUL_X, MUL_Y, CONV_X, CONV_Y, FINISH} state_t;
    state_t state;

    logic [255:0] z_inv, z_inv2, z_inv3, xM, yM;
    
    // Field Inverter Interface
    logic inv_start, inv_done;
    logic [255:0] inv_out;
    field_inv inverter (
        .clk(clk), .reset_n(reset_n), .start(inv_start),
        .aM(i_Z), .out(inv_out), .busy(), .done(inv_done)
    );

    // Montgomery-to-Normal Conversion logic (from_mont)
    logic [255:0] conv_in, conv_out;
    from_mont conv_inst (.aM(conv_in), .out(conv_out));

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state     <= IDLE;
            inv_start <= 1'b0;
            busy      <= 1'b0;
            done      <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        if (i_Z == ZERO) begin
                            state <= FINISH; // INF handling
                        end else begin
                            inv_start <= 1'b1;
                            busy      <= 1'b1;
                            state     <= INV_WAIT;
                        end
                    end
                end

                INV_WAIT: begin
                    inv_start <= 1'b0;
                    if (inv_done) begin
                        z_inv <= inv_out;
                        state <= MUL_Z2;
                    end
                end

                MUL_Z2: begin
                    z_inv2 <= f_mul(z_inv, z_inv);
                    state  <= MUL_Z3;
                end

                MUL_Z3: begin
                    z_inv3 <= f_mul(z_inv2, z_inv);
                    state  <= MUL_X;
                end

                MUL_X: begin
                    xM    <= f_mul(i_X, z_inv2);
                    state <= MUL_Y;
                end

                MUL_Y: begin
                    yM    <= f_mul(i_Y, z_inv3);
                    state <= CONV_X;
                end

                CONV_X: begin
                    o_x   <= conv_out; // conv_in linked to xM
                    state <= CONV_Y;
                end

                CONV_Y: begin
                    o_y   <= conv_out; // conv_in linked to yM
                    state <= FINISH;
                end

                FINISH: begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

    // Mux for conversion input
    assign conv_in = (state == CONV_X) ? xM : yM;

endmodule
```
