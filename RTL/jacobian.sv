`timescale 1ns / 1ps

import field_pkg::*;

package jacobian_pkg;import field_pkg::*;

function automatic [255:0] f_add(input [255:0] a, input [255:0] b);
    logic [256:0] sum;
    begin
        sum = {1'b0,a} + {1'b0,b};
        f_add = (sum >= {1'b0,P}) ? (sum[255:0] - P) : sum[255:0];
    end
endfunction

function automatic [255:0] f_sub(input [255:0] a, input [255:0] b);
    logic [256:0] diff;
    begin
        diff = {1'b0,a} - {1'b0,b};
        f_sub = diff[256] ? (diff[255:0] + P) : diff[255:0];
    end
endfunction

function automatic [255:0] mul2(input [255:0] a);
    mul2 = f_add(a,a);
endfunction

function automatic [255:0] mul3(input [255:0] a);
    mul3 = f_add(a,mul2(a));
endfunction

function automatic [255:0] mul4(input [255:0] a);
    mul4 = mul2(mul2(a));
endfunction

function automatic [255:0] mul8(input [255:0] a);
    mul8 = mul2(mul4(a));
endfunction

endpackage

import jacobian_pkg::*;

// ============================================================// Jacobian point doubling// ============================================================module jacobian_double (input  logic [255:0] i_X1,input  logic [255:0] i_Y1,input  logic [255:0] i_Z1,output logic [255:0] o_X3,output logic [255:0] o_Y3,output logic [255:0] o_Z3);

logic [255:0] Y1_sq, S, X1_sq, M;
logic [255:0] M_sq, two_S;
logic [255:0] X3_int, Y3_int, Z3_int;
logic [255:0] Y1_sq_sq;
logic [255:0] tmp1, tmp2;

field_mul u_mul_y_sq (.a(i_Y1), .b(i_Y1), .out(Y1_sq));
field_mul u_mul_x_y  (.a(i_X1), .b(Y1_sq), .out(tmp1));
field_mul u_mul_x_sq (.a(i_X1), .b(i_X1), .out(X1_sq));
field_mul u_mul_m_sq (.a(M), .b(M), .out(M_sq));
field_mul u_mul_y4   (.a(Y1_sq), .b(Y1_sq), .out(Y1_sq_sq));
field_mul u_mul_m_t  (.a(M), .b(tmp2), .out(tmp1_y));
field_mul u_mul_z    (.a(mul2(i_Y1)), .b(i_Z1), .out(Z3_int));

logic [255:0] tmp1_y;

always_comb begin
    if (i_Z1 == ZERO || i_Y1 == ZERO) begin
        o_X3 = ZERO;
        o_Y3 = ZERO;
        o_Z3 = ZERO;
    end else begin
        S       = mul4(tmp1);
        M       = mul3(X1_sq);
        two_S   = mul2(S);
        X3_int  = f_sub(M_sq, two_S);
        tmp2    = f_sub(S, X3_int);
        Y3_int  = f_sub(tmp1_y, mul8(Y1_sq_sq));

        o_X3 = X3_int;
        o_Y3 = Y3_int;
        o_Z3 = Z3_int;
    end
end

endmodule

// ============================================================// Jacobian + affine Montgomery point// ============================================================module jacobian_mixed_add_mont (input  logic [255:0] i_X1,input  logic [255:0] i_Y1,input  logic [255:0] i_Z1,input  logic [255:0] i_X2,input  logic [255:0] i_Y2,output logic [255:0] o_X3,output logic [255:0] o_Y3,output logic [255:0] o_Z3);

logic [255:0] Z1_sq, Z1_cu, U2, S2;
logic [255:0] H, Rr, H_sq, H_cu, X1H2;
logic [255:0] X3_int, Y3_int, Z3_int;
logic [255:0] d_X, d_Y, d_Z;
logic [255:0] Rr_sq, two_X1H2, tmp_y1, tmp_y2;

field_mul u_z1sq  (.a(i_Z1), .b(i_Z1), .out(Z1_sq));
field_mul u_u2    (.a(i_X2), .b(Z1_sq), .out(U2));
field_mul u_z1cu  (.a(Z1_sq), .b(i_Z1), .out(Z1_cu));
field_mul u_s2    (.a(i_Y2), .b(Z1_cu), .out(S2));
field_mul u_hsq   (.a(H), .b(H), .out(H_sq));
field_mul u_hcu   (.a(H_sq), .b(H), .out(H_cu));
field_mul u_x1h2  (.a(i_X1), .b(H_sq), .out(X1H2));
field_mul u_rrsq  (.a(Rr), .b(Rr), .out(Rr_sq));
field_mul u_y1hcu (.a(i_Y1), .b(H_cu), .out(tmp_y2));
field_mul u_z3    (.a(i_Z1), .b(H), .out(Z3_int));
field_mul u_y3mul (.a(Rr), .b(tmp_y1), .out(tmp_y1_mul));

logic [255:0] tmp_y1_mul;

jacobian_double u_double (
    .i_X1(i_X1),
    .i_Y1(i_Y1),
    .i_Z1(i_Z1),
    .o_X3(d_X),
    .o_Y3(d_Y),
    .o_Z3(d_Z)
);

always_comb begin
    if (i_Z1 == ZERO) begin
        o_X3 = i_X2;
        o_Y3 = i_Y2;
        o_Z3 = ONE_M;
    end else begin
        H  = f_sub(U2, i_X1);
        Rr = f_sub(S2, i_Y1);

        if (U2 == i_X1) begin
            if (S2 != i_Y1) begin
                o_X3 = ZERO;
                o_Y3 = ZERO;
                o_Z3 = ZERO;
            end else begin
                o_X3 = d_X;
                o_Y3 = d_Y;
                o_Z3 = d_Z;
            end
        end else begin
            two_X1H2 = mul2(X1H2);
            X3_int   = f_sub(f_sub(Rr_sq, H_cu), two_X1H2);
            tmp_y1   = f_sub(X1H2, X3_int);
            Y3_int   = f_sub(tmp_y1_mul, tmp_y2);

            o_X3 = X3_int;
            o_Y3 = Y3_int;
            o_Z3 = Z3_int;
        end
    end
end

endmodule

// ============================================================// Jacobian + Jacobian point// ============================================================module jacobian_add (input  logic [255:0] i_X1,input  logic [255:0] i_Y1,input  logic [255:0] i_Z1,input  logic [255:0] i_X2,input  logic [255:0] i_Y2,input  logic [255:0] i_Z2,output logic [255:0] o_X3,output logic [255:0] o_Y3,output logic [255:0] o_Z3);

logic [255:0] Z2_sq, U1, Z1_sq, U2;
logic [255:0] Z2_cu, S1, Z1_cu, S2;
logic [255:0] H, Rr, H_sq, H_cu, U1H2;
logic [255:0] X3_int, Y3_int, Z3_int;
logic [255:0] d_X, d_Y, d_Z;
logic [255:0] Rr_sq, two_U1H2, tmp_y1, tmp_y1_mul, tmp_y2;
logic [255:0] z1z2;

field_mul u_z2sq  (.a(i_Z2), .b(i_Z2), .out(Z2_sq));
field_mul u_u1    (.a(i_X1), .b(Z2_sq), .out(U1));
field_mul u_z1sq  (.a(i_Z1), .b(i_Z1), .out(Z1_sq));
field_mul u_u2    (.a(i_X2), .b(Z1_sq), .out(U2));

field_mul u_z2cu  (.a(Z2_sq), .b(i_Z2), .out(Z2_cu));
field_mul u_s1    (.a(i_Y1), .b(Z2_cu), .out(S1));
field_mul u_z1cu  (.a(Z1_sq), .b(i_Z1), .out(Z1_cu));
field_mul u_s2    (.a(i_Y2), .b(Z1_cu), .out(S2));

field_mul u_hsq   (.a(H), .b(H), .out(H_sq));
field_mul u_hcu   (.a(H_sq), .b(H), .out(H_cu));
field_mul u_u1h2  (.a(U1), .b(H_sq), .out(U1H2));
field_mul u_rrsq  (.a(Rr), .b(Rr), .out(Rr_sq));
field_mul u_y3mul (.a(Rr), .b(tmp_y1), .out(tmp_y1_mul));
field_mul u_s1hcu (.a(S1), .b(H_cu), .out(tmp_y2));
field_mul u_z1z2  (.a(i_Z1), .b(i_Z2), .out(z1z2));
field_mul u_z3    (.a(z1z2), .b(H), .out(Z3_int));

jacobian_double u_double (
    .i_X1(i_X1),
    .i_Y1(i_Y1),
    .i_Z1(i_Z1),
    .o_X3(d_X),
    .o_Y3(d_Y),
    .o_Z3(d_Z)
);

always_comb begin
    if (i_Z1 == ZERO) begin
        o_X3 = i_X2;
        o_Y3 = i_Y2;
        o_Z3 = i_Z2;
    end else if (i_Z2 == ZERO) begin
        o_X3 = i_X1;
        o_Y3 = i_Y1;
        o_Z3 = i_Z1;
    end else begin
        H  = f_sub(U2, U1);
        Rr = f_sub(S2, S1);

        if (U1 == U2) begin
            if (S1 != S2) begin
                o_X3 = ZERO;
                o_Y3 = ZERO;
                o_Z3 = ZERO;
            end else begin
                o_X3 = d_X;
                o_Y3 = d_Y;
                o_Z3 = d_Z;
            end
        end else begin
            two_U1H2 = mul2(U1H2);
            X3_int   = f_sub(f_sub(Rr_sq, H_cu), two_U1H2);
            tmp_y1   = f_sub(U1H2, X3_int);
            Y3_int   = f_sub(tmp_y1_mul, tmp_y2);

            o_X3 = X3_int;
            o_Y3 = Y3_int;
            o_Z3 = Z3_int;
        end
    end
end

endmodule