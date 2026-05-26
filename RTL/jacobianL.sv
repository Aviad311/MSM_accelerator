
module jacobian_doubleL (
    input  logic [255:0] i_X1,
    input  logic [255:0] i_Y1,
    input  logic [255:0] i_Z1,
    output logic [255:0] o_X3,
    output logic [255:0] o_Y3,
    output logic [255:0] o_Z3
);

    localparam logic [255:0] P    = 256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F;
    localparam logic [255:0] ZERO = 256'h0;

    function automatic logic [255:0] f_add(input logic [255:0] a, input logic [255:0] b);
        logic [256:0] sum;
        begin
            sum = {1'b0, a} + {1'b0, b};
            if (sum >= {1'b0, P})
                f_add = sum[255:0] - P;
            else
                f_add = sum[255:0];
        end
    endfunction

    function automatic logic [255:0] f_sub(input logic [255:0] a, input logic [255:0] b);
        logic [256:0] diff;
        begin
            diff = {1'b0, a} - {1'b0, b};
            if (diff[256])
                f_sub = diff[255:0] + P;
            else
                f_sub = diff[255:0];
        end
    endfunction

    function automatic logic [255:0] mul2(input logic [255:0] a);
        begin
            mul2 = f_add(a, a);
        end
    endfunction

    function automatic logic [255:0] mul3(input logic [255:0] a);
        begin
            mul3 = f_add(a, mul2(a));
        end
    endfunction

    function automatic logic [255:0] mul4(input logic [255:0] a);
        begin
            mul4 = mul2(mul2(a));
        end
    endfunction

    function automatic logic [255:0] mul8(input logic [255:0] a);
        begin
            mul8 = mul2(mul4(a));
        end
    endfunction

    logic [255:0] Y1_sq;
    logic [255:0] X1_Y1_sq;
    logic [255:0] X1_sq;
    logic [255:0] M;
    logic [255:0] M_sq;
    logic [255:0] S;
    logic [255:0] two_S;
    logic [255:0] tmp2;
    logic [255:0] M_tmp2;
    logic [255:0] Y1_4;
    logic [255:0] Z3_int;
    logic [255:0] X3_int;
    logic [255:0] Y3_int;

    field_mul u_mul_y_sq (
        .a   (i_Y1),
        .b   (i_Y1),
        .out (Y1_sq)
    );

    field_mul u_mul_x_y_sq (
        .a   (i_X1),
        .b   (Y1_sq),
        .out (X1_Y1_sq)
    );

    field_mul u_mul_x_sq (
        .a   (i_X1),
        .b   (i_X1),
        .out (X1_sq)
    );

    field_mul u_mul_m_sq (
        .a   (M),
        .b   (M),
        .out (M_sq)
    );

    field_mul u_mul_m_tmp2 (
        .a   (M),
        .b   (tmp2),
        .out (M_tmp2)
    );

    field_mul u_mul_y1_4 (
        .a   (Y1_sq),
        .b   (Y1_sq),
        .out (Y1_4)
    );

    field_mul u_mul_z3 (
        .a   (mul2(i_Y1)),
        .b   (i_Z1),
        .out (Z3_int)
    );

    always_comb begin
        if ((i_Z1 == ZERO) || (i_Y1 == ZERO)) begin
            o_X3 = ZERO;
            o_Y3 = ZERO;
            o_Z3 = ZERO;
        end else begin
            S      = mul4(X1_Y1_sq);
            M      = mul3(X1_sq);
            two_S  = mul2(S);
            X3_int = f_sub(M_sq, two_S);
            tmp2   = f_sub(S, X3_int);
            Y3_int = f_sub(M_tmp2, mul8(Y1_4));

            o_X3 = X3_int;
            o_Y3 = Y3_int;
            o_Z3 = Z3_int;
        end
    end

endmodule





module jacobian_mixed_addL (
    input  logic [255:0] i_X1,
    input  logic [255:0] i_Y1,
    input  logic [255:0] i_Z1,

    input  logic [255:0] i_X2,
    input  logic [255:0] i_Y2,

    output logic [255:0] o_X3,
    output logic [255:0] o_Y3,
    output logic [255:0] o_Z3
);

    localparam logic [255:0] P =
        256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F;

    localparam logic [255:0] ZERO =
        256'h0;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;


    function automatic logic [255:0] f_add(input logic [255:0] a, input logic [255:0] b);
        logic [256:0] sum;
        begin
            sum = {1'b0, a} + {1'b0, b};
            if (sum >= {1'b0, P})
                f_add = sum[255:0] - P;
            else
                f_add = sum[255:0];
        end
    endfunction


    function automatic logic [255:0] f_sub(input logic [255:0] a, input logic [255:0] b);
        logic [256:0] diff;
        begin
            diff = {1'b0, a} - {1'b0, b};
            if (diff[256])
                f_sub = diff[255:0] + P;
            else
                f_sub = diff[255:0];
        end
    endfunction


    function automatic logic [255:0] mul2(input logic [255:0] a);
        begin
            mul2 = f_add(a, a);
        end
    endfunction


    logic [255:0] Z1_sq;
    logic [255:0] U2;
    logic [255:0] Z1_cu;
    logic [255:0] S2;

    logic [255:0] H;
    logic [255:0] Rr;
    logic [255:0] H_sq;
    logic [255:0] H_cu;
    logic [255:0] X1H2;

    logic [255:0] Rr_sq;
    logic [255:0] two_X1H2;
    logic [255:0] X3_int;
    logic [255:0] tmp_y;
    logic [255:0] Rr_tmp_y;
    logic [255:0] Y1_H_cu;
    logic [255:0] Y3_int;
    logic [255:0] Z3_int;

    logic [255:0] dbl_X;
    logic [255:0] dbl_Y;
    logic [255:0] dbl_Z;


    field_mul u_z1_sq (
        .a   (i_Z1),
        .b   (i_Z1),
        .out (Z1_sq)
    );

    field_mul u_u2 (
        .a   (i_X2),
        .b   (Z1_sq),
        .out (U2)
    );

    field_mul u_z1_cu (
        .a   (Z1_sq),
        .b   (i_Z1),
        .out (Z1_cu)
    );

    field_mul u_s2 (
        .a   (i_Y2),
        .b   (Z1_cu),
        .out (S2)
    );

    field_mul u_h_sq (
        .a   (H),
        .b   (H),
        .out (H_sq)
    );

    field_mul u_h_cu (
        .a   (H_sq),
        .b   (H),
        .out (H_cu)
    );

    field_mul u_x1h2 (
        .a   (i_X1),
        .b   (H_sq),
        .out (X1H2)
    );

    field_mul u_rr_sq (
        .a   (Rr),
        .b   (Rr),
        .out (Rr_sq)
    );

    field_mul u_rr_tmp_y (
        .a   (Rr),
        .b   (tmp_y),
        .out (Rr_tmp_y)
    );

    field_mul u_y1_hcu (
        .a   (i_Y1),
        .b   (H_cu),
        .out (Y1_H_cu)
    );

    field_mul u_z3 (
        .a   (i_Z1),
        .b   (H),
        .out (Z3_int)
    );


    jacobian_doubleL u_double (
        .i_X1 (i_X1),
        .i_Y1 (i_Y1),
        .i_Z1 (i_Z1),
        .o_X3 (dbl_X),
        .o_Y3 (dbl_Y),
        .o_Z3 (dbl_Z)
    );


    always_comb begin
        H          = f_sub(U2, i_X1);
        Rr         = f_sub(S2, i_Y1);
        two_X1H2   = mul2(X1H2);
        X3_int     = f_sub(f_sub(Rr_sq, H_cu), two_X1H2);
        tmp_y      = f_sub(X1H2, X3_int);
        Y3_int     = f_sub(Rr_tmp_y, Y1_H_cu);

        if (i_Z1 == ZERO) begin
            o_X3 = i_X2;
            o_Y3 = i_Y2;
            o_Z3 = ONE_M;
        end else if (U2 == i_X1) begin
            if (S2 != i_Y1) begin
                o_X3 = ZERO;
                o_Y3 = ZERO;
                o_Z3 = ZERO;
            end else begin
                o_X3 = dbl_X;
                o_Y3 = dbl_Y;
                o_Z3 = dbl_Z;
            end
        end else begin
            o_X3 = X3_int;
            o_Y3 = Y3_int;
            o_Z3 = Z3_int;
        end
    end

endmodule
module jacobian_addL (
    input  logic [255:0] i_X1,
    input  logic [255:0] i_Y1,
    input  logic [255:0] i_Z1,

    input  logic [255:0] i_X2,
    input  logic [255:0] i_Y2,
    input  logic [255:0] i_Z2,

    output logic [255:0] o_X3,
    output logic [255:0] o_Y3,
    output logic [255:0] o_Z3
);

    localparam logic [255:0] P =
        256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F;

    localparam logic [255:0] ZERO =
        256'h0;


    function automatic logic [255:0] f_add(input logic [255:0] a, input logic [255:0] b);
        logic [256:0] sum;
        begin
            sum = {1'b0, a} + {1'b0, b};
            if (sum >= {1'b0, P})
                f_add = sum[255:0] - P;
            else
                f_add = sum[255:0];
        end
    endfunction


    function automatic logic [255:0] f_sub(input logic [255:0] a, input logic [255:0] b);
        logic [256:0] diff;
        begin
            diff = {1'b0, a} - {1'b0, b};
            if (diff[256])
                f_sub = diff[255:0] + P;
            else
                f_sub = diff[255:0];
        end
    endfunction


    function automatic logic [255:0] mul2(input logic [255:0] a);
        begin
            mul2 = f_add(a, a);
        end
    endfunction


    logic [255:0] Z2_sq;
    logic [255:0] U1;
    logic [255:0] Z1_sq;
    logic [255:0] U2;

    logic [255:0] Z2_cu;
    logic [255:0] S1;
    logic [255:0] Z1_cu;
    logic [255:0] S2;

    logic [255:0] H;
    logic [255:0] Rr;
    logic [255:0] H_sq;
    logic [255:0] H_cu;
    logic [255:0] U1H2;

    logic [255:0] Rr_sq;
    logic [255:0] two_U1H2;
    logic [255:0] X3_int;
    logic [255:0] tmp_y;
    logic [255:0] Rr_tmp_y;
    logic [255:0] S1_H_cu;
    logic [255:0] Y3_int;

    logic [255:0] Z1Z2;
    logic [255:0] Z3_int;

    logic [255:0] dbl_X;
    logic [255:0] dbl_Y;
    logic [255:0] dbl_Z;


    // ------------------------------------------------------------
    // U1 = X1 * Z2^2
    // U2 = X2 * Z1^2
    // ------------------------------------------------------------
    field_mul u_z2_sq (
        .a   (i_Z2),
        .b   (i_Z2),
        .out (Z2_sq)
    );

    field_mul u_u1 (
        .a   (i_X1),
        .b   (Z2_sq),
        .out (U1)
    );

    field_mul u_z1_sq (
        .a   (i_Z1),
        .b   (i_Z1),
        .out (Z1_sq)
    );

    field_mul u_u2 (
        .a   (i_X2),
        .b   (Z1_sq),
        .out (U2)
    );


    // ------------------------------------------------------------
    // S1 = Y1 * Z2^3
    // S2 = Y2 * Z1^3
    // ------------------------------------------------------------
    field_mul u_z2_cu (
        .a   (Z2_sq),
        .b   (i_Z2),
        .out (Z2_cu)
    );

    field_mul u_s1 (
        .a   (i_Y1),
        .b   (Z2_cu),
        .out (S1)
    );

    field_mul u_z1_cu (
        .a   (Z1_sq),
        .b   (i_Z1),
        .out (Z1_cu)
    );

    field_mul u_s2 (
        .a   (i_Y2),
        .b   (Z1_cu),
        .out (S2)
    );


    // ------------------------------------------------------------
    // H = U2 - U1
    // Rr = S2 - S1
    // ------------------------------------------------------------
    field_mul u_h_sq (
        .a   (H),
        .b   (H),
        .out (H_sq)
    );

    field_mul u_h_cu (
        .a   (H_sq),
        .b   (H),
        .out (H_cu)
    );

    field_mul u_u1h2 (
        .a   (U1),
        .b   (H_sq),
        .out (U1H2)
    );

    field_mul u_rr_sq (
        .a   (Rr),
        .b   (Rr),
        .out (Rr_sq)
    );

    field_mul u_rr_tmp_y (
        .a   (Rr),
        .b   (tmp_y),
        .out (Rr_tmp_y)
    );

    field_mul u_s1_hcu (
        .a   (S1),
        .b   (H_cu),
        .out (S1_H_cu)
    );


    // ------------------------------------------------------------
    // Z3 = Z1 * Z2 * H
    // ------------------------------------------------------------
    field_mul u_z1z2 (
        .a   (i_Z1),
        .b   (i_Z2),
        .out (Z1Z2)
    );

    field_mul u_z3 (
        .a   (Z1Z2),
        .b   (H),
        .out (Z3_int)
    );


    // ------------------------------------------------------------
    // If P == Q, use doubling.
    // ------------------------------------------------------------
    jacobian_doubleL u_double (
        .i_X1 (i_X1),
        .i_Y1 (i_Y1),
        .i_Z1 (i_Z1),
        .o_X3 (dbl_X),
        .o_Y3 (dbl_Y),
        .o_Z3 (dbl_Z)
    );


    always_comb begin
        H          = f_sub(U2, U1);
        Rr         = f_sub(S2, S1);
        two_U1H2   = mul2(U1H2);
        X3_int     = f_sub(f_sub(Rr_sq, H_cu), two_U1H2);
        tmp_y      = f_sub(U1H2, X3_int);
        Y3_int     = f_sub(Rr_tmp_y, S1_H_cu);

        if (i_Z1 == ZERO) begin
            o_X3 = i_X2;
            o_Y3 = i_Y2;
            o_Z3 = i_Z2;
        end else if (i_Z2 == ZERO) begin
            o_X3 = i_X1;
            o_Y3 = i_Y1;
            o_Z3 = i_Z1;
        end else if (U1 == U2) begin
            if (S1 != S2) begin
                o_X3 = ZERO;
                o_Y3 = ZERO;
                o_Z3 = ZERO;
            end else begin
                o_X3 = dbl_X;
                o_Y3 = dbl_Y;
                o_Z3 = dbl_Z;
            end
        end else begin
            o_X3 = X3_int;
            o_Y3 = Y3_int;
            o_Z3 = Z3_int;
        end
    end

endmodule