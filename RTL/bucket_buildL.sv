module bucket_build_4L (
    input  logic [255:0] i_P0_X,
    input  logic [255:0] i_P0_Y,
    input  logic [1:0]   i_P0_bid,

    input  logic [255:0] i_P1_X,
    input  logic [255:0] i_P1_Y,
    input  logic [1:0]   i_P1_bid,

    input  logic [255:0] i_P2_X,
    input  logic [255:0] i_P2_Y,
    input  logic [1:0]   i_P2_bid,

    input  logic [255:0] i_P3_X,
    input  logic [255:0] i_P3_Y,
    input  logic [1:0]   i_P3_bid,

    output logic [255:0] o_B1_X,
    output logic [255:0] o_B1_Y,
    output logic [255:0] o_B1_Z,

    output logic [255:0] o_B2_X,
    output logic [255:0] o_B2_Y,
    output logic [255:0] o_B2_Z,

    output logic [255:0] o_B3_X,
    output logic [255:0] o_B3_Y,
    output logic [255:0] o_B3_Z
);

    localparam logic [255:0] ZERO  = 256'h0;
    localparam logic [255:0] ONE_M = 256'h00000000000000000000000000000000000000000000000000000001000003D1;

    // ------------------------------------------------------------
    // Stage 0 buckets: all buckets start as INF = (0,0,0)
    // ------------------------------------------------------------
    logic [255:0] b1_s0_X, b1_s0_Y, b1_s0_Z;
    logic [255:0] b2_s0_X, b2_s0_Y, b2_s0_Z;
    logic [255:0] b3_s0_X, b3_s0_Y, b3_s0_Z;

    assign b1_s0_X = ZERO;
    assign b1_s0_Y = ZERO;
    assign b1_s0_Z = ZERO;

    assign b2_s0_X = ZERO;
    assign b2_s0_Y = ZERO;
    assign b2_s0_Z = ZERO;

    assign b3_s0_X = ZERO;
    assign b3_s0_Y = ZERO;
    assign b3_s0_Z = ZERO;


    // ------------------------------------------------------------
    // Stage 1: insert P0
    // ------------------------------------------------------------
    logic [255:0] b1_s1_X, b1_s1_Y, b1_s1_Z;
    logic [255:0] b2_s1_X, b2_s1_Y, b2_s1_Z;
    logic [255:0] b3_s1_X, b3_s1_Y, b3_s1_Z;

    logic [255:0] b1_p0_X, b1_p0_Y, b1_p0_Z;
    logic [255:0] b2_p0_X, b2_p0_Y, b2_p0_Z;
    logic [255:0] b3_p0_X, b3_p0_Y, b3_p0_Z;

    jacobian_mixed_addL u_b1_add_p0 (
        .i_X1(b1_s0_X), .i_Y1(b1_s0_Y), .i_Z1(b1_s0_Z),
        .i_X2(i_P0_X),  .i_Y2(i_P0_Y),
        .o_X3(b1_p0_X), .o_Y3(b1_p0_Y), .o_Z3(b1_p0_Z)
    );

    jacobian_mixed_addL u_b2_add_p0 (
        .i_X1(b2_s0_X), .i_Y1(b2_s0_Y), .i_Z1(b2_s0_Z),
        .i_X2(i_P0_X),  .i_Y2(i_P0_Y),
        .o_X3(b2_p0_X), .o_Y3(b2_p0_Y), .o_Z3(b2_p0_Z)
    );

    jacobian_mixed_addL u_b3_add_p0 (
        .i_X1(b3_s0_X), .i_Y1(b3_s0_Y), .i_Z1(b3_s0_Z),
        .i_X2(i_P0_X),  .i_Y2(i_P0_Y),
        .o_X3(b3_p0_X), .o_Y3(b3_p0_Y), .o_Z3(b3_p0_Z)
    );

    always_comb begin
        b1_s1_X = b1_s0_X; b1_s1_Y = b1_s0_Y; b1_s1_Z = b1_s0_Z;
        b2_s1_X = b2_s0_X; b2_s1_Y = b2_s0_Y; b2_s1_Z = b2_s0_Z;
        b3_s1_X = b3_s0_X; b3_s1_Y = b3_s0_Y; b3_s1_Z = b3_s0_Z;

        case (i_P0_bid)
            2'd1: begin b1_s1_X = b1_p0_X; b1_s1_Y = b1_p0_Y; b1_s1_Z = b1_p0_Z; end
            2'd2: begin b2_s1_X = b2_p0_X; b2_s1_Y = b2_p0_Y; b2_s1_Z = b2_p0_Z; end
            2'd3: begin b3_s1_X = b3_p0_X; b3_s1_Y = b3_p0_Y; b3_s1_Z = b3_p0_Z; end
            default: begin end
        endcase
    end


    // ------------------------------------------------------------
    // Stage 2: insert P1
    // ------------------------------------------------------------
    logic [255:0] b1_s2_X, b1_s2_Y, b1_s2_Z;
    logic [255:0] b2_s2_X, b2_s2_Y, b2_s2_Z;
    logic [255:0] b3_s2_X, b3_s2_Y, b3_s2_Z;

    logic [255:0] b1_p1_X, b1_p1_Y, b1_p1_Z;
    logic [255:0] b2_p1_X, b2_p1_Y, b2_p1_Z;
    logic [255:0] b3_p1_X, b3_p1_Y, b3_p1_Z;

    jacobian_mixed_addL u_b1_add_p1 (
        .i_X1(b1_s1_X), .i_Y1(b1_s1_Y), .i_Z1(b1_s1_Z),
        .i_X2(i_P1_X),  .i_Y2(i_P1_Y),
        .o_X3(b1_p1_X), .o_Y3(b1_p1_Y), .o_Z3(b1_p1_Z)
    );

    jacobian_mixed_addL u_b2_add_p1 (
        .i_X1(b2_s1_X), .i_Y1(b2_s1_Y), .i_Z1(b2_s1_Z),
        .i_X2(i_P1_X),  .i_Y2(i_P1_Y),
        .o_X3(b2_p1_X), .o_Y3(b2_p1_Y), .o_Z3(b2_p1_Z)
    );

    jacobian_mixed_addL u_b3_add_p1 (
        .i_X1(b3_s1_X), .i_Y1(b3_s1_Y), .i_Z1(b3_s1_Z),
        .i_X2(i_P1_X),  .i_Y2(i_P1_Y),
        .o_X3(b3_p1_X), .o_Y3(b3_p1_Y), .o_Z3(b3_p1_Z)
    );

    always_comb begin
        b1_s2_X = b1_s1_X; b1_s2_Y = b1_s1_Y; b1_s2_Z = b1_s1_Z;
        b2_s2_X = b2_s1_X; b2_s2_Y = b2_s1_Y; b2_s2_Z = b2_s1_Z;
        b3_s2_X = b3_s1_X; b3_s2_Y = b3_s1_Y; b3_s2_Z = b3_s1_Z;

        case (i_P1_bid)
            2'd1: begin b1_s2_X = b1_p1_X; b1_s2_Y = b1_p1_Y; b1_s2_Z = b1_p1_Z; end
            2'd2: begin b2_s2_X = b2_p1_X; b2_s2_Y = b2_p1_Y; b2_s2_Z = b2_p1_Z; end
            2'd3: begin b3_s2_X = b3_p1_X; b3_s2_Y = b3_p1_Y; b3_s2_Z = b3_p1_Z; end
            default: begin end
        endcase
    end


    // ------------------------------------------------------------
    // Stage 3: insert P2
    // ------------------------------------------------------------
    logic [255:0] b1_s3_X, b1_s3_Y, b1_s3_Z;
    logic [255:0] b2_s3_X, b2_s3_Y, b2_s3_Z;
    logic [255:0] b3_s3_X, b3_s3_Y, b3_s3_Z;

    logic [255:0] b1_p2_X, b1_p2_Y, b1_p2_Z;
    logic [255:0] b2_p2_X, b2_p2_Y, b2_p2_Z;
    logic [255:0] b3_p2_X, b3_p2_Y, b3_p2_Z;

    jacobian_mixed_addL u_b1_add_p2 (
        .i_X1(b1_s2_X), .i_Y1(b1_s2_Y), .i_Z1(b1_s2_Z),
        .i_X2(i_P2_X),  .i_Y2(i_P2_Y),
        .o_X3(b1_p2_X), .o_Y3(b1_p2_Y), .o_Z3(b1_p2_Z)
    );

    jacobian_mixed_addL u_b2_add_p2 (
        .i_X1(b2_s2_X), .i_Y1(b2_s2_Y), .i_Z1(b2_s2_Z),
        .i_X2(i_P2_X),  .i_Y2(i_P2_Y),
        .o_X3(b2_p2_X), .o_Y3(b2_p2_Y), .o_Z3(b2_p2_Z)
    );

    jacobian_mixed_addL u_b3_add_p2 (
        .i_X1(b3_s2_X), .i_Y1(b3_s2_Y), .i_Z1(b3_s2_Z),
        .i_X2(i_P2_X),  .i_Y2(i_P2_Y),
        .o_X3(b3_p2_X), .o_Y3(b3_p2_Y), .o_Z3(b3_p2_Z)
    );

    always_comb begin
        b1_s3_X = b1_s2_X; b1_s3_Y = b1_s2_Y; b1_s3_Z = b1_s2_Z;
        b2_s3_X = b2_s2_X; b2_s3_Y = b2_s2_Y; b2_s3_Z = b2_s2_Z;
        b3_s3_X = b3_s2_X; b3_s3_Y = b3_s2_Y; b3_s3_Z = b3_s2_Z;

        case (i_P2_bid)
            2'd1: begin b1_s3_X = b1_p2_X; b1_s3_Y = b1_p2_Y; b1_s3_Z = b1_p2_Z; end
            2'd2: begin b2_s3_X = b2_p2_X; b2_s3_Y = b2_p2_Y; b2_s3_Z = b2_p2_Z; end
            2'd3: begin b3_s3_X = b3_p2_X; b3_s3_Y = b3_p2_Y; b3_s3_Z = b3_p2_Z; end
            default: begin end
        endcase
    end


    // ------------------------------------------------------------
    // Stage 4: insert P3
    // ------------------------------------------------------------
    logic [255:0] b1_p3_X, b1_p3_Y, b1_p3_Z;
    logic [255:0] b2_p3_X, b2_p3_Y, b2_p3_Z;
    logic [255:0] b3_p3_X, b3_p3_Y, b3_p3_Z;

    jacobian_mixed_addL u_b1_add_p3 (
        .i_X1(b1_s3_X), .i_Y1(b1_s3_Y), .i_Z1(b1_s3_Z),
        .i_X2(i_P3_X),  .i_Y2(i_P3_Y),
        .o_X3(b1_p3_X), .o_Y3(b1_p3_Y), .o_Z3(b1_p3_Z)
    );

    jacobian_mixed_addL u_b2_add_p3 (
        .i_X1(b2_s3_X), .i_Y1(b2_s3_Y), .i_Z1(b2_s3_Z),
        .i_X2(i_P3_X),  .i_Y2(i_P3_Y),
        .o_X3(b2_p3_X), .o_Y3(b2_p3_Y), .o_Z3(b2_p3_Z)
    );

    jacobian_mixed_addL u_b3_add_p3 (
        .i_X1(b3_s3_X), .i_Y1(b3_s3_Y), .i_Z1(b3_s3_Z),
        .i_X2(i_P3_X),  .i_Y2(i_P3_Y),
        .o_X3(b3_p3_X), .o_Y3(b3_p3_Y), .o_Z3(b3_p3_Z)
    );

    always_comb begin
        o_B1_X = b1_s3_X; o_B1_Y = b1_s3_Y; o_B1_Z = b1_s3_Z;
        o_B2_X = b2_s3_X; o_B2_Y = b2_s3_Y; o_B2_Z = b2_s3_Z;
        o_B3_X = b3_s3_X; o_B3_Y = b3_s3_Y; o_B3_Z = b3_s3_Z;

        case (i_P3_bid)
            2'd1: begin o_B1_X = b1_p3_X; o_B1_Y = b1_p3_Y; o_B1_Z = b1_p3_Z; end
            2'd2: begin o_B2_X = b2_p3_X; o_B2_Y = b2_p3_Y; o_B2_Z = b2_p3_Z; end
            2'd3: begin o_B3_X = b3_p3_X; o_B3_Y = b3_p3_Y; o_B3_Z = b3_p3_Z; end
            default: begin end
        endcase
    end

endmodule