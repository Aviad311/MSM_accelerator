module pippenger_window_4L (
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

    output logic [255:0] o_X,
    output logic [255:0] o_Y,
    output logic [255:0] o_Z
);

    logic [255:0] B1_X;
    logic [255:0] B1_Y;
    logic [255:0] B1_Z;

    logic [255:0] B2_X;
    logic [255:0] B2_Y;
    logic [255:0] B2_Z;

    logic [255:0] B3_X;
    logic [255:0] B3_Y;
    logic [255:0] B3_Z;

    bucket_build_4L u_bucket_build (
        .i_P0_X   (i_P0_X),
        .i_P0_Y   (i_P0_Y),
        .i_P0_bid (i_P0_bid),

        .i_P1_X   (i_P1_X),
        .i_P1_Y   (i_P1_Y),
        .i_P1_bid (i_P1_bid),

        .i_P2_X   (i_P2_X),
        .i_P2_Y   (i_P2_Y),
        .i_P2_bid (i_P2_bid),

        .i_P3_X   (i_P3_X),
        .i_P3_Y   (i_P3_Y),
        .i_P3_bid (i_P3_bid),

        .o_B1_X   (B1_X),
        .o_B1_Y   (B1_Y),
        .o_B1_Z   (B1_Z),

        .o_B2_X   (B2_X),
        .o_B2_Y   (B2_Y),
        .o_B2_Z   (B2_Z),

        .o_B3_X   (B3_X),
        .o_B3_Y   (B3_Y),
        .o_B3_Z   (B3_Z)
    );

    reduce_buckets_4L u_reduce (
        .i_B1_X (B1_X),
        .i_B1_Y (B1_Y),
        .i_B1_Z (B1_Z),

        .i_B2_X (B2_X),
        .i_B2_Y (B2_Y),
        .i_B2_Z (B2_Z),

        .i_B3_X (B3_X),
        .i_B3_Y (B3_Y),
        .i_B3_Z (B3_Z),

        .o_X    (o_X),
        .o_Y    (o_Y),
        .o_Z    (o_Z)
    );

endmodule