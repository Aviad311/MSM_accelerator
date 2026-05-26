module reduce_buckets_4L (
    input  logic [255:0] i_B1_X,
    input  logic [255:0] i_B1_Y,
    input  logic [255:0] i_B1_Z,

    input  logic [255:0] i_B2_X,
    input  logic [255:0] i_B2_Y,
    input  logic [255:0] i_B2_Z,

    input  logic [255:0] i_B3_X,
    input  logic [255:0] i_B3_Y,
    input  logic [255:0] i_B3_Z,

    output logic [255:0] o_X,
    output logic [255:0] o_Y,
    output logic [255:0] o_Z
);

    // running_sum after processing bucket 3:
    // running_sum_3 = B3
    logic [255:0] rs3_X;
    logic [255:0] rs3_Y;
    logic [255:0] rs3_Z;

    // result after bucket 3:
    // result_3 = B3
    logic [255:0] res3_X;
    logic [255:0] res3_Y;
    logic [255:0] res3_Z;

    // running_sum after bucket 2:
    // running_sum_2 = running_sum_3 + B2
    logic [255:0] rs2_X;
    logic [255:0] rs2_Y;
    logic [255:0] rs2_Z;

    // result after bucket 2:
    // result_2 = result_3 + running_sum_2
    logic [255:0] res2_X;
    logic [255:0] res2_Y;
    logic [255:0] res2_Z;

    // running_sum after bucket 1:
    // running_sum_1 = running_sum_2 + B1
    logic [255:0] rs1_X;
    logic [255:0] rs1_Y;
    logic [255:0] rs1_Z;

    // result after bucket 1:
    // result = result_2 + running_sum_1
    logic [255:0] res1_X;
    logic [255:0] res1_Y;
    logic [255:0] res1_Z;


    // Since initial running_sum = INF and result = INF:
    // after bucket 3, both are just B3.
    assign rs3_X  = i_B3_X;
    assign rs3_Y  = i_B3_Y;
    assign rs3_Z  = i_B3_Z;

    assign res3_X = i_B3_X;
    assign res3_Y = i_B3_Y;
    assign res3_Z = i_B3_Z;


    // running_sum_2 = B3 + B2
    jacobian_addL u_add_rs2 (
        .i_X1 (rs3_X),
        .i_Y1 (rs3_Y),
        .i_Z1 (rs3_Z),

        .i_X2 (i_B2_X),
        .i_Y2 (i_B2_Y),
        .i_Z2 (i_B2_Z),

        .o_X3 (rs2_X),
        .o_Y3 (rs2_Y),
        .o_Z3 (rs2_Z)
    );


    // result_2 = B3 + running_sum_2
    jacobian_addL u_add_res2 (
        .i_X1 (res3_X),
        .i_Y1 (res3_Y),
        .i_Z1 (res3_Z),

        .i_X2 (rs2_X),
        .i_Y2 (rs2_Y),
        .i_Z2 (rs2_Z),

        .o_X3 (res2_X),
        .o_Y3 (res2_Y),
        .o_Z3 (res2_Z)
    );


    // running_sum_1 = running_sum_2 + B1
    jacobian_addL u_add_rs1 (
        .i_X1 (rs2_X),
        .i_Y1 (rs2_Y),
        .i_Z1 (rs2_Z),

        .i_X2 (i_B1_X),
        .i_Y2 (i_B1_Y),
        .i_Z2 (i_B1_Z),

        .o_X3 (rs1_X),
        .o_Y3 (rs1_Y),
        .o_Z3 (rs1_Z)
    );


    // result = result_2 + running_sum_1
    jacobian_addL u_add_res1 (
        .i_X1 (res2_X),
        .i_Y1 (res2_Y),
        .i_Z1 (res2_Z),

        .i_X2 (rs1_X),
        .i_Y2 (rs1_Y),
        .i_Z2 (rs1_Z),

        .o_X3 (res1_X),
        .o_Y3 (res1_Y),
        .o_Z3 (res1_Z)
    );


    assign o_X = res1_X;
    assign o_Y = res1_Y;
    assign o_Z = res1_Z;

endmodule