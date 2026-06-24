// =================================================================
// File: RTL/montgomery/secp256k1_montgomery_mul.sv
// =================================================================
// 16-cycle Montgomery multiplier adapted for secp256k1 field.
//
// Original structure came from bn254_montgomery_mul,
// but constants were changed from BN254 to secp256k1.
//
// Computes:
//   result = op_a * op_b * R^-1 mod p
//
// Where for secp256k1:
//   p = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
//
// Interface:
//   in_valid  : pulse high for one cycle with valid op_a/op_b
//   out_valid : high when result is valid
//   ready     : currently always 1
//
// NOTE:
//   Even though ready is always 1, for our first FSM bring-up we will use
//   this conservatively: issue one multiplication, wait for out_valid,
//   then issue the next multiplication.
// =================================================================

`timescale 1ns/1ps

module secp256k1_montgomery_mul #(
    parameter int WIDTH = 256
) (
    input  logic                clk,
    input  logic                rst_n,
    input  logic                in_valid,
    input  logic [WIDTH-1:0]    op_a,
    input  logic [WIDTH-1:0]    op_b,
    output logic                out_valid,
    output logic [WIDTH-1:0]    result,
    output logic                ready
);

    // -------------------------------------------------------------
    // secp256k1 field constants
    // -------------------------------------------------------------
    //
    // p = 2^256 - 2^32 - 977
    //
    localparam [255:0] MODULUS_HC =
        256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F;

    // MU_HC is the low 64 bits of:
    //
    // NPRIME = -p^-1 mod 2^256
    //        = C9BD1905155383999C46C2C295F2B761BCB223FEDC24A059D838091DD2253531
    //
    // Low 64 bits:
    //
    localparam [63:0] MU_HC =
        64'hD838091DD2253531;

    assign ready = 1'b1;

    // -------------------------------------------------------------
    // 1. FREE-RUNNING VALID PIPELINE
    // -------------------------------------------------------------
    logic [16:1] v_reg;
    logic [16:0] v;

    assign v = {v_reg, in_valid};

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_reg <= '0;
        end else begin
            v_reg <= {v_reg[15:1], in_valid};
        end
    end

    // -------------------------------------------------------------
    // 2. TAPERED SHIFT REGISTERS
    // -------------------------------------------------------------
    logic [255:0] a_s0 [1:3];
    logic [191:0] a_s1 [0:3];
    logic [127:0] a_s2 [0:3];
    logic [63:0]  a_s3 [0:3];

    logic [255:0] b   [1:16];
    logic [321:0] acc [1:16];

    // -------------------------------------------------------------
    // 3. RETIMED CIOS DATAPATH
    // -------------------------------------------------------------
    genvar i;
    generate
        for (i = 0; i < 4; i++) begin : CIOS_STAGES
            localparam int IDX = i * 4;

            // Current 64-bit word from operand A.
            logic [63:0] current_a_chunk;

            always_comb begin
                if (i == 0) begin
                    current_a_chunk = op_a[63:0];
                end else if (i == 1) begin
                    current_a_chunk = a_s1[0][63:0];
                end else if (i == 2) begin
                    current_a_chunk = a_s2[0][63:0];
                end else begin
                    current_a_chunk = a_s3[0][63:0];
                end
            end

            // Current B and accumulator.
            logic [255:0] current_b;
            logic [321:0] current_acc;

            always_comb begin
                if (IDX == 0) begin
                    current_b   = op_b;
                    current_acc = '0;
                end else begin
                    current_b   = b[IDX];
                    current_acc = acc[IDX];
                end
            end

            // -----------------------------------------------------
            // Cycle 1: partial product
            // -----------------------------------------------------
            logic [319:0] c1_prod;

            always_ff @(posedge clk) begin
                if (v[IDX]) begin
                    b[IDX+1]   <= current_b;
                    acc[IDX+1] <= current_acc;
                    c1_prod    <= 320'(current_a_chunk) * 320'(current_b);

                    if (i == 0) begin
                        a_s0[1] <= op_a;
                    end else if (i == 1) begin
                        a_s1[1] <= a_s1[0];
                    end else if (i == 2) begin
                        a_s2[1] <= a_s2[0];
                    end else begin
                        a_s3[1] <= a_s3[0];
                    end
                end
            end

            // -----------------------------------------------------
            // Cycle 2: accumulation and quotient
            // -----------------------------------------------------
            logic [321:0] c2_acc_new;
            logic [63:0]  c2_q;

            always_comb begin
                c2_acc_new = acc[IDX+1] + c1_prod;
            end

            always_ff @(posedge clk) begin
                if (v[IDX+1]) begin
                    b[IDX+2]   <= b[IDX+1];
                    acc[IDX+2] <= c2_acc_new;
                    c2_q       <= c2_acc_new[63:0] * MU_HC;

                    if (i == 0) begin
                        a_s0[2] <= a_s0[1];
                    end else if (i == 1) begin
                        a_s1[2] <= a_s1[1];
                    end else if (i == 2) begin
                        a_s2[2] <= a_s2[1];
                    end else begin
                        a_s3[2] <= a_s3[1];
                    end
                end
            end

            // -----------------------------------------------------
            // Cycle 3: reduction product
            // -----------------------------------------------------
            logic [319:0] c3_red;

            always_ff @(posedge clk) begin
                if (v[IDX+2]) begin
                    b[IDX+3]   <= b[IDX+2];
                    acc[IDX+3] <= acc[IDX+2];
                    c3_red     <= 320'(c2_q) * 320'(MODULUS_HC);

                    if (i == 0) begin
                        a_s0[3] <= a_s0[2];
                    end else if (i == 1) begin
                        a_s1[3] <= a_s1[2];
                    end else if (i == 2) begin
                        a_s2[3] <= a_s2[2];
                    end else begin
                        a_s3[3] <= a_s3[2];
                    end
                end
            end

            // -----------------------------------------------------
            // Cycle 4: final addition and taper shift
            // -----------------------------------------------------
            logic [321:0] c4_acc;

            always_comb begin
                c4_acc = (acc[IDX+3] + c3_red) >> 64;
            end

            always_ff @(posedge clk) begin
                if (v[IDX+3]) begin
                    b[IDX+4] <= b[IDX+3];

                    // Conditional subtraction only on final 64-bit word.
                    if (i == 3) begin
                        if (c4_acc >= MODULUS_HC) begin
                            acc[IDX+4] <= 322'(c4_acc - MODULUS_HC);
                        end else begin
                            acc[IDX+4] <= 322'(c4_acc);
                        end
                    end else begin
                        acc[IDX+4] <= c4_acc;
                    end

                    // Taper A: drop already-consumed 64-bit word.
                    if (i == 0) begin
                        a_s1[0] <= a_s0[3][255:64];
                    end else if (i == 1) begin
                        a_s2[0] <= a_s1[3][191:64];
                    end else if (i == 2) begin
                        a_s3[0] <= a_s2[3][127:64];
                    end
                end
            end
        end
    endgenerate

    assign out_valid = v[16];
    assign result    = acc[16][255:0];

endmodule