```systemverilog
// =============================================================================
// ASIC-style Reference Model: Field Arithmetic for secp256k1 (GF(p))
// Montgomery-native Datapath
// =============================================================================

`timescale 1ns / 1ps

package field_pkg;
    // secp256k1 prime: p = 2^256 - 2^32 - 977
    localparam [255:0] P = 256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F;

    // Montgomery Parameters
    // NPRIME = -p^{-1} mod 2^256
    localparam [255:0] NPRIME = 256'hD838091DD22535310895C0DC1180B1E428C00684F04D7C07B483C6F8F5539591;

    // R2 = R^2 mod p, where R = 2^256
    // Calculated as: (2^256)^2 mod p = (2^32 + 977)^2 = 2^64 + 1954*2^32 + 954529
    localparam [255:0] R2 = 256'h000000000000000000000000000000000000000000000001000007A2000E90A1;

    // ONE_M = 1 in Montgomery domain = R mod p = 2^32 + 977
    localparam [255:0] ONE_M = 256'h00000000000000000000000000000000000000000000000000000001000003D1;
    localparam [255:0] ZERO  = 256'h0;
endpackage

import field_pkg::*;

// -----------------------------------------------------------------------------
// Addition (Combinational)
// Logical Flow: s = a + b; if s >= p: s -= p
// -----------------------------------------------------------------------------
module field_add (
    input  logic [255:0] a,
    input  logic [255:0] b,
    output logic [255:0] s
);
    logic [256:0] sum;
    assign sum = a + b;

    always_comb begin
        if (sum >= P) begin
            s = sum - P;
        end else begin
            s = sum[255:0];
        end
    end
endmodule

// -----------------------------------------------------------------------------
// Subtraction (Combinational)
// Logical Flow: d = a - b; if d < 0: d += p
// -----------------------------------------------------------------------------
module field_sub (
    input  logic [255:0] a,
    input  logic [255:0] b,
    output logic [255:0] d
);
    logic [256:0] diff;
    assign diff = {1'b0, a} - {1'b0, b};

    always_comb begin
        if (a < b) begin
            d = diff[255:0] + P;
        end else begin
            d = diff[255:0];
        end
    end
endmodule

// -----------------------------------------------------------------------------
// Negation (Combinational)
// Logical Flow: 0 if a == 0 else (p - a)
// -----------------------------------------------------------------------------
module field_neg (
    input  logic [255:0] a,
    output logic [255:0] out
);
    always_comb begin
        if (a == 256'h0) begin
            out = 256'h0;
        end else begin
            out = P - a;
        end
    end
endmodule

// -----------------------------------------------------------------------------
// Montgomery Reduction (REDC) (Combinational Datapath)
// Logical Flow:
//   m = (t * NPRIME) mod R
//   u = (t + m*p) / R
//   if u >= p: u -= p
// -----------------------------------------------------------------------------
module mont_red (
    input  logic [511:0] t,
    output logic [255:0] out
);
    logic [255:0] m;
    logic [511:0] m_p;
    logic [512:0] t_plus_mp;
    logic [256:0] u;

    // m = (t * NPRIME) mod R
    assign m = t[255:0] * NPRIME; 

    // u = (t + m*p) / R
    assign m_p = m * P;
    assign t_plus_mp = t + m_p;
    assign u = t_plus_mp[512:256];

    always_comb begin
        if (u >= P) begin
            out = u - P;
        end else begin
            out = u[255:0];
        end
    end
endmodule

// -----------------------------------------------------------------------------
// Montgomery Multiplication (Combinational Datapath)
// Logical Flow: mont_red(a * b)
// -----------------------------------------------------------------------------
module field_mul (
    input  logic [255:0] a,
    input  logic [255:0] b,
    output logic [255:0] out
);
    logic [511:0] prod;
    assign prod = a * b;

    mont_red red_inst (
        .t(prod),
        .out(out)
    );
endmodule

// -----------------------------------------------------------------------------
// Boundary Conversions (Combinational)
// -----------------------------------------------------------------------------

// to_mont: Convert normal a -> aM = a*R mod p
module to_mont (
    input  logic [255:0] a,
    output logic [255:0] out
);
    logic [255:0] a_canon;
    assign a_canon = (a >= P) ? (a - P) : a;

    field_mul mul_r2 (
        .a(a_canon),
        .b(R2),
        .out(out)
    );
endmodule

// from_mont: Convert Montgomery aM -> normal a
module from_mont (
    input  logic [255:0] aM,
    output logic [255:0] out
);
    mont_red red_inst (
        .t({256'h0, aM}),
        .out(out)
    );
endmodule

// -----------------------------------------------------------------------------
// Field Inversion (Sequential)
// Logical Flow: Square-and-Multiply algorithm for aM^(p-2) mod p
// -----------------------------------------------------------------------------
module field_inv (
    input  logic         clk,
    input  logic         reset_n,
    input  logic         start,
    input  logic [255:0] aM,
    output logic [255:0] out,
    output logic         busy,
    output logic         done
);
    localparam [255:0] EXP = P - 2;

    typedef enum logic [1:0] {IDLE, SQUARE, MULTIPLY, FINISH} state_t;
    state_t state;

    logic [255:0] res;
    logic [255:0] base;
    logic [8:0]   bit_idx;

    // Multiplier Interface
    logic [255:0] mul_a, mul_b, mul_out;
    field_mul inst_mul (.a(mul_a), .b(mul_b), .out(mul_out));

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state   <= IDLE;
            busy    <= 1'b0;
            done    <= 1'b0;
            bit_idx <= 9'd255;
            out     <= 256'h0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        res     <= ONE_M;
                        base    <= aM;
                        bit_idx <= 9'd255;
                        busy    <= 1'b1;
                        state   <= SQUARE;
                    end else begin
                        busy <= 1'b0;
                    end
                end

                SQUARE: begin
                    // mul_a = res, mul_b = res
                    res   <= mul_out;
                    state <= MULTIPLY;
                end

                MULTIPLY: begin
                    if (EXP[bit_idx]) begin
                        // mul_a = res, mul_b = base
                        res <= mul_out;
                    end
                    
                    if (bit_idx == 0) begin
                        state <= FINISH;
                    end else begin
                        bit_idx <= bit_idx - 1;
                        state   <= SQUARE;
                    end
                end

                FINISH: begin
                    out   <= res;
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    state <= IDLE;
                end
            endcase
        end
    end

    // Combinational Mux for the shared multiplier
    always_comb begin
        if (state == SQUARE) begin
            mul_a = res;
            mul_b = res;
        end else begin
            mul_a = res;
            mul_b = base;
        end
    end

endmodule
```
