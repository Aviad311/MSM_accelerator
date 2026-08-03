
`timescale 1ns/1ps

// ============================================================================
// sram_8192x256_macro
// ----------------------------------------------------------------------------
// 8192-word x 256-bit synchronous single-port SRAM.
//
// Physical implementation:
//   4 x ARM s8192x64 SRAM macros operating in parallel.
//
// Slice mapping:
//   u_sram_0 : bits [ 63:  0]
//   u_sram_1 : bits [127: 64]
//   u_sram_2 : bits [191:128]
//   u_sram_3 : bits [255:192]
//
// Interface:
//   en = 0                  : SRAM idle
//   en = 1, we = 0         : synchronous read
//   en = 1, we = 1         : synchronous write
//
// A read request is sampled on the rising clock edge.
// rdata is produced by the SRAM macro after that edge.
// rvalid indicates that rdata corresponds to a read request sampled on the
// current rising edge.
//
// Notes:
//   - The SRAM contents are not reset by rst_n.
//   - Reading an address before writing it may return X in simulation.
//   - The generated s8192x64.v file must also be included in compilation.
// ============================================================================

module sram_8192x256_macro (
    input  logic         clk,
    input  logic         rst_n,

    input  logic         en,
    input  logic         we,
    input  logic [12:0]  addr,
    input  logic [255:0] wdata,

    output logic [255:0] rdata,
    output logic         rvalid
);

    // ------------------------------------------------------------------------
    // ARM SRAM control pins are active-low.
    // ------------------------------------------------------------------------
    logic cen_n;
    logic wen_n;

    assign cen_n = ~en;
    assign wen_n = ~we;

    // ------------------------------------------------------------------------
    // Individual 64-bit SRAM outputs.
    // ------------------------------------------------------------------------
    logic [63:0] q0;
    logic [63:0] q1;
    logic [63:0] q2;
    logic [63:0] q3;

    assign rdata = {
        q3,
        q2,
        q1,
        q0
    };

    // ------------------------------------------------------------------------
    // Read-valid tracking.
    //
    // The macro samples a read request at the rising edge and updates Q after
    // its CLK-to-Q delay. This register marks that same sampled read request.
    // ------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rvalid <= 1'b0;
        end else begin
            rvalid <= en && !we;
        end
    end

    // ------------------------------------------------------------------------
    // SRAM slice 0: wdata/rdata [63:0]
    // ------------------------------------------------------------------------
    s8192x64 u_sram_0 (
        .CENY       (),
        .WENY       (),
        .AY         (),
        .Q          (q0),
        .SO         (),

        .CLK        (clk),
        .CEN        (cen_n),
        .WEN        (wen_n),
        .A          (addr),
        .D          (wdata[63:0]),

        .EMA        (3'b010),
        .EMAW       (2'b00),

        .TEN        (1'b1),
        .TCEN       (1'b1),
        .TWEN       (1'b1),
        .TA         (13'b0),
        .TD         (64'b0),

        .RET1N      (1'b1),
        .SI         (2'b0),
        .SE         (1'b0),
        .DFTRAMBYP  (1'b0)
    );

    // ------------------------------------------------------------------------
    // SRAM slice 1: wdata/rdata [127:64]
    // ------------------------------------------------------------------------
    s8192x64 u_sram_1 (
        .CENY       (),
        .WENY       (),
        .AY         (),
        .Q          (q1),
        .SO         (),

        .CLK        (clk),
        .CEN        (cen_n),
        .WEN        (wen_n),
        .A          (addr),
        .D          (wdata[127:64]),

        .EMA        (3'b010),
        .EMAW       (2'b00),

        .TEN        (1'b1),
        .TCEN       (1'b1),
        .TWEN       (1'b1),
        .TA         (13'b0),
        .TD         (64'b0),

        .RET1N      (1'b1),
        .SI         (2'b0),
        .SE         (1'b0),
        .DFTRAMBYP  (1'b0)
    );

    // ------------------------------------------------------------------------
    // SRAM slice 2: wdata/rdata [191:128]
    // ------------------------------------------------------------------------
    s8192x64 u_sram_2 (
        .CENY       (),
        .WENY       (),
        .AY         (),
        .Q          (q2),
        .SO         (),

        .CLK        (clk),
        .CEN        (cen_n),
        .WEN        (wen_n),
        .A          (addr),
        .D          (wdata[191:128]),

        .EMA        (3'b010),
        .EMAW       (2'b00),

        .TEN        (1'b1),
        .TCEN       (1'b1),
        .TWEN       (1'b1),
        .TA         (13'b0),
        .TD         (64'b0),

        .RET1N      (1'b1),
        .SI         (2'b0),
        .SE         (1'b0),
        .DFTRAMBYP  (1'b0)
    );

    // ------------------------------------------------------------------------
    // SRAM slice 3: wdata/rdata [255:192]
    // ------------------------------------------------------------------------
    s8192x64 u_sram_3 (
        .CENY       (),
        .WENY       (),
        .AY         (),
        .Q          (q3),
        .SO         (),

        .CLK        (clk),
        .CEN        (cen_n),
        .WEN        (wen_n),
        .A          (addr),
        .D          (wdata[255:192]),

        .EMA        (3'b010),
        .EMAW       (2'b00),

        .TEN        (1'b1),
        .TCEN       (1'b1),
        .TWEN       (1'b1),
        .TA         (13'b0),
        .TD         (64'b0),

        .RET1N      (1'b1),
        .SI         (2'b0),
        .SE         (1'b0),
        .DFTRAMBYP  (1'b0)
    );

endmodule

