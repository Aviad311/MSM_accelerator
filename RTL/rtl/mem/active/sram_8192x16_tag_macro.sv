`timescale 1ns/1ps

// ============================================================================
// sram_8192x16_tag_macro
// ----------------------------------------------------------------------------
// 8192-word x 16-bit synchronous single-port SRAM wrapper for generation tags.
//
// Physical implementation:
//   1 x ARM s8192x16 SRAM macro.
//
// Interface:
//   en = 0          : SRAM idle
//   en = 1, we = 0  : synchronous read
//   en = 1, we = 1  : synchronous write
//
// ARM macro controls CEN and WEN are active-low.
// ============================================================================

module sram_8192x16_tag_macro (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        en,
    input  logic        we,
    input  logic [12:0] addr,
    input  logic [15:0] wdata,

    output logic [15:0] rdata,
    output logic        rvalid
);

    logic cen_n;
    logic wen_n;

    assign cen_n = ~en;
    assign wen_n = ~we;

    // Track a read request sampled by the synchronous SRAM.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            rvalid <= 1'b0;
        else
            rvalid <= en && !we;
    end

    s8192x16 u_tag_sram (
        .Q     (rdata),
        .CLK   (clk),
        .CEN   (cen_n),
        .WEN   (wen_n),
        .A     (addr),
        .D     (wdata),

        // Use the same functional margin settings as the coordinate SRAM.
        .EMA   (3'b010),
        .EMAW  (2'b00),
        .EMAS  (1'b0),

        // Retention remains enabled during normal operation.
        .RET1N (1'b1)
    );

endmodule