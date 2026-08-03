`timescale 1ns/1ps

// ============================================================================
// bucket_mem_3coord_sram_macro_v2
// ----------------------------------------------------------------------------
// Fully macro-backed bucket memory.
//
// Per bank:
//   X   : 8192 x 256, implemented by 4 x s8192x64
//   Y   : 8192 x 256, implemented by 4 x s8192x64
//   Z   : 8192 x 256, implemented by 4 x s8192x64
//   tag : 8192 x 16,  implemented by 1 x s8192x16
//
// Therefore each bank uses:
//   12 coordinate macros + 1 tag macro = 13 SRAM macros.
//
// Across eight banks:
//   8 x 13 = 104 SRAM macros.
// ============================================================================

module bucket_mem_3coord_sram_macro_v2 #(
    parameter int ADDR_W          = 13,
    parameter int DATA_W          = 256,
    parameter int DEPTH           = 8192,
    parameter int SRAM_RD_LATENCY = 1,
    parameter int GEN_W           = 16
)(
    input  logic                clk,
    input  logic                rst_n,

    input  logic                valid,
    input  logic                write_en,
    input  logic [ADDR_W-1:0]   addr,

    input  logic [DATA_W-1:0]   wdata_x,
    input  logic [DATA_W-1:0]   wdata_y,
    input  logic [DATA_W-1:0]   wdata_z,

    input  logic                tag_write_en,
    input  logic [GEN_W-1:0]    tag_wdata,

    output logic                ready,
    output logic                rvalid,

    output logic [DATA_W-1:0]   rdata_x,
    output logic [DATA_W-1:0]   rdata_y,
    output logic [DATA_W-1:0]   rdata_z,
    output logic [GEN_W-1:0]    tag_rdata
);

    logic rvalid_x;
    logic rvalid_y;
    logic rvalid_z;
    logic rvalid_tag;

    initial begin
        if (ADDR_W != 13)
            $fatal(1,
                "bucket_mem_3coord_sram_macro_v2 requires ADDR_W=13, got %0d",
                ADDR_W);

        if (DATA_W != 256)
            $fatal(1,
                "bucket_mem_3coord_sram_macro_v2 requires DATA_W=256, got %0d",
                DATA_W);

        if (DEPTH != 8192)
            $fatal(1,
                "bucket_mem_3coord_sram_macro_v2 requires DEPTH=8192, got %0d",
                DEPTH);

        if (SRAM_RD_LATENCY != 1)
            $fatal(1,
                "bucket_mem_3coord_sram_macro_v2 requires SRAM_RD_LATENCY=1, got %0d",
                SRAM_RD_LATENCY);

        if (GEN_W != 16)
            $fatal(1,
                "bucket_mem_3coord_sram_macro_v2 requires GEN_W=16, got %0d",
                GEN_W);
    end

    // All physical SRAM macros are fixed-latency and have no backpressure.
    assign ready = 1'b1;

    // A bucket read is complete only when all four memories report the same
    // accepted synchronous read request.
    assign rvalid = rvalid_x & rvalid_y & rvalid_z & rvalid_tag;

    sram_8192x256_macro u_bucket_x_mem (
        .clk    (clk),
        .rst_n  (rst_n),
        .en     (valid),
        .we     (write_en),
        .addr   (addr),
        .wdata  (wdata_x),
        .rdata  (rdata_x),
        .rvalid (rvalid_x)
    );

    sram_8192x256_macro u_bucket_y_mem (
        .clk    (clk),
        .rst_n  (rst_n),
        .en     (valid),
        .we     (write_en),
        .addr   (addr),
        .wdata  (wdata_y),
        .rdata  (rdata_y),
        .rvalid (rvalid_y)
    );

    sram_8192x256_macro u_bucket_z_mem (
        .clk    (clk),
        .rst_n  (rst_n),
        .en     (valid),
        .we     (write_en),
        .addr   (addr),
        .wdata  (wdata_z),
        .rdata  (rdata_z),
        .rvalid (rvalid_z)
    );

    sram_8192x16_tag_macro u_bucket_tag_mem (
        .clk    (clk),
        .rst_n  (rst_n),
        .en     (valid),
        .we     (tag_write_en),
        .addr   (addr),
        .wdata  (tag_wdata),
        .rdata  (tag_rdata),
        .rvalid (rvalid_tag)
    );

endmodule