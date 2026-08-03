`timescale 1ns/1ps

module bucket_mem_3coord #(
    parameter int ADDR_W          = 4,
    parameter int DATA_W          = 256,
    parameter int DEPTH           = (1 << ADDR_W),
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

    logic ready_x,  ready_y,  ready_z,  ready_tag;
    logic rvalid_x, rvalid_y, rvalid_z, rvalid_tag;

    assign ready  = ready_x  & ready_y  & ready_z  & ready_tag;
    assign rvalid = rvalid_x & rvalid_y & rvalid_z & rvalid_tag;

    simple_sync_sram_1rw #(
        .ADDR_W     (ADDR_W),
        .DATA_W     (DATA_W),
        .DEPTH      (DEPTH),
        .RD_LATENCY (SRAM_RD_LATENCY)
    ) u_bucket_x_mem (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid    (valid),
        .write_en (write_en),
        .addr     (addr),
        .wdata    (wdata_x),
        .ready    (ready_x),
        .rvalid   (rvalid_x),
        .rdata    (rdata_x)
    );

    simple_sync_sram_1rw #(
        .ADDR_W     (ADDR_W),
        .DATA_W     (DATA_W),
        .DEPTH      (DEPTH),
        .RD_LATENCY (SRAM_RD_LATENCY)
    ) u_bucket_y_mem (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid    (valid),
        .write_en (write_en),
        .addr     (addr),
        .wdata    (wdata_y),
        .ready    (ready_y),
        .rvalid   (rvalid_y),
        .rdata    (rdata_y)
    );

    simple_sync_sram_1rw #(
        .ADDR_W     (ADDR_W),
        .DATA_W     (DATA_W),
        .DEPTH      (DEPTH),
        .RD_LATENCY (SRAM_RD_LATENCY)
    ) u_bucket_z_mem (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid    (valid),
        .write_en (write_en),
        .addr     (addr),
        .wdata    (wdata_z),
        .ready    (ready_z),
        .rvalid   (rvalid_z),
        .rdata    (rdata_z)
    );

    simple_sync_sram_1rw #(
        .ADDR_W     (ADDR_W),
        .DATA_W     (GEN_W),
        .DEPTH      (DEPTH),
        .RD_LATENCY (SRAM_RD_LATENCY)
    ) u_bucket_tag_mem (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid    (valid),
        .write_en (tag_write_en),
        .addr     (addr),
        .wdata    (tag_wdata),
        .ready    (ready_tag),
        .rvalid   (rvalid_tag),
        .rdata    (tag_rdata)
    );

endmodule