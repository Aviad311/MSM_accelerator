`timescale 1ns/1ps

// ============================================================================
// File: rtl/frontend/async_fifo_gray_v1.sv
//
// Generic dual-clock FIFO using Gray-coded pointers.
//
// Intended use in this project:
//   AXI clock domain writes complete 1024-bit point frames.
//   MSM clock domain reads one complete frame per cycle.
//
// Notes:
// - DEPTH must be a power of two.
// - Memory is modeled as a dual-port array. For final ASIC implementation,
//   replace it with an appropriate dual-port SRAM macro if required.
// ============================================================================

module async_fifo_gray_v1 #(
    parameter int DATA_W = 1024,
    parameter int DEPTH  = 64
)(
    input  logic              wr_clk,
    input  logic              wr_rst_n,
    input  logic              wr_en,
    input  logic [DATA_W-1:0] wr_data,
    output logic              wr_full,

    input  logic              rd_clk,
    input  logic              rd_rst_n,
    input  logic              rd_en,
    output logic [DATA_W-1:0] rd_data,
    output logic              rd_empty
);

    localparam int ADDR_W = $clog2(DEPTH);
    localparam int PTR_W  = ADDR_W + 1;

    logic [DATA_W-1:0] mem [0:DEPTH-1];

    logic [PTR_W-1:0] wr_bin_q, wr_bin_n;
    logic [PTR_W-1:0] wr_gray_q, wr_gray_n;
    logic [PTR_W-1:0] rd_bin_q, rd_bin_n;
    logic [PTR_W-1:0] rd_gray_q, rd_gray_n;

    logic [PTR_W-1:0] rd_gray_sync1_q, rd_gray_sync2_q;
    logic [PTR_W-1:0] wr_gray_sync1_q, wr_gray_sync2_q;

    logic wr_push;
    logic rd_pop;

    assign wr_push = wr_en && !wr_full;
    assign rd_pop  = rd_en && !rd_empty;

    assign wr_bin_n  = wr_bin_q + wr_push;
    assign wr_gray_n = (wr_bin_n >> 1) ^ wr_bin_n;

    assign rd_bin_n  = rd_bin_q + rd_pop;
    assign rd_gray_n = (rd_bin_n >> 1) ^ rd_bin_n;

    // Full when the next write pointer equals the synchronized read pointer
    // with the two MSBs inverted.
    assign wr_full =
        (wr_gray_n ==
         {~rd_gray_sync2_q[PTR_W-1:PTR_W-2],
           rd_gray_sync2_q[PTR_W-3:0]});

    assign rd_empty = (rd_gray_q == wr_gray_sync2_q);

    always_ff @(posedge wr_clk or negedge wr_rst_n) begin
        if (!wr_rst_n) begin
            wr_bin_q  <= '0;
            wr_gray_q <= '0;
        end else begin
            wr_bin_q  <= wr_bin_n;
            wr_gray_q <= wr_gray_n;

            if (wr_push)
                mem[wr_bin_q[ADDR_W-1:0]] <= wr_data;
        end
    end

    always_ff @(posedge rd_clk or negedge rd_rst_n) begin
        if (!rd_rst_n) begin
            rd_bin_q  <= '0;
            rd_gray_q <= '0;
        end else begin
            rd_bin_q  <= rd_bin_n;
            rd_gray_q <= rd_gray_n;
        end
    end

    // FWFT-style asynchronous read view for simulation/reference RTL.
    // Replace with a synchronous-read macro plus skid/output register if needed.
    assign rd_data = mem[rd_bin_q[ADDR_W-1:0]];

    // Synchronize read pointer into write clock domain.
    always_ff @(posedge wr_clk or negedge wr_rst_n) begin
        if (!wr_rst_n) begin
            rd_gray_sync1_q <= '0;
            rd_gray_sync2_q <= '0;
        end else begin
            rd_gray_sync1_q <= rd_gray_q;
            rd_gray_sync2_q <= rd_gray_sync1_q;
        end
    end

    // Synchronize write pointer into read clock domain.
    always_ff @(posedge rd_clk or negedge rd_rst_n) begin
        if (!rd_rst_n) begin
            wr_gray_sync1_q <= '0;
            wr_gray_sync2_q <= '0;
        end else begin
            wr_gray_sync1_q <= wr_gray_q;
            wr_gray_sync2_q <= wr_gray_sync1_q;
        end
    end

    initial begin
        if (DEPTH < 4)
            $fatal(1, "async_fifo_gray_v1 DEPTH must be >= 4");

        if ((DEPTH & (DEPTH-1)) != 0)
            $fatal(1, "async_fifo_gray_v1 DEPTH must be a power of two");
    end

endmodule