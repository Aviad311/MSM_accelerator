`timescale 1ns/1ps

// ============================================================================
// File: rtl/frontend/axi_point_stream_source_dualclk_v1.sv
//
// Performance-oriented AXI point source.
//
// AXI side:
//   - 512-bit AXI read data
//   - runs in axi_clk domain
//   - collects two 512-bit beats into one 1024-bit point frame
//
// MSM side:
//   - runs in msm_clk domain
//   - dequeues one complete point frame per MSM cycle when point_ready=1
//
// Memory format:
//   Beat 0 [255:0]   = X Montgomery
//          [511:256] = Y Montgomery
//
//   Beat 1 [GLOBAL_ADDR_W-1:0] = bucket_id
//          [GLOBAL_ADDR_W]     = last_point
//
// This module is designed for:
//   axi_clk = 2 × msm_clk
//
// so the AXI side can receive two 512-bit beats during one MSM cycle.
// ============================================================================

module axi_point_stream_source_dualclk_v1 #(
    parameter int GLOBAL_ADDR_W   = 16,
    parameter int DATA_W          = 256,
    parameter int ASYNC_FIFO_DEPTH = 64,
    parameter int MAX_BURST_BEATS = 16
)(
    input  logic axi_clk,
    input  logic axi_rst_n,

    input  logic msm_clk,
    input  logic msm_rst_n,

    input  logic        start_axi,
    input  logic [63:0] base_addr,
    input  logic [31:0] logical_point_count,

    output logic        axi_busy,
    output logic        axi_done,

    // AXI4 read-address channel
    output logic [63:0] m_axi_araddr,
    output logic [7:0]  m_axi_arlen,
    output logic [2:0]  m_axi_arsize,
    output logic [1:0]  m_axi_arburst,
    output logic        m_axi_arvalid,
    input  logic        m_axi_arready,

    // AXI4 read-data channel
    input  logic [511:0] m_axi_rdata,
    input  logic [1:0]   m_axi_rresp,
    input  logic         m_axi_rlast,
    input  logic         m_axi_rvalid,
    output logic         m_axi_rready,

    // MSM point stream
    output logic                     point_valid,
    input  logic                     point_ready,
    output logic [GLOBAL_ADDR_W-1:0] point_bucket_id,
    output logic [DATA_W-1:0]        point_x,
    output logic [DATA_W-1:0]        point_y,
    output logic                     point_last
);

    localparam int FRAME_W = 1024;

    logic [63:0] cur_araddr_q;
    logic [31:0] beats_left_to_req_q;
    logic [31:0] beats_left_to_rx_q;
    logic [8:0]  request_beats;

    logic        half_frame_q;
    logic [511:0] beat0_q;

    logic [FRAME_W-1:0] fifo_wr_data;
    logic               fifo_wr_en;
    logic               fifo_wr_full;

    logic [FRAME_W-1:0] fifo_rd_data;
    logic               fifo_rd_en;
    logic               fifo_rd_empty;

    logic [31:0] accepted_points_msm_q;

    logic ar_fire;
    logic r_fire;

    assign ar_fire = m_axi_arvalid && m_axi_arready;
    assign r_fire  = m_axi_rvalid && m_axi_rready;

    assign m_axi_arsize  = 3'b110;
    assign m_axi_arburst = 2'b01;

    // Never accept beat 1 of a frame unless the complete-frame FIFO has room.
    // Beat 0 may be accepted and held locally.
    assign m_axi_rready =
        !half_frame_q || !fifo_wr_full;

    assign request_beats =
        (beats_left_to_req_q >= MAX_BURST_BEATS) ?
        MAX_BURST_BEATS[8:0] :
        beats_left_to_req_q[8:0];

    assign fifo_wr_data = {m_axi_rdata, beat0_q};
    assign fifo_wr_en   = r_fire && half_frame_q;

    assign fifo_rd_en = point_valid && point_ready;
    assign point_valid = !fifo_rd_empty;

    assign point_x =
        fifo_rd_data[DATA_W-1:0];

    assign point_y =
        fifo_rd_data[(2*DATA_W)-1:DATA_W];

    assign point_bucket_id =
        fifo_rd_data[512 +: GLOBAL_ADDR_W];

    assign point_last =
        fifo_rd_data[512 + GLOBAL_ADDR_W];

    assign axi_busy =
        (beats_left_to_rx_q != 0) ||
        m_axi_arvalid ||
        half_frame_q;

    always_ff @(posedge axi_clk or negedge axi_rst_n) begin
        if (!axi_rst_n) begin
            cur_araddr_q       <= '0;
            beats_left_to_req_q <= '0;
            beats_left_to_rx_q  <= '0;
            m_axi_araddr       <= '0;
            m_axi_arlen        <= '0;
            m_axi_arvalid      <= 1'b0;
            half_frame_q       <= 1'b0;
            beat0_q            <= '0;
            axi_done           <= 1'b0;
        end else begin
            axi_done <= 1'b0;

            if (start_axi) begin
                cur_araddr_q        <= base_addr;
                beats_left_to_req_q <= logical_point_count << 1;
                beats_left_to_rx_q  <= logical_point_count << 1;
                m_axi_arvalid       <= 1'b0;
                half_frame_q        <= 1'b0;
            end else begin
                if (!m_axi_arvalid && (beats_left_to_req_q != 0)) begin
                    m_axi_araddr  <= cur_araddr_q;
                    m_axi_arlen   <= request_beats[7:0] - 1'b1;
                    m_axi_arvalid <= 1'b1;
                end else if (ar_fire) begin
                    m_axi_arvalid       <= 1'b0;
                    cur_araddr_q        <= cur_araddr_q +
                                           ({55'd0, request_beats} << 6);
                    beats_left_to_req_q <= beats_left_to_req_q -
                                           request_beats;
                end

                if (r_fire) begin
                    beats_left_to_rx_q <= beats_left_to_rx_q - 1'b1;

                    if (!half_frame_q) begin
                        beat0_q      <= m_axi_rdata;
                        half_frame_q <= 1'b1;
                    end else begin
                        half_frame_q <= 1'b0;
                    end

                    if (beats_left_to_rx_q == 1)
                        axi_done <= 1'b1;
                end
            end
        end
    end

    always_ff @(posedge msm_clk or negedge msm_rst_n) begin
        if (!msm_rst_n) begin
            accepted_points_msm_q <= '0;
        end else if (fifo_rd_en) begin
            accepted_points_msm_q <= accepted_points_msm_q + 1'b1;
        end
    end

    async_fifo_gray_v1 #(
        .DATA_W (FRAME_W),
        .DEPTH  (ASYNC_FIFO_DEPTH)
    ) u_point_cdc_fifo (
        .wr_clk   (axi_clk),
        .wr_rst_n (axi_rst_n),
        .wr_en    (fifo_wr_en),
        .wr_data  (fifo_wr_data),
        .wr_full  (fifo_wr_full),

        .rd_clk   (msm_clk),
        .rd_rst_n (msm_rst_n),
        .rd_en    (fifo_rd_en),
        .rd_data  (fifo_rd_data),
        .rd_empty (fifo_rd_empty)
    );

    initial begin
        if (DATA_W != 256)
            $fatal(1, "axi_point_stream_source_dualclk_v1 expects DATA_W=256");

        if (GLOBAL_ADDR_W != 16)
            $fatal(1, "axi_point_stream_source_dualclk_v1 expects GLOBAL_ADDR_W=16");

        if ((MAX_BURST_BEATS < 1) || (MAX_BURST_BEATS > 256))
            $fatal(1, "MAX_BURST_BEATS must be in [1,256]");
    end

endmodule