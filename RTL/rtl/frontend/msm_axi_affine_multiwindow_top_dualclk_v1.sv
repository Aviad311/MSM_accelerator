`timescale 1ns/1ps

module msm_axi_affine_multiwindow_top_dualclk_v1 #(
    parameter int ADDR_W           = 16,
    parameter int DATA_W           = 256,
    parameter int DEPTH            = (1 << ADDR_W),
    parameter int SRAM_RD_LATENCY  = 1,
    parameter int GEN_W            = 16,
    parameter int FIFO_DEPTH       = 16,
    parameter int SLOT_COUNT       = 16,
    parameter int MIX_CTX_COUNT    = 40,
    parameter int MUL_LATENCY      = 16,
    parameter int WINDOW_BITS      = 16,
    parameter int NUM_WINDOWS      = 16,
    parameter int ASYNC_FIFO_DEPTH = 64,
    parameter int MAX_BURST_BEATS  = 16,
    parameter int CONV_FIFO_DEPTH  = 32
)(
    input  logic axi_clk,
    input  logic axi_rst_n,

    input  logic msm_clk,
    input  logic msm_rst_n,

    input  logic        start,
    input  logic [63:0] base_addr,
    input  logic [31:0] logical_points_per_window,

    output logic busy,
    output logic done,
    output logic [$clog2(NUM_WINDOWS)-1:0] window_index,

    output logic [DATA_W-1:0] result_x,
    output logic [DATA_W-1:0] result_y,
    output logic [DATA_W-1:0] result_z,

    output logic [63:0] m_axi_araddr,
    output logic [7:0]  m_axi_arlen,
    output logic [2:0]  m_axi_arsize,
    output logic [1:0]  m_axi_arburst,
    output logic        m_axi_arvalid,
    input  logic        m_axi_arready,

    input  logic [511:0] m_axi_rdata,
    input  logic [1:0]   m_axi_rresp,
    input  logic         m_axi_rlast,
    input  logic         m_axi_rvalid,
    output logic         m_axi_rready,

    output logic converter_busy,
    output logic [$clog2(CONV_FIFO_DEPTH+1)-1:0]
                 converter_pending_count,
    output logic [$clog2(CONV_FIFO_DEPTH+1)-1:0]
                 converter_result_count
);

    localparam int WIN_IDX_W =
        (NUM_WINDOWS <= 1) ? 1 : $clog2(NUM_WINDOWS);

    logic controller_start_q;
    logic controller_busy;
    logic controller_done;

    logic source_start_axi_q;
    logic source_axi_busy;
    logic source_axi_done;

    // Normal-domain point stream from the AXI/async-FIFO source.
    logic point_valid;
    logic point_ready;
    logic [ADDR_W-1:0]  point_bucket_id;
    logic [DATA_W-1:0]  point_x;
    logic [DATA_W-1:0]  point_y;
    logic               point_last;

    // Montgomery-domain point stream into the MSM controller.
    logic mont_valid;
    logic mont_ready;
    logic [ADDR_W-1:0]  mont_bucket_id;
    logic [DATA_W-1:0]  mont_x;
    logic [DATA_W-1:0]  mont_y;
    logic               mont_last;

    logic [WIN_IDX_W-1:0] launched_window_msm_q;
    logic have_launched_window_msm_q;

    logic [WIN_IDX_W-1:0] launch_window_axi_sync1_q;
    logic [WIN_IDX_W-1:0] launch_window_axi_sync2_q;
    logic launch_toggle_msm_q;
    logic launch_toggle_axi_sync1_q;
    logic launch_toggle_axi_sync2_q;
    logic launch_toggle_axi_seen_q;

    logic [63:0] bytes_per_window;
    logic [63:0] source_base_addr_axi;

    assign bytes_per_window =
        {32'd0, logical_points_per_window} << 7;

    assign source_base_addr_axi =
        base_addr + bytes_per_window * launch_window_axi_sync2_q;

    assign busy = controller_busy;
    assign done = controller_done;

    // Start the MSM controller in the MSM clock domain.
    always_ff @(posedge msm_clk or negedge msm_rst_n) begin
        if (!msm_rst_n) begin
            controller_start_q <= 1'b0;
        end else begin
            controller_start_q <= 1'b0;
            if (start && !controller_busy)
                controller_start_q <= 1'b1;
        end
    end

    // Detect controller window changes and send an event to the AXI domain.
    always_ff @(posedge msm_clk or negedge msm_rst_n) begin
        if (!msm_rst_n) begin
            launched_window_msm_q      <= '0;
            have_launched_window_msm_q <= 1'b0;
            launch_toggle_msm_q        <= 1'b0;
        end else begin
            if (start && !controller_busy) begin
                have_launched_window_msm_q <= 1'b0;
            end else if (controller_busy) begin
                if (!have_launched_window_msm_q ||
                    (window_index != launched_window_msm_q)) begin
                    launched_window_msm_q      <= window_index;
                    have_launched_window_msm_q <= 1'b1;
                    launch_toggle_msm_q        <= ~launch_toggle_msm_q;
                end
            end

            if (controller_done)
                have_launched_window_msm_q <= 1'b0;
        end
    end

    // Synchronize launch payload/event into the AXI clock domain.
    always_ff @(posedge axi_clk or negedge axi_rst_n) begin
        if (!axi_rst_n) begin
            launch_window_axi_sync1_q <= '0;
            launch_window_axi_sync2_q <= '0;
            launch_toggle_axi_sync1_q <= 1'b0;
            launch_toggle_axi_sync2_q <= 1'b0;
            launch_toggle_axi_seen_q  <= 1'b0;
            source_start_axi_q        <= 1'b0;
        end else begin
            launch_window_axi_sync1_q <= launched_window_msm_q;
            launch_window_axi_sync2_q <= launch_window_axi_sync1_q;
            launch_toggle_axi_sync1_q <= launch_toggle_msm_q;
            launch_toggle_axi_sync2_q <= launch_toggle_axi_sync1_q;

            source_start_axi_q <= 1'b0;

            if ((launch_toggle_axi_sync2_q != launch_toggle_axi_seen_q) &&
                !source_axi_busy) begin
                launch_toggle_axi_seen_q <= launch_toggle_axi_sync2_q;
                source_start_axi_q       <= 1'b1;
            end
        end
    end

    // AXI read + async FIFO + point unpacker.
    // This source now carries normal-domain affine coordinates.
    axi_point_stream_source_dualclk_v1 #(
        .GLOBAL_ADDR_W    (ADDR_W),
        .DATA_W           (DATA_W),
        .ASYNC_FIFO_DEPTH (ASYNC_FIFO_DEPTH),
        .MAX_BURST_BEATS  (MAX_BURST_BEATS)
    ) u_axi_source (
        .axi_clk             (axi_clk),
        .axi_rst_n           (axi_rst_n),
        .msm_clk             (msm_clk),
        .msm_rst_n           (msm_rst_n),

        .start_axi           (source_start_axi_q),
        .base_addr           (source_base_addr_axi),
        .logical_point_count (logical_points_per_window),
        .axi_busy            (source_axi_busy),
        .axi_done            (source_axi_done),

        .m_axi_araddr        (m_axi_araddr),
        .m_axi_arlen         (m_axi_arlen),
        .m_axi_arsize        (m_axi_arsize),
        .m_axi_arburst       (m_axi_arburst),
        .m_axi_arvalid       (m_axi_arvalid),
        .m_axi_arready       (m_axi_arready),

        .m_axi_rdata         (m_axi_rdata),
        .m_axi_rresp         (m_axi_rresp),
        .m_axi_rlast         (m_axi_rlast),
        .m_axi_rvalid        (m_axi_rvalid),
        .m_axi_rready        (m_axi_rready),

        .point_valid         (point_valid),
        .point_ready         (point_ready),
        .point_bucket_id     (point_bucket_id),
        .point_x             (point_x),
        .point_y             (point_y),
        .point_last          (point_last)
    );

    // Convert normal affine coordinates to Montgomery representation in
    // the MSM clock domain, after the clock-domain crossing.
    point_to_montgomery_stream #(
        .DATA_W      (DATA_W),
        .BUCKET_W    (ADDR_W),
        .MUL_LATENCY (MUL_LATENCY),
        .FIFO_DEPTH  (CONV_FIFO_DEPTH)
    ) u_point_to_montgomery (
        .clk               (msm_clk),
        .rst_n             (msm_rst_n),

        .in_valid          (point_valid),
        .in_ready          (point_ready),
        .in_point_x        (point_x),
        .in_point_y        (point_y),
        .in_bucket_id      (point_bucket_id),
        .in_last_point     (point_last),

        .out_valid         (mont_valid),
        .out_ready         (mont_ready),
        .out_point_x_m     (mont_x),
        .out_point_y_m     (mont_y),
        .out_bucket_id     (mont_bucket_id),
        .out_last_point    (mont_last),

        .busy              (converter_busy),
        .pending_count_dbg (converter_pending_count),
        .result_count_dbg  (converter_result_count)
    );

    msm_multiwindow_controller_v1 #(
        .ADDR_W          (ADDR_W),
        .DATA_W          (DATA_W),
        .DEPTH           (DEPTH),
        .SRAM_RD_LATENCY (SRAM_RD_LATENCY),
        .GEN_W           (GEN_W),
        .FIFO_DEPTH      (FIFO_DEPTH),
        .SLOT_COUNT      (SLOT_COUNT),
        .MIX_CTX_COUNT   (MIX_CTX_COUNT),
        .MUL_LATENCY     (MUL_LATENCY),
        .WINDOW_BITS     (WINDOW_BITS),
        .NUM_WINDOWS     (NUM_WINDOWS)
    ) u_controller (
        .clk          (msm_clk),
        .rst_n        (msm_rst_n),
        .start        (controller_start_q),

        .in_valid     (mont_valid),
        .in_ready     (mont_ready),
        .in_bucket_id (mont_bucket_id),
        .in_point_x   (mont_x),
        .in_point_y   (mont_y),
        .last_point   (mont_last),

        .window_index (window_index),
        .busy         (controller_busy),
        .done         (controller_done),

        .result_x     (result_x),
        .result_y     (result_y),
        .result_z     (result_z)
    );

    initial begin
        if (DATA_W != 256)
            $fatal(1, "[AXI_AFFINE_TOP] DATA_W must be 256");
        if (NUM_WINDOWS < 2)
            $fatal(1, "[AXI_AFFINE_TOP] NUM_WINDOWS must be at least 2");
    end

endmodule