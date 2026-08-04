`timescale 1ns/1ps

// ============================================================================
// File: tb/frontend/tb_axi_point_stream_source_dualclk_v1.sv
//
// Focused verification for axi_point_stream_source_dualclk_v1.
//
// Goals:
//   1. Verify two 512-bit AXI beats are paired into one logical point.
//   2. Verify CDC through async_fifo_gray_v1.
//   3. Prove sustained one accepted logical point per MSM cycle after warm-up.
//   4. Verify downstream backpressure causes no loss, duplication, or reordering.
//   5. Verify AXI burst length, address progression, RLAST, and total beat count.
//
// Clock relationship:
//   axi_clk = 500 MHz  (2 ns period)
//   msm_clk = 250 MHz  (4 ns period)
// ============================================================================

module tb_axi_point_stream_source_dualclk_v1;

    localparam int GLOBAL_ADDR_W    = 16;
    localparam int DATA_W           = 256;
    localparam int ASYNC_FIFO_DEPTH = 64;
    localparam int MAX_BURST_BEATS  = 256;

    localparam int NUM_POINTS       = 256;
    localparam int TOTAL_BEATS      = 2 * NUM_POINTS;
    localparam int BYTES_PER_BEAT   = 64;

    localparam logic [63:0] BASE_ADDR = 64'h0000_0000_0200_0000;

    localparam time AXI_CLK_PERIOD = 2ns;
    localparam time MSM_CLK_PERIOD = 4ns;

    logic axi_clk;
    logic axi_rst_n;
    logic msm_clk;
    logic msm_rst_n;

    logic        start_axi;
    logic [63:0] base_addr;
    logic [31:0] logical_point_count;
    logic        axi_busy;
    logic        axi_done;

    logic [63:0] m_axi_araddr;
    logic [7:0]  m_axi_arlen;
    logic [2:0]  m_axi_arsize;
    logic [1:0]  m_axi_arburst;
    logic        m_axi_arvalid;
    logic        m_axi_arready;

    logic [511:0] m_axi_rdata;
    logic [1:0]   m_axi_rresp;
    logic         m_axi_rlast;
    logic         m_axi_rvalid;
    logic         m_axi_rready;

    logic                     point_valid;
    logic                     point_ready;
    logic [GLOBAL_ADDR_W-1:0] point_bucket_id;
    logic [DATA_W-1:0]        point_x;
    logic [DATA_W-1:0]        point_y;
    logic                     point_last;

    logic [511:0] axi_mem [0:TOTAL_BEATS-1];

    integer error_count;
    integer ar_count;
    integer r_count;
    integer accepted_count;
    integer consecutive_accepts;
    integer max_consecutive_accepts;
    integer msm_cycle_count;
    integer first_accept_cycle;
    integer last_accept_cycle;

    logic slave_active_q;
    logic [63:0] slave_addr_q;
    logic [8:0] slave_beats_left_q;

    logic [31:0] expected_index_q;
    logic backpressure_phase_q;

    function automatic logic [DATA_W-1:0] mk_x(input int unsigned idx);
        logic [DATA_W-1:0] value;
        begin
            value = {16{16'hA5A5}};
            value[31:0] = idx;
            return value;
        end
    endfunction

    function automatic logic [DATA_W-1:0] mk_y(input int unsigned idx);
        logic [DATA_W-1:0] value;
        begin
            value = {16{16'h5A5A}};
            value[31:0] = idx ^ 32'h55AA_00FF;
            return value;
        end
    endfunction

    function automatic logic [GLOBAL_ADDR_W-1:0] mk_bucket(
        input int unsigned idx
    );
        return 16'((idx * 73 + 11) % 65535 + 1);
    endfunction

    initial axi_clk = 1'b0;
    always #(AXI_CLK_PERIOD/2) axi_clk = ~axi_clk;

    initial msm_clk = 1'b0;
    always #(MSM_CLK_PERIOD/2) msm_clk = ~msm_clk;

    axi_point_stream_source_dualclk_v1 #(
        .GLOBAL_ADDR_W    (GLOBAL_ADDR_W),
        .DATA_W           (DATA_W),
        .ASYNC_FIFO_DEPTH (ASYNC_FIFO_DEPTH),
        .MAX_BURST_BEATS  (MAX_BURST_BEATS)
    ) dut (
        .axi_clk             (axi_clk),
        .axi_rst_n           (axi_rst_n),
        .msm_clk             (msm_clk),
        .msm_rst_n           (msm_rst_n),

        .start_axi           (start_axi),
        .base_addr           (base_addr),
        .logical_point_count (logical_point_count),
        .axi_busy            (axi_busy),
        .axi_done            (axi_done),

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

    // ---------------------------------------------------------------------
    // Initialize packed AXI memory image.
    // ---------------------------------------------------------------------
    initial begin : init_memory
        integer i;
        for (i = 0; i < NUM_POINTS; i = i + 1) begin
            axi_mem[2*i] = {mk_y(i), mk_x(i)};
            axi_mem[2*i+1] = '0;
            axi_mem[2*i+1][GLOBAL_ADDR_W-1:0] = mk_bucket(i);
            axi_mem[2*i+1][GLOBAL_ADDR_W] = (i == NUM_POINTS-1);
        end
    end

    // ---------------------------------------------------------------------
    // AXI slave model:
    // - One outstanding burst.
    // - AR accepted whenever idle.
    // - Once a burst starts, RVALID remains asserted continuously and the
    //   next beat is presented on every AXI cycle.
    // - If RREADY falls, RVALID/RDATA/RLAST remain stable.
    //
    // This is important for the throughput test: the old model inserted an
    // artificial empty AXI cycle after every accepted R beat, limiting the
    // source to half of the intended AXI bandwidth.
    // ---------------------------------------------------------------------
    assign m_axi_arready = !slave_active_q;

    always @(posedge axi_clk or negedge axi_rst_n) begin : axi_slave
        integer mem_index;
        if (!axi_rst_n) begin
            slave_active_q     <= 1'b0;
            slave_addr_q       <= '0;
            slave_beats_left_q <= '0;
            m_axi_rdata        <= '0;
            m_axi_rresp        <= 2'b00;
            m_axi_rlast        <= 1'b0;
            m_axi_rvalid       <= 1'b0;
            ar_count           <= 0;
            r_count            <= 0;
        end else begin
            // Accept a new burst and preload its first beat.
            if (m_axi_arvalid && m_axi_arready) begin
                if (m_axi_arsize !== 3'b110) begin
                    error_count++;
                    $error("[AXI] ARSIZE expected 6, got %0d", m_axi_arsize);
                end

                if (m_axi_arburst !== 2'b01) begin
                    error_count++;
                    $error("[AXI] ARBURST expected INCR, got %0b",
                           m_axi_arburst);
                end

                if (m_axi_araddr[5:0] != 0) begin
                    error_count++;
                    $error("[AXI] unaligned ARADDR=%h", m_axi_araddr);
                end

                mem_index = (m_axi_araddr - BASE_ADDR) >> 6;
                if ((mem_index < 0) || (mem_index >= TOTAL_BEATS)) begin
                    error_count++;
                    $error("[AXI] invalid first mem_index=%0d addr=%h",
                           mem_index, m_axi_araddr);
                    m_axi_rdata <= '0;
                end else begin
                    m_axi_rdata <= axi_mem[mem_index];
                end

                slave_active_q     <= 1'b1;
                slave_addr_q       <= m_axi_araddr;
                slave_beats_left_q <= {1'b0, m_axi_arlen} + 9'd1;
                m_axi_rresp        <= 2'b00;
                m_axi_rlast        <= (m_axi_arlen == 0);
                m_axi_rvalid       <= 1'b1;
                ar_count           <= ar_count + 1;
            end

            // Consume one beat. If more beats remain, preload the next beat
            // and keep RVALID asserted without inserting a bubble.
            if (m_axi_rvalid && m_axi_rready) begin
                r_count <= r_count + 1;

                if (slave_beats_left_q == 1) begin
                    slave_active_q     <= 1'b0;
                    slave_beats_left_q <= '0;
                    m_axi_rvalid       <= 1'b0;
                    m_axi_rlast        <= 1'b0;
                end else begin
                    slave_addr_q       <= slave_addr_q + BYTES_PER_BEAT;
                    slave_beats_left_q <= slave_beats_left_q - 1'b1;

                    mem_index =
                        ((slave_addr_q + BYTES_PER_BEAT) - BASE_ADDR) >> 6;

                    if ((mem_index < 0) || (mem_index >= TOTAL_BEATS)) begin
                        error_count++;
                        $error("[AXI] invalid next mem_index=%0d addr=%h",
                               mem_index,
                               slave_addr_q + BYTES_PER_BEAT);
                        m_axi_rdata <= '0;
                    end else begin
                        m_axi_rdata <= axi_mem[mem_index];
                    end

                    m_axi_rlast <= (slave_beats_left_q == 2);
                    m_axi_rvalid <= 1'b1;
                end
            end
        end
    end

    // ---------------------------------------------------------------------
    // MSM-side ready generation.
    //
    // Phase 1:
    //   Keep ready high to measure sustained II=1.
    //
    // Phase 2:
    //   Add deterministic backpressure after 128 points.
    // ---------------------------------------------------------------------
    always_ff @(posedge msm_clk or negedge msm_rst_n) begin
        if (!msm_rst_n) begin
            point_ready         <= 1'b0;
            backpressure_phase_q <= 1'b0;
        end else begin
            if (accepted_count < 128) begin
                point_ready <= 1'b1;
            end else begin
                backpressure_phase_q <= 1'b1;
                // Pattern: 5 ready cycles, 3 blocked cycles.
                point_ready <= ((msm_cycle_count % 8) < 5);
            end
        end
    end

    // ---------------------------------------------------------------------
    // Scoreboard and throughput monitor.
    // ---------------------------------------------------------------------
    always @(posedge msm_clk or negedge msm_rst_n) begin
        if (!msm_rst_n) begin
            expected_index_q       <= 0;
            accepted_count         <= 0;
            consecutive_accepts    <= 0;
            max_consecutive_accepts <= 0;
            msm_cycle_count        <= 0;
            first_accept_cycle     <= -1;
            last_accept_cycle      <= -1;
        end else begin
            msm_cycle_count <= msm_cycle_count + 1;

            if (point_valid && point_ready) begin
                if (point_x !== mk_x(expected_index_q)) begin
                    error_count++;
                    $error("[MSM] X mismatch idx=%0d", expected_index_q);
                end

                if (point_y !== mk_y(expected_index_q)) begin
                    error_count++;
                    $error("[MSM] Y mismatch idx=%0d", expected_index_q);
                end

                if (point_bucket_id !== mk_bucket(expected_index_q)) begin
                    error_count++;
                    $error("[MSM] bucket mismatch idx=%0d exp=%0h got=%0h",
                           expected_index_q,
                           mk_bucket(expected_index_q),
                           point_bucket_id);
                end

                if (point_last !== (expected_index_q == NUM_POINTS-1)) begin
                    error_count++;
                    $error("[MSM] last mismatch idx=%0d exp=%0b got=%0b",
                           expected_index_q,
                           (expected_index_q == NUM_POINTS-1),
                           point_last);
                end

                if (first_accept_cycle < 0)
                    first_accept_cycle <= msm_cycle_count;

                last_accept_cycle <= msm_cycle_count;
                expected_index_q  <= expected_index_q + 1'b1;
                accepted_count    <= accepted_count + 1;

                consecutive_accepts <= consecutive_accepts + 1;
                if (consecutive_accepts + 1 > max_consecutive_accepts)
                    max_consecutive_accepts <= consecutive_accepts + 1;
            end else begin
                consecutive_accepts <= 0;
            end
        end
    end

    initial begin : test_sequence
        error_count = 0;

        axi_rst_n = 1'b0;
        msm_rst_n = 1'b0;
        start_axi = 1'b0;
        base_addr = BASE_ADDR;
        logical_point_count = NUM_POINTS;

        repeat (8) @(posedge axi_clk);
        axi_rst_n = 1'b1;

        repeat (4) @(posedge msm_clk);
        msm_rst_n = 1'b1;

        repeat (4) @(posedge axi_clk);
        @(negedge axi_clk);
        start_axi = 1'b1;
        @(negedge axi_clk);
        start_axi = 1'b0;

        wait (accepted_count == NUM_POINTS);

        repeat (8) @(posedge msm_clk);

        if (r_count != TOTAL_BEATS) begin
            error_count++;
            $error("[TB] expected %0d R beats, got %0d",
                   TOTAL_BEATS, r_count);
        end

        if (accepted_count != NUM_POINTS) begin
            error_count++;
            $error("[TB] expected %0d accepted points, got %0d",
                   NUM_POINTS, accepted_count);
        end

        if (expected_index_q != NUM_POINTS) begin
            error_count++;
            $error("[TB] expected scoreboard index %0d, got %0d",
                   NUM_POINTS, expected_index_q);
        end

        if (max_consecutive_accepts < 96) begin
            error_count++;
            $error("[TB] II=1 proof too short: expected at least 96, got %0d",
                   max_consecutive_accepts);
        end

        if (error_count == 0) begin
            $display("");
            $display("============================================================");
            $display("[TB_DUALCLK] AXI DUAL-CLOCK POINT SOURCE PASSED");
            $display("[TB_DUALCLK] AXI clock period          = %0t",
                     AXI_CLK_PERIOD);
            $display("[TB_DUALCLK] MSM clock period          = %0t",
                     MSM_CLK_PERIOD);
            $display("[TB_DUALCLK] logical points            = %0d",
                     NUM_POINTS);
            $display("[TB_DUALCLK] AXI beats                 = %0d",
                     r_count);
            $display("[TB_DUALCLK] AXI bursts                = %0d",
                     ar_count);
            $display("[TB_DUALCLK] accepted points           = %0d",
                     accepted_count);
            $display("[TB_DUALCLK] max consecutive accepts   = %0d",
                     max_consecutive_accepts);
            $display("[TB_DUALCLK] first accept MSM cycle    = %0d",
                     first_accept_cycle);
            $display("[TB_DUALCLK] last accept MSM cycle     = %0d",
                     last_accept_cycle);
            $display("[TB_DUALCLK] verified 2-beat pairing, CDC, II=1,");
            $display("[TB_DUALCLK] ordering, last flag, and backpressure");
            $display("============================================================");
        end else begin
            $fatal(1, "[TB_DUALCLK] FAILED with %0d errors", error_count);
        end

        $finish;
    end

    initial begin : watchdog
        #2ms;
        $fatal(1, "[TB_DUALCLK] WATCHDOG TIMEOUT");
    end

endmodule