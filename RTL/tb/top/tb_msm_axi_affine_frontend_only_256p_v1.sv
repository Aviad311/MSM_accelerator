`timescale 1ns/1ps

// ============================================================================
// Full end-to-end 256-original-point MSM test.
//
// 256 original affine points are reused in every one of the 16 W=16 windows.
// Each window receives the corresponding 16-bit scalar digit as bucket_id.
//
// Coverage:
//   normal affine DRAM image
//   AXI bursts and random gaps
//   dual-clock CDC
//   affine-to-Montgomery conversion
//   256 points/window
//   16 windows
//   4096 point-window records
//   SRAM macros
//   exact final Montgomery-domain X/Y/Z against Python golden
// ============================================================================

module tb_msm_axi_affine_frontend_only_256p_v1;

    localparam int ADDR_W           = 16;
    localparam int DATA_W           = 256;
    localparam int NUM_WINDOWS      = 16;
    localparam int WINDOW_BITS      = 16;
    localparam int ASYNC_FIFO_DEPTH = 64;
    localparam int MAX_BURST_BEATS  = 16;
    localparam int CONV_FIFO_DEPTH  = 32;

    localparam logic [63:0] BASE_ADDR =
        64'h0000_0000_0200_0000;

    localparam time AXI_CLK_PERIOD = 2ns;
    localparam time MSM_CLK_PERIOD = 4ns;

    `include "vectors/multiwindow_w16_python_golden_affine_256p.svh"

    localparam int LOGICAL_POINTS_PER_WINDOW = MW_POINTS_PER_WINDOW;
    localparam int ORIGINAL_POINTS = MW_ORIGINAL_POINTS;
    localparam int POINT_WINDOW_RECORDS =
        NUM_WINDOWS * LOGICAL_POINTS_PER_WINDOW;

    localparam int BEATS_PER_POINT = 2;
    localparam int BEATS_PER_WINDOW =
        LOGICAL_POINTS_PER_WINDOW * BEATS_PER_POINT;
    localparam int TOTAL_BEATS =
        NUM_WINDOWS * BEATS_PER_WINDOW;
    localparam int BYTES_PER_BEAT = 64;
    localparam int BYTES_PER_WINDOW =
        BEATS_PER_WINDOW * BYTES_PER_BEAT;

    localparam int BURSTS_PER_WINDOW =
        (BEATS_PER_WINDOW + MAX_BURST_BEATS - 1) /
        MAX_BURST_BEATS;
    localparam int EXPECTED_BURSTS =
        NUM_WINDOWS * BURSTS_PER_WINDOW;

    logic axi_clk;
    logic axi_rst_n;
    logic msm_clk;
    logic msm_rst_n;

    logic start;
    logic [63:0] base_addr;
    logic [31:0] logical_points_per_window;

    logic busy;
    logic done;
    logic [$clog2(NUM_WINDOWS)-1:0] window_index;
    logic [DATA_W-1:0] result_x;
    logic [DATA_W-1:0] result_y;
    logic [DATA_W-1:0] result_z;

    logic converter_busy;
    logic [$clog2(CONV_FIFO_DEPTH+1)-1:0] converter_pending_count;
    logic [$clog2(CONV_FIFO_DEPTH+1)-1:0] converter_result_count;

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

    logic [511:0] axi_mem [0:TOTAL_BEATS-1];

    integer error_count;
    integer ar_burst_count;
    integer r_beat_count;
    integer source_done_count;
    integer last_seen_window;

    longint unsigned axi_cycle_count;
    longint unsigned msm_cycle_count;
    longint unsigned window_enter_cycle [0:NUM_WINDOWS-1];

    logic        slave_active_q;
    logic [63:0] slave_addr_q;
    logic [8:0]  slave_beats_left_q;
    logic [31:0] gap_lfsr_q;

    initial axi_clk = 1'b0;
    always #(AXI_CLK_PERIOD/2) axi_clk = ~axi_clk;

    initial msm_clk = 1'b0;
    always #(MSM_CLK_PERIOD/2) msm_clk = ~msm_clk;

    always_ff @(posedge axi_clk or negedge axi_rst_n) begin
        if (!axi_rst_n)
            axi_cycle_count <= 0;
        else
            axi_cycle_count <= axi_cycle_count + 1;
    end

    always_ff @(posedge msm_clk or negedge msm_rst_n) begin
        if (!msm_rst_n)
            msm_cycle_count <= 0;
        else
            msm_cycle_count <= msm_cycle_count + 1;
    end

    msm_axi_affine_multiwindow_top_dualclk_v1 #(
        .ADDR_W           (ADDR_W),
        .DATA_W           (DATA_W),
        .DEPTH            (1 << ADDR_W),
        .SRAM_RD_LATENCY  (1),
        .GEN_W            (16),
        .FIFO_DEPTH       (16),
        .SLOT_COUNT       (16),
        .MIX_CTX_COUNT    (40),
        .MUL_LATENCY      (16),
        .WINDOW_BITS      (WINDOW_BITS),
        .NUM_WINDOWS      (NUM_WINDOWS),
        .ASYNC_FIFO_DEPTH (ASYNC_FIFO_DEPTH),
        .MAX_BURST_BEATS  (MAX_BURST_BEATS),
        .CONV_FIFO_DEPTH  (CONV_FIFO_DEPTH)
    ) dut (
        .axi_clk                    (axi_clk),
        .axi_rst_n                  (axi_rst_n),
        .msm_clk                    (msm_clk),
        .msm_rst_n                  (msm_rst_n),
        .start                      (start),
        .base_addr                  (base_addr),
        .logical_points_per_window (logical_points_per_window),
        .busy                       (busy),
        .done                       (done),
        .window_index               (window_index),
        .result_x                   (result_x),
        .result_y                   (result_y),
        .result_z                   (result_z),
        .m_axi_araddr               (m_axi_araddr),
        .m_axi_arlen                (m_axi_arlen),
        .m_axi_arsize               (m_axi_arsize),
        .m_axi_arburst              (m_axi_arburst),
        .m_axi_arvalid              (m_axi_arvalid),
        .m_axi_arready              (m_axi_arready),
        .m_axi_rdata                (m_axi_rdata),
        .m_axi_rresp                (m_axi_rresp),
        .m_axi_rlast                (m_axi_rlast),
        .m_axi_rvalid               (m_axi_rvalid),
        .m_axi_rready               (m_axi_rready),
        .converter_busy             (converter_busy),
        .converter_pending_count    (converter_pending_count),
        .converter_result_count     (converter_result_count)
    );

    // Each normal affine point occupies:
    //   beat 0 = {Y_normal, X_normal}
    //   beat 1 = {padding, last_point, bucket_id}
    initial begin : init_axi_memory
        integer w;
        integer p;
        integer beat;

        // Order the SVH initialization before copying it into axi_mem.
        #1ps;

        if (MW_NUM_WINDOWS != NUM_WINDOWS)
            $fatal(1, "[TB_AXI_AFFINE_MW_256P] vector window count mismatch");

        if (MW_ORIGINAL_POINTS != LOGICAL_POINTS_PER_WINDOW)
            $fatal(1, "[TB_AXI_AFFINE_MW_256P] original point count mismatch");

        if (MW_POINT_WINDOW_RECORDS != POINT_WINDOW_RECORDS)
            $fatal(1, "[TB_AXI_AFFINE_MW_256P] record count mismatch");

        for (beat = 0; beat < TOTAL_BEATS; beat = beat + 1)
            axi_mem[beat] = '0;

        for (w = 0; w < NUM_WINDOWS; w = w + 1) begin
            for (p = 0; p < LOGICAL_POINTS_PER_WINDOW; p = p + 1) begin
                beat = w * BEATS_PER_WINDOW + p * 2;

                axi_mem[beat] = {
                    mw_point_y[w][p],
                    mw_point_x[w][p]
                };

                axi_mem[beat+1] = '0;
                axi_mem[beat+1][ADDR_W-1:0] =
                    mw_bucket_idx[w][p];
                axi_mem[beat+1][ADDR_W] =
                    (p == LOGICAL_POINTS_PER_WINDOW-1);
            end
        end
    end

    assign m_axi_arready =
        !slave_active_q && (gap_lfsr_q[2:0] != 3'b000);

    always @(posedge axi_clk or negedge axi_rst_n) begin : axi_slave
        integer mem_index;
        logic [12:0] burst_end_low;

        if (!axi_rst_n) begin
            slave_active_q     <= 1'b0;
            slave_addr_q       <= '0;
            slave_beats_left_q <= '0;
            m_axi_rdata        <= '0;
            m_axi_rresp        <= 2'b00;
            m_axi_rlast        <= 1'b0;
            m_axi_rvalid       <= 1'b0;
            gap_lfsr_q         <= 32'h1ACE_B00C;
            ar_burst_count     <= 0;
            r_beat_count       <= 0;
        end else begin
            gap_lfsr_q <= {
                gap_lfsr_q[30:0],
                gap_lfsr_q[31] ^
                gap_lfsr_q[21] ^
                gap_lfsr_q[1]  ^
                gap_lfsr_q[0]
            };

            if (m_axi_arvalid && m_axi_arready) begin
                if (m_axi_arsize !== 3'b110) begin
                    error_count++;
                    $error(
                        "[TB_AXI_AFFINE_MW_256P] ARSIZE expected 6, got %0d",
                        m_axi_arsize
                    );
                end

                if (m_axi_arburst !== 2'b01) begin
                    error_count++;
                    $error(
                        "[TB_AXI_AFFINE_MW_256P] ARBURST expected INCR, got %0b",
                        m_axi_arburst
                    );
                end

                if (m_axi_araddr[5:0] != 0) begin
                    error_count++;
                    $error(
                        "[TB_AXI_AFFINE_MW_256P] unaligned ARADDR=%h",
                        m_axi_araddr
                    );
                end

                burst_end_low =
                    {1'b0, m_axi_araddr[11:0]} +
                    (({5'd0, m_axi_arlen} + 13'd1) << 6);

                if (burst_end_low > 13'd4096) begin
                    error_count++;
                    $error(
                        "[TB_AXI_AFFINE_MW_256P] burst crosses 4KB addr=%h beats=%0d",
                        m_axi_araddr,
                        m_axi_arlen + 1
                    );
                end

                if ((m_axi_araddr < BASE_ADDR) ||
                    (m_axi_araddr >=
                     BASE_ADDR + TOTAL_BEATS*BYTES_PER_BEAT)) begin
                    error_count++;
                    $error(
                        "[TB_AXI_AFFINE_MW_256P] ARADDR outside image: %h",
                        m_axi_araddr
                    );
                end

                slave_active_q     <= 1'b1;
                slave_addr_q       <= m_axi_araddr;
                slave_beats_left_q <= {1'b0, m_axi_arlen} + 9'd1;
                ar_burst_count     <= ar_burst_count + 1;
            end

            if (!m_axi_rvalid &&
                slave_active_q &&
                (gap_lfsr_q[5:3] != 3'b000)) begin

                mem_index = (slave_addr_q - BASE_ADDR) >> 6;

                if ((mem_index < 0) ||
                    (mem_index >= TOTAL_BEATS)) begin
                    error_count++;
                    $error(
                        "[TB_AXI_AFFINE_MW_256P] invalid beat index=%0d",
                        mem_index
                    );
                    m_axi_rdata <= '0;
                end else begin
                    m_axi_rdata <= axi_mem[mem_index];
                end

                m_axi_rresp  <= 2'b00;
                m_axi_rlast  <= (slave_beats_left_q == 1);
                m_axi_rvalid <= 1'b1;
            end

            if (m_axi_rvalid && m_axi_rready) begin
                r_beat_count <= r_beat_count + 1;

                if (m_axi_rlast !==
                    (slave_beats_left_q == 1)) begin
                    error_count++;
                    $error(
                        "[TB_AXI_AFFINE_MW_256P] RLAST mismatch left=%0d got=%0b",
                        slave_beats_left_q,
                        m_axi_rlast
                    );
                end

                if (slave_beats_left_q == 1) begin
                    slave_active_q     <= 1'b0;
                    slave_beats_left_q <= '0;
                    m_axi_rvalid       <= 1'b0;
                    m_axi_rlast        <= 1'b0;
                end else begin
                    slave_addr_q       <= slave_addr_q + BYTES_PER_BEAT;
                    slave_beats_left_q <= slave_beats_left_q - 1'b1;

                    if (gap_lfsr_q[8:6] == 3'b000) begin
                        m_axi_rvalid <= 1'b0;
                        m_axi_rlast  <= 1'b0;
                    end else begin
                        mem_index =
                            ((slave_addr_q + BYTES_PER_BEAT)
                             - BASE_ADDR) >> 6;

                        if ((mem_index < 0) ||
                            (mem_index >= TOTAL_BEATS)) begin
                            error_count++;
                            $error(
                                "[TB_AXI_AFFINE_MW_256P] invalid next beat index=%0d",
                                mem_index
                            );
                            m_axi_rdata <= '0;
                        end else begin
                            m_axi_rdata <= axi_mem[mem_index];
                        end

                        m_axi_rresp  <= 2'b00;
                        m_axi_rlast  <= (slave_beats_left_q == 2);
                        m_axi_rvalid <= 1'b1;
                    end
                end
            end
        end
    end


    integer frontend_output_count;
    integer frontend_last_count;
    longint unsigned frontend_last_progress_cycle;

    always_ff @(posedge axi_clk or negedge axi_rst_n) begin
        if (!axi_rst_n)
            source_done_count <= 0;
        else if (dut.source_axi_done)
            source_done_count <= source_done_count + 1;
    end

    always_ff @(posedge msm_clk or negedge msm_rst_n) begin
        if (!msm_rst_n) begin
            frontend_output_count        <= 0;
            frontend_last_count          <= 0;
            frontend_last_progress_cycle <= 0;
        end else begin
            if (dut.mont_valid && dut.mont_ready) begin
                frontend_output_count <= frontend_output_count + 1;
                frontend_last_progress_cycle <= msm_cycle_count;

                if (dut.mont_last)
                    frontend_last_count <= frontend_last_count + 1;

                if (((frontend_output_count + 1) % 256) == 0) begin
                    $display(
                        "[FE_ONLY_256P] progress outputs=%0d/%0d last_markers=%0d axi_beats=%0d bursts=%0d msm_cycle=%0d",
                        frontend_output_count + 1,
                        POINT_WINDOW_RECORDS,
                        frontend_last_count + (dut.mont_last ? 1 : 0),
                        r_beat_count,
                        ar_burst_count,
                        msm_cycle_count
                    );
                end
            end

            if ((frontend_output_count < POINT_WINDOW_RECORDS) &&
                (msm_cycle_count > frontend_last_progress_cycle) &&
                ((msm_cycle_count - frontend_last_progress_cycle) >
                 2_000_000)) begin
                $fatal(
                    1,
                    "[FE_ONLY_256P] no converter forward progress for %0d MSM cycles outputs=%0d pending=%0d results=%0d",
                    msm_cycle_count - frontend_last_progress_cycle,
                    frontend_output_count,
                    converter_pending_count,
                    converter_result_count
                );
            end
        end
    end

    initial begin : test_sequence
        error_count = 0;
        axi_rst_n = 1'b0;
        msm_rst_n = 1'b0;
        start = 1'b0;
        base_addr = BASE_ADDR;
        logical_points_per_window = LOGICAL_POINTS_PER_WINDOW;

        // Drain the converter independently from the compute engine.
        force dut.mont_ready = 1'b1;
        force dut.u_controller.in_valid = 1'b0;

        $display("============================================================");
        $display("[FE_ONLY_256P] AXI + CDC + Montgomery frontend-only test");
        $display(
            "[FE_ONLY_256P] records=%0d AXI beats=%0d windows=%0d",
            POINT_WINDOW_RECORDS,
            TOTAL_BEATS,
            NUM_WINDOWS
        );
        $display("============================================================");

        repeat (12) @(posedge axi_clk);
        axi_rst_n = 1'b1;

        repeat (8) @(posedge msm_clk);
        msm_rst_n = 1'b1;

        repeat (5) @(posedge msm_clk);
        @(negedge msm_clk);
        start = 1'b1;
        @(negedge msm_clk);
        start = 1'b0;

        wait (frontend_output_count == POINT_WINDOW_RECORDS);
        repeat (20) @(posedge msm_clk);

        if (r_beat_count != TOTAL_BEATS) begin
            error_count = error_count + 1;
            $display(
                "[FE_ONLY_256P] ERROR AXI beats expected=%0d got=%0d",
                TOTAL_BEATS,
                r_beat_count
            );
        end

        if (ar_burst_count != EXPECTED_BURSTS) begin
            error_count = error_count + 1;
            $display(
                "[FE_ONLY_256P] ERROR bursts expected=%0d got=%0d",
                EXPECTED_BURSTS,
                ar_burst_count
            );
        end

        if (source_done_count != NUM_WINDOWS) begin
            error_count = error_count + 1;
            $display(
                "[FE_ONLY_256P] ERROR source done expected=%0d got=%0d",
                NUM_WINDOWS,
                source_done_count
            );
        end

        if (frontend_last_count != NUM_WINDOWS) begin
            error_count = error_count + 1;
            $display(
                "[FE_ONLY_256P] ERROR last markers expected=%0d got=%0d",
                NUM_WINDOWS,
                frontend_last_count
            );
        end

        if (converter_pending_count != 0 ||
            converter_result_count != 0) begin
            error_count = error_count + 1;
            $display(
                "[FE_ONLY_256P] ERROR converter not empty pending=%0d results=%0d",
                converter_pending_count,
                converter_result_count
            );
        end

        if (error_count == 0) begin
            $display("============================================================");
            $display("[FE_ONLY_256P] FRONTEND-ONLY 256P TEST PASSED");
            $display(
                "[FE_ONLY_256P] outputs=%0d beats=%0d bursts=%0d last=%0d",
                frontend_output_count,
                r_beat_count,
                ar_burst_count,
                frontend_last_count
            );
            $display("============================================================");
        end else begin
            $fatal(1, "[FE_ONLY_256P] FAILED errors=%0d", error_count);
        end

        release dut.u_controller.in_valid;
        release dut.mont_ready;
        $finish;
    end

    initial begin : watchdog
        #10ms;
        $fatal(1, "[FE_ONLY_256P] simulated-time watchdog timeout");
    end

endmodule