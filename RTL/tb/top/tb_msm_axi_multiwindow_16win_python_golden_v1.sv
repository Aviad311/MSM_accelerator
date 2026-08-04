`timescale 1ns/1ps

module tb_msm_axi_multiwindow_16win_python_golden_v1;

    localparam int ADDR_W          = 16;
    localparam int DATA_W          = 256;
    localparam int NUM_WINDOWS     = 16;
    localparam int WINDOW_BITS     = 16;
    localparam int AXI_FIFO_DEPTH  = 64;
    localparam int MAX_BURST_BEATS = 16;

    localparam logic [63:0] BASE_ADDR = 64'h0000_0000_0100_0000;
    localparam time CLK_PERIOD = 10ns;

    `include "vectors/multiwindow_w16_python_golden.svh"

    localparam int LOGICAL_POINTS_PER_WINDOW = MW_POINTS_PER_WINDOW;
    localparam int BEATS_PER_POINT           = 2;
    localparam int BEATS_PER_WINDOW          =
        LOGICAL_POINTS_PER_WINDOW * BEATS_PER_POINT;
    localparam int TOTAL_BEATS =
        NUM_WINDOWS * BEATS_PER_WINDOW;
    localparam int BYTES_PER_BEAT = 64;

    logic clk;
    logic rst_n;
    logic start;

    logic [63:0] base_addr;
    logic [31:0] logical_points_per_window;

    logic busy;
    logic done;
    logic [$clog2(NUM_WINDOWS)-1:0] window_index;
    logic [DATA_W-1:0] result_x;
    logic [DATA_W-1:0] result_y;
    logic [DATA_W-1:0] result_z;

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

    logic axi_source_busy;
    logic axi_source_done;
    logic [63:0] active_window_base_addr;

    logic [511:0] axi_mem [0:TOTAL_BEATS-1];

    integer error_count;
    integer ar_burst_count;
    integer r_beat_count;
    integer source_done_count;
    integer last_seen_window;
    longint unsigned cycle_count;

    // One-outstanding-burst AXI slave state.
    logic        slave_active;
    logic [63:0] slave_addr;
    logic [8:0]  slave_beats_left;
    logic [7:0]  slave_beat_in_burst;
    logic [31:0] gap_lfsr;

    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cycle_count <= 0;
        else
            cycle_count <= cycle_count + 1;
    end

    msm_axi_multiwindow_top_v2 #(
        .ADDR_W          (ADDR_W),
        .DATA_W          (DATA_W),
        .DEPTH           (1 << ADDR_W),
        .SRAM_RD_LATENCY (1),
        .GEN_W           (16),
        .FIFO_DEPTH      (16),
        .SLOT_COUNT      (16),
        .MIX_CTX_COUNT   (40),
        .MUL_LATENCY     (16),
        .WINDOW_BITS     (WINDOW_BITS),
        .NUM_WINDOWS     (NUM_WINDOWS),
        .AXI_FIFO_DEPTH  (AXI_FIFO_DEPTH),
        .MAX_BURST_BEATS (MAX_BURST_BEATS)
    ) dut (
        .clk                       (clk),
        .rst_n                     (rst_n),
        .start                     (start),
        .base_addr                 (base_addr),
        .logical_points_per_window (logical_points_per_window),
        .busy                      (busy),
        .done                      (done),
        .window_index              (window_index),
        .result_x                  (result_x),
        .result_y                  (result_y),
        .result_z                  (result_z),
        .m_axi_araddr              (m_axi_araddr),
        .m_axi_arlen               (m_axi_arlen),
        .m_axi_arsize              (m_axi_arsize),
        .m_axi_arburst             (m_axi_arburst),
        .m_axi_arvalid             (m_axi_arvalid),
        .m_axi_arready             (m_axi_arready),
        .m_axi_rdata               (m_axi_rdata),
        .m_axi_rresp               (m_axi_rresp),
        .m_axi_rlast               (m_axi_rlast),
        .m_axi_rvalid              (m_axi_rvalid),
        .m_axi_rready              (m_axi_rready),
        .axi_source_busy           (axi_source_busy),
        .axi_source_done           (axi_source_done),
        .active_window_base_addr   (active_window_base_addr)
    );

    // ---------------------------------------------------------------------
    // Construct the AXI memory image from the existing Python-golden vectors.
    // Window w is stored at BASE_ADDR + w*window_stride.
    // X/Y are already Montgomery affine coordinates.
    // ---------------------------------------------------------------------
    initial begin : init_axi_memory
        integer w;
        integer p;
        integer beat;
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
                axi_mem[beat+1][ADDR_W-1:0] = mw_bucket_idx[w][p];
                axi_mem[beat+1][ADDR_W] =
                    (p == LOGICAL_POINTS_PER_WINDOW-1);
            end
        end
    end

    // ---------------------------------------------------------------------
    // AXI slave.
    // - One read burst outstanding at a time.
    // - Deterministic occasional AR/R gaps.
    // - Holds RVALID and RDATA stable while RREADY is low.
    // ---------------------------------------------------------------------
    assign m_axi_arready =
        !slave_active && (gap_lfsr[2:0] != 3'b000);

    always @(posedge clk or negedge rst_n) begin : axi_slave
        integer mem_index;
        if (!rst_n) begin
            slave_active        <= 1'b0;
            slave_addr          <= '0;
            slave_beats_left    <= '0;
            slave_beat_in_burst <= '0;
            m_axi_rdata         <= '0;
            m_axi_rresp         <= 2'b00;
            m_axi_rlast         <= 1'b0;
            m_axi_rvalid        <= 1'b0;
            gap_lfsr            <= 32'h1ACE_B00C;
            ar_burst_count      <= 0;
            r_beat_count        <= 0;
        end else begin
            gap_lfsr <= {
                gap_lfsr[30:0],
                gap_lfsr[31] ^ gap_lfsr[21] ^ gap_lfsr[1] ^ gap_lfsr[0]
            };

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
                    $error("[AXI] unaligned ARADDR %h", m_axi_araddr);
                end
                if ((m_axi_araddr < BASE_ADDR) ||
                    (m_axi_araddr >=
                     BASE_ADDR + TOTAL_BEATS*BYTES_PER_BEAT)) begin
                    error_count++;
                    $error("[AXI] ARADDR outside memory image: %h",
                           m_axi_araddr);
                end

                slave_active        <= 1'b1;
                slave_addr          <= m_axi_araddr;
                slave_beats_left    <= {1'b0, m_axi_arlen} + 9'd1;
                slave_beat_in_burst <= 0;
                ar_burst_count      <= ar_burst_count + 1;
            end

            if (!m_axi_rvalid && slave_active &&
                (gap_lfsr[5:3] != 3'b000)) begin
                mem_index =
                    (slave_addr - BASE_ADDR) >> 6;

                if ((mem_index < 0) || (mem_index >= TOTAL_BEATS)) begin
                    error_count++;
                    $error("[AXI] invalid memory beat index %0d", mem_index);
                    m_axi_rdata <= '0;
                end else begin
                    m_axi_rdata <= axi_mem[mem_index];
                end

                m_axi_rresp  <= 2'b00;
                m_axi_rlast  <= (slave_beats_left == 1);
                m_axi_rvalid <= 1'b1;
            end

            if (m_axi_rvalid && m_axi_rready) begin
                r_beat_count <= r_beat_count + 1;

                if (m_axi_rlast !== (slave_beats_left == 1)) begin
                    error_count++;
                    $error("[AXI] RLAST mismatch, beats_left=%0d rlast=%0b",
                           slave_beats_left, m_axi_rlast);
                end

                m_axi_rvalid <= 1'b0;
                m_axi_rlast  <= 1'b0;

                if (slave_beats_left == 1) begin
                    slave_active     <= 1'b0;
                    slave_beats_left <= 0;
                end else begin
                    slave_addr          <= slave_addr + BYTES_PER_BEAT;
                    slave_beats_left    <= slave_beats_left - 1'b1;
                    slave_beat_in_burst <= slave_beat_in_burst + 1'b1;
                end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            source_done_count <= 0;
            last_seen_window  <= NUM_WINDOWS;
        end else begin
            if (axi_source_done)
                source_done_count <= source_done_count + 1;

            if (busy && (window_index != last_seen_window)) begin
                $display(
                    "[TB_AXI_MW] controller entered window=%0d base=%h cycle=%0d",
                    window_index,
                    active_window_base_addr,
                    cycle_count
                );
                last_seen_window <= window_index;
            end
        end
    end

    initial begin : test_sequence
        error_count = 0;
        rst_n = 1'b0;
        start = 1'b0;
        base_addr = BASE_ADDR;
        logical_points_per_window = LOGICAL_POINTS_PER_WINDOW;

        repeat (8) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        repeat (5) @(posedge clk);
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        wait (done === 1'b1);

        if (result_x !== MW_EXPECTED_X) begin
            error_count++;
            $error("[TB_AXI_MW] result_x mismatch exp=%h got=%h",
                   MW_EXPECTED_X, result_x);
        end

        if (result_y !== MW_EXPECTED_Y) begin
            error_count++;
            $error("[TB_AXI_MW] result_y mismatch exp=%h got=%h",
                   MW_EXPECTED_Y, result_y);
        end

        if (result_z !== MW_EXPECTED_Z) begin
            error_count++;
            $error("[TB_AXI_MW] result_z mismatch exp=%h got=%h",
                   MW_EXPECTED_Z, result_z);
        end

        if (source_done_count != NUM_WINDOWS) begin
            error_count++;
            $error("[TB_AXI_MW] source_done_count expected=%0d got=%0d",
                   NUM_WINDOWS, source_done_count);
        end

        if (r_beat_count != TOTAL_BEATS) begin
            error_count++;
            $error("[TB_AXI_MW] R beat count expected=%0d got=%0d",
                   TOTAL_BEATS, r_beat_count);
        end

        if (error_count == 0) begin
            $display("");
            $display("============================================================");
            $display("[TB_AXI_MW] FULL AXI -> 16-WINDOW PYTHON GOLDEN PASSED");
            $display("[TB_AXI_MW] windows               = %0d", NUM_WINDOWS);
            $display("[TB_AXI_MW] logical points/window = %0d",
                     LOGICAL_POINTS_PER_WINDOW);
            $display("[TB_AXI_MW] total logical points  = %0d",
                     NUM_WINDOWS*LOGICAL_POINTS_PER_WINDOW);
            $display("[TB_AXI_MW] total AXI beats       = %0d", r_beat_count);
            $display("[TB_AXI_MW] AXI bursts            = %0d", ar_burst_count);
            $display("[TB_AXI_MW] source done pulses     = %0d",
                     source_done_count);
            $display("[TB_AXI_MW] done cycle             = %0d", cycle_count);
            $display("[TB_AXI_MW] verified AXI bursts, R gaps, backpressure,");
            $display("[TB_AXI_MW] 2-beat pairing, all windows, SRAM macros,");
            $display("[TB_AXI_MW] and final Python Montgomery X/Y/Z");
            $display("============================================================");
        end else begin
            $fatal(1, "[TB_AXI_MW] FAILED with %0d errors", error_count);
        end

        $finish;
    end

    initial begin : watchdog
        #2s;
        $fatal(1, "[TB_AXI_MW] WATCHDOG TIMEOUT");
    end

endmodule