`timescale 1ns/1ps

module tb_msm_axi_affine_multiwindow_dualclk_16win_python_golden_v1;

    localparam int ADDR_W           = 16;
    localparam int DATA_W           = 256;
    localparam int NUM_WINDOWS      = 16;
    localparam int WINDOW_BITS      = 16;
    localparam int ASYNC_FIFO_DEPTH = 64;
    localparam int MAX_BURST_BEATS  = 16;
    localparam int CONV_FIFO_DEPTH  = 32;

    localparam logic [63:0] BASE_ADDR =
        64'h0000_0000_0100_0000;

    localparam time AXI_CLK_PERIOD = 2ns;
    localparam time MSM_CLK_PERIOD = 4ns;

    `include "vectors/multiwindow_w16_python_golden_affine.svh"

    localparam int LOGICAL_POINTS_PER_WINDOW = MW_POINTS_PER_WINDOW;
    localparam int BEATS_PER_POINT = 2;
    localparam int BEATS_PER_WINDOW =
        LOGICAL_POINTS_PER_WINDOW * BEATS_PER_POINT;
    localparam int TOTAL_BEATS =
        NUM_WINDOWS * BEATS_PER_WINDOW;
    localparam int BYTES_PER_BEAT = 64;
    localparam int BYTES_PER_WINDOW =
        BEATS_PER_WINDOW * BYTES_PER_BEAT;

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
    //   beat 1 = {padding, last, bucket_id}
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
                    $error("[TB_AXI_AFFINE_MW] ARSIZE expected 6, got %0d",
                           m_axi_arsize);
                end

                if (m_axi_arburst !== 2'b01) begin
                    error_count++;
                    $error("[TB_AXI_AFFINE_MW] ARBURST expected INCR, got %0b",
                           m_axi_arburst);
                end

                if (m_axi_araddr[5:0] != 0) begin
                    error_count++;
                    $error("[TB_AXI_AFFINE_MW] unaligned ARADDR=%h",
                           m_axi_araddr);
                end

                if ((m_axi_araddr < BASE_ADDR) ||
                    (m_axi_araddr >=
                     BASE_ADDR + TOTAL_BEATS*BYTES_PER_BEAT)) begin
                    error_count++;
                    $error("[TB_AXI_AFFINE_MW] ARADDR outside image: %h",
                           m_axi_araddr);
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
                    $error("[TB_AXI_AFFINE_MW] invalid beat index=%0d",
                           mem_index);
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
                        "[TB_AXI_AFFINE_MW] RLAST mismatch left=%0d got=%0b",
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
                                "[TB_AXI_AFFINE_MW] invalid next beat index=%0d",
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

    always_ff @(posedge axi_clk or negedge axi_rst_n) begin
        if (!axi_rst_n)
            source_done_count <= 0;
        else if (dut.source_axi_done)
            source_done_count <= source_done_count + 1;
    end

    always_ff @(posedge msm_clk or negedge msm_rst_n) begin
        if (!msm_rst_n) begin
            last_seen_window <= NUM_WINDOWS;
        end else if (busy &&
                     (window_index != last_seen_window)) begin
            $display(
                "[TB_AXI_AFFINE_MW] entered window=%0d base=%h msm_cycle=%0d",
                window_index,
                BASE_ADDR + window_index*BYTES_PER_WINDOW,
                msm_cycle_count
            );
            last_seen_window <= window_index;
        end
    end

    initial begin : test_sequence
        error_count = 0;
        axi_rst_n = 1'b0;
        msm_rst_n = 1'b0;
        start = 1'b0;
        base_addr = BASE_ADDR;
        logical_points_per_window = LOGICAL_POINTS_PER_WINDOW;

        repeat (12) @(posedge axi_clk);
        axi_rst_n = 1'b1;

        repeat (8) @(posedge msm_clk);
        msm_rst_n = 1'b1;

        repeat (5) @(posedge msm_clk);
        @(negedge msm_clk);
        start = 1'b1;
        @(negedge msm_clk);
        start = 1'b0;

        wait (done === 1'b1);
        repeat (8) @(posedge axi_clk);

        if (result_x !== MW_EXPECTED_X) begin
            error_count++;
            $error("[TB_AXI_AFFINE_MW] result_x mismatch exp=%h got=%h",
                   MW_EXPECTED_X, result_x);
        end

        if (result_y !== MW_EXPECTED_Y) begin
            error_count++;
            $error("[TB_AXI_AFFINE_MW] result_y mismatch exp=%h got=%h",
                   MW_EXPECTED_Y, result_y);
        end

        if (result_z !== MW_EXPECTED_Z) begin
            error_count++;
            $error("[TB_AXI_AFFINE_MW] result_z mismatch exp=%h got=%h",
                   MW_EXPECTED_Z, result_z);
        end

        if (converter_busy !== 1'b0) begin
            error_count++;
            $error(
                "[TB_AXI_AFFINE_MW] converter not empty pending=%0d result=%0d",
                converter_pending_count,
                converter_result_count
            );
        end

        if (source_done_count != NUM_WINDOWS) begin
            error_count++;
            $error("[TB_AXI_AFFINE_MW] source done expected=%0d got=%0d",
                   NUM_WINDOWS, source_done_count);
        end

        if (r_beat_count != TOTAL_BEATS) begin
            error_count++;
            $error("[TB_AXI_AFFINE_MW] R beats expected=%0d got=%0d",
                   TOTAL_BEATS, r_beat_count);
        end

        if (ar_burst_count !=
            NUM_WINDOWS *
            ((BEATS_PER_WINDOW + MAX_BURST_BEATS - 1) /
             MAX_BURST_BEATS)) begin
            error_count++;
            $error("[TB_AXI_AFFINE_MW] burst count unexpected got=%0d",
                   ar_burst_count);
        end

        if (error_count == 0) begin
            $display("");
            $display("============================================================");
            $display(
                "[TB_AXI_AFFINE_MW] FULL AXI AFFINE -> MONTGOMERY -> MSM PASSED"
            );
            $display("[TB_AXI_AFFINE_MW] windows               = %0d",
                     NUM_WINDOWS);
            $display("[TB_AXI_AFFINE_MW] points/window         = %0d",
                     LOGICAL_POINTS_PER_WINDOW);
            $display("[TB_AXI_AFFINE_MW] total affine points   = %0d",
                     NUM_WINDOWS*LOGICAL_POINTS_PER_WINDOW);
            $display("[TB_AXI_AFFINE_MW] total AXI beats       = %0d",
                     r_beat_count);
            $display("[TB_AXI_AFFINE_MW] AXI bursts            = %0d",
                     ar_burst_count);
            $display("[TB_AXI_AFFINE_MW] source done pulses    = %0d",
                     source_done_count);
            $display("[TB_AXI_AFFINE_MW] done MSM cycle        = %0d",
                     msm_cycle_count);
            $display(
                "[TB_AXI_AFFINE_MW] verified dual clocks, CDC, AXI gaps,"
            );
            $display(
                "[TB_AXI_AFFINE_MW] normal affine DRAM input, Montgomery conversion,"
            );
            $display(
                "[TB_AXI_AFFINE_MW] 16 windows, SRAM macros and exact golden X/Y/Z"
            );
            $display("============================================================");
        end else begin
            $fatal(1, "[TB_AXI_AFFINE_MW] FAILED with %0d errors",
                   error_count);
        end

        #20;
        $finish;
    end

    initial begin
        #(64'd2000000000);
        $fatal(1, "[TB_AXI_AFFINE_MW] Watchdog timeout.");
    end

endmodule