`timescale 1ns/1ps

// ============================================================================
// File: tb/frontend/tb_axi_point_stream_source_dualclk_multitransfer_backpressure_v1.sv
//
// New coverage beyond tb_axi_point_stream_source_dualclk_4kb_v1:
//   * Non-integer asynchronous clock ratio.
//   * Three back-to-back logical transfers without reset.
//   * Different transfer lengths and 4KB boundary positions.
//   * Randomized AXI ARREADY and R-channel gaps.
//   * Randomized MSM point_ready plus deliberate long stalls.
//   * Small FIFO depth to force real backpressure.
//   * Exact ordering / no-loss / no-duplication checks.
// ============================================================================

module tb_axi_point_stream_source_dualclk_multitransfer_backpressure_v1;

    localparam int GLOBAL_ADDR_W     = 16;
    localparam int DATA_W            = 256;
    localparam int ASYNC_FIFO_DEPTH  = 16;
    localparam int MAX_BURST_BEATS   = 32;
    localparam int BYTES_PER_BEAT    = 64;

    localparam int NUM_TRANSFERS = 3;
    localparam int MAX_POINTS    = 173;
    localparam int MAX_BEATS     = 2 * MAX_POINTS;

    // Intentionally asynchronous, non-integer clock ratio.
    localparam time AXI_CLK_PERIOD = 3ns;
    localparam time MSM_CLK_PERIOD = 7ns;

    logic axi_clk, axi_rst_n;
    logic msm_clk, msm_rst_n;

    logic        start_axi;
    logic [63:0] base_addr;
    logic [31:0] logical_point_count;
    logic        axi_busy;
    logic        axi_done;

    logic [63:0]  m_axi_araddr;
    logic [7:0]   m_axi_arlen;
    logic [2:0]   m_axi_arsize;
    logic [1:0]   m_axi_arburst;
    logic         m_axi_arvalid;
    logic         m_axi_arready;

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

    logic [511:0] axi_mem [0:MAX_BEATS-1];

    int unsigned transfer_points [0:NUM_TRANSFERS-1];
    logic [63:0] transfer_base [0:NUM_TRANSFERS-1];

    integer error_count;
    integer active_transfer;
    integer expected_index;
    integer accepted_in_transfer;
    integer total_accepted;
    integer total_expected;
    integer ar_count_total;
    integer r_count_total;
    integer output_stall_cycles;
    integer axi_r_gap_cycles;
    integer max_output_stall_run;
    integer output_stall_run;
    integer fifo_backpressure_cycles;

    logic        slave_active_q;
    logic [63:0] slave_addr_q;
    logic [8:0]  slave_beats_left_q;
    integer      slave_mem_index_q;

    logic [31:0] axi_lfsr_q;
    logic [31:0] msm_lfsr_q;
    integer msm_cycle_q;

    function automatic logic [DATA_W-1:0] mk_x(
        input int unsigned txn,
        input int unsigned idx
    );
        logic [DATA_W-1:0] v;
        begin
            v = '0;
            v[31:0]    = 32'h1000_0000 + (txn << 16) + idx;
            v[95:64]   = 32'h1357_0000 ^ (txn * 32'h1021) ^ idx;
            v[255:224] = 32'hA5A5_0000 | txn[15:0];
            return v;
        end
    endfunction

    function automatic logic [DATA_W-1:0] mk_y(
        input int unsigned txn,
        input int unsigned idx
    );
        logic [DATA_W-1:0] v;
        begin
            v = '0;
            v[31:0]    = 32'h2000_0000 + (txn << 16) + idx;
            v[159:128] = 32'h2468_0000 ^ (txn * 32'h1F3D) ^ (idx * 3);
            v[255:224] = 32'h5A5A_0000 | txn[15:0];
            return v;
        end
    endfunction

    function automatic logic [GLOBAL_ADDR_W-1:0] mk_bucket(
        input int unsigned txn,
        input int unsigned idx
    );
        int unsigned value;
        begin
            value = ((idx * 251) + (txn * 4093) + 17) % 65535;
            return GLOBAL_ADDR_W'(value + 1);
        end
    endfunction

    task automatic load_transfer_memory(input int unsigned txn);
        integer i;
        begin
            for (i = 0; i < transfer_points[txn]; i = i + 1) begin
                axi_mem[2*i] = {mk_y(txn, i), mk_x(txn, i)};
                axi_mem[2*i+1] = '0;
                axi_mem[2*i+1][GLOBAL_ADDR_W-1:0] = mk_bucket(txn, i);
                axi_mem[2*i+1][GLOBAL_ADDR_W] =
                    (i == transfer_points[txn]-1);
            end
        end
    endtask

    task automatic start_transfer(input int unsigned txn);
        begin
            wait (!axi_busy);
            load_transfer_memory(txn);

            active_transfer      = txn;
            expected_index       = 0;
            accepted_in_transfer = 0;
            base_addr            = transfer_base[txn];
            logical_point_count  = transfer_points[txn];

            repeat (3) @(posedge axi_clk);
            @(negedge axi_clk);
            start_axi = 1'b1;
            @(negedge axi_clk);
            start_axi = 1'b0;

            $display(
                "[TB_STRESS] transfer %0d started: base=%h points=%0d",
                txn, transfer_base[txn], transfer_points[txn]
            );

            wait (accepted_in_transfer == transfer_points[txn]);
            wait (!axi_busy);
            repeat (5) @(posedge msm_clk);

            $display(
                "[TB_STRESS] transfer %0d completed: accepted=%0d",
                txn, accepted_in_transfer
            );
        end
    endtask

    initial axi_clk = 1'b0;
    always #(AXI_CLK_PERIOD/2.0) axi_clk = ~axi_clk;

    initial msm_clk = 1'b0;
    always #(MSM_CLK_PERIOD/2.0) msm_clk = ~msm_clk;

    axi_point_stream_source_dualclk_v2 #(
        .GLOBAL_ADDR_W     (GLOBAL_ADDR_W),
        .DATA_W            (DATA_W),
        .ASYNC_FIFO_DEPTH  (ASYNC_FIFO_DEPTH),
        .MAX_BURST_BEATS   (MAX_BURST_BEATS)
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

    // AXI-side pseudo-random sequence.
    always_ff @(posedge axi_clk or negedge axi_rst_n) begin
        if (!axi_rst_n)
            axi_lfsr_q <= 32'h1ACE_B00C;
        else
            axi_lfsr_q <= {axi_lfsr_q[30:0],
                           axi_lfsr_q[31] ^ axi_lfsr_q[21] ^
                           axi_lfsr_q[1]  ^ axi_lfsr_q[0]};
    end

    // AR channel receives random wait states while no burst is active.
    assign m_axi_arready =
        !slave_active_q && (axi_lfsr_q[2:0] != 3'b000);

    // AXI slave with random gaps between R beats.
    always @(posedge axi_clk or negedge axi_rst_n) begin : axi_slave
        integer burst_bytes;
        logic [12:0] low_sum;
        logic send_next_beat;

        if (!axi_rst_n) begin
            slave_active_q     <= 1'b0;
            slave_addr_q       <= '0;
            slave_beats_left_q <= '0;
            slave_mem_index_q  <= 0;
            m_axi_rdata        <= '0;
            m_axi_rresp        <= 2'b00;
            m_axi_rlast        <= 1'b0;
            m_axi_rvalid       <= 1'b0;
            ar_count_total     <= 0;
            r_count_total      <= 0;
            axi_r_gap_cycles   <= 0;
        end else begin
            send_next_beat = (axi_lfsr_q[5:3] != 3'b000);

            if (m_axi_arvalid && m_axi_arready) begin
                burst_bytes =
                    ({1'b0, m_axi_arlen} + 9'd1) * BYTES_PER_BEAT;
                low_sum =
                    {1'b0, m_axi_araddr[11:0]} + burst_bytes;

                if (low_sum > 13'd4096) begin
                    error_count++;
                    $error(
                        "[TB_STRESS] burst crosses 4KB: addr=%h beats=%0d",
                        m_axi_araddr, m_axi_arlen + 1
                    );
                end

                if (m_axi_arsize !== 3'b110) begin
                    error_count++;
                    $error("[TB_STRESS] ARSIZE mismatch");
                end

                if (m_axi_arburst !== 2'b01) begin
                    error_count++;
                    $error("[TB_STRESS] ARBURST mismatch");
                end

                if (m_axi_araddr[5:0] != 0) begin
                    error_count++;
                    $error("[TB_STRESS] ARADDR is not 64-byte aligned");
                end

                slave_active_q     <= 1'b1;
                slave_addr_q       <= m_axi_araddr;
                slave_beats_left_q <= {1'b0, m_axi_arlen} + 9'd1;
                slave_mem_index_q  <=
                    (m_axi_araddr - transfer_base[active_transfer]) >> 6;
                m_axi_rvalid       <= 1'b0;
                m_axi_rlast        <= 1'b0;
                ar_count_total     <= ar_count_total + 1;
            end

            if (slave_active_q) begin
                if (!m_axi_rvalid) begin
                    if (send_next_beat) begin
                        m_axi_rdata  <= axi_mem[slave_mem_index_q];
                        m_axi_rresp  <= 2'b00;
                        m_axi_rlast  <= (slave_beats_left_q == 1);
                        m_axi_rvalid <= 1'b1;
                    end else begin
                        axi_r_gap_cycles <= axi_r_gap_cycles + 1;
                    end
                end else if (m_axi_rvalid && m_axi_rready) begin
                    r_count_total <= r_count_total + 1;

                    if (slave_beats_left_q == 1) begin
                        slave_active_q     <= 1'b0;
                        slave_beats_left_q <= '0;
                        m_axi_rvalid       <= 1'b0;
                        m_axi_rlast        <= 1'b0;
                    end else begin
                        slave_addr_q       <= slave_addr_q + BYTES_PER_BEAT;
                        slave_beats_left_q <= slave_beats_left_q - 1'b1;
                        slave_mem_index_q  <= slave_mem_index_q + 1;
                        m_axi_rvalid       <= 1'b0;
                        m_axi_rlast        <= 1'b0;
                    end
                end
            end
        end
    end

    // MSM-side pseudo-random backpressure plus deterministic long stalls.
    always_ff @(posedge msm_clk or negedge msm_rst_n) begin
        if (!msm_rst_n) begin
            msm_lfsr_q  <= 32'hC001_D00D;
            msm_cycle_q <= 0;
        end else begin
            msm_lfsr_q <= {msm_lfsr_q[30:0],
                           msm_lfsr_q[31] ^ msm_lfsr_q[28] ^
                           msm_lfsr_q[3]  ^ msm_lfsr_q[0]};
            msm_cycle_q <= msm_cycle_q + 1;
        end
    end

    always_comb begin
        // A deliberate 36-cycle stop every 128 MSM cycles.
        if ((msm_cycle_q[6:0] >= 7'd80) &&
            (msm_cycle_q[6:0] <  7'd116))
            point_ready = 1'b0;
        else
            point_ready = (msm_lfsr_q[3:0] != 4'h0);
    end

    always @(posedge msm_clk or negedge msm_rst_n) begin
        if (!msm_rst_n) begin
            expected_index          <= 0;
            accepted_in_transfer    <= 0;
            total_accepted          <= 0;
            output_stall_cycles     <= 0;
            max_output_stall_run    <= 0;
            output_stall_run        <= 0;
        end else begin
            if (point_valid && !point_ready) begin
                output_stall_cycles <= output_stall_cycles + 1;
                output_stall_run    <= output_stall_run + 1;
                if (output_stall_run + 1 > max_output_stall_run)
                    max_output_stall_run <= output_stall_run + 1;
            end else begin
                output_stall_run <= 0;
            end

            if (point_valid && point_ready) begin
                if (point_x !== mk_x(active_transfer, expected_index)) begin
                    error_count++;
                    $error(
                        "[TB_STRESS] X mismatch txn=%0d idx=%0d",
                        active_transfer, expected_index
                    );
                end

                if (point_y !== mk_y(active_transfer, expected_index)) begin
                    error_count++;
                    $error(
                        "[TB_STRESS] Y mismatch txn=%0d idx=%0d",
                        active_transfer, expected_index
                    );
                end

                if (point_bucket_id !==
                    mk_bucket(active_transfer, expected_index)) begin
                    error_count++;
                    $error(
                        "[TB_STRESS] bucket mismatch txn=%0d idx=%0d",
                        active_transfer, expected_index
                    );
                end

                if (point_last !==
                    (expected_index ==
                     transfer_points[active_transfer]-1)) begin
                    error_count++;
                    $error(
                        "[TB_STRESS] last mismatch txn=%0d idx=%0d got=%0b",
                        active_transfer, expected_index, point_last
                    );
                end

                expected_index       <= expected_index + 1;
                accepted_in_transfer <= accepted_in_transfer + 1;
                total_accepted       <= total_accepted + 1;
            end
        end
    end

    // Count cycles where AXI data is blocked by a full/pressured CDC path.
    always_ff @(posedge axi_clk or negedge axi_rst_n) begin
        if (!axi_rst_n)
            fifo_backpressure_cycles <= 0;
        else if (m_axi_rvalid && !m_axi_rready)
            fifo_backpressure_cycles <= fifo_backpressure_cycles + 1;
    end

    initial begin : test_sequence
        integer txn;

        transfer_points[0] = 97;
        transfer_points[1] = 173;
        transfer_points[2] = 65;

        // 1 beat, 7 beats and 32 beats before the next 4KB boundary.
        // This also checks a point frame split across an AXI burst boundary.
        transfer_base[0] = 64'h0000_0000_0300_0FC0;
        transfer_base[1] = 64'h0000_0000_0310_0E40;
        transfer_base[2] = 64'h0000_0000_0320_0800;

        total_expected =
            transfer_points[0] +
            transfer_points[1] +
            transfer_points[2];

        error_count             = 0;
        active_transfer         = 0;
        expected_index          = 0;
        accepted_in_transfer    = 0;
        total_accepted          = 0;
        start_axi               = 1'b0;
        base_addr               = '0;
        logical_point_count     = '0;
        axi_rst_n               = 1'b0;
        msm_rst_n               = 1'b0;

        repeat (10) @(posedge axi_clk);
        axi_rst_n = 1'b1;

        repeat (7) @(posedge msm_clk);
        msm_rst_n = 1'b1;

        for (txn = 0; txn < NUM_TRANSFERS; txn = txn + 1)
            start_transfer(txn);

        repeat (12) @(posedge msm_clk);

        if (total_accepted != total_expected) begin
            error_count++;
            $error(
                "[TB_STRESS] accepted mismatch expected=%0d actual=%0d",
                total_expected, total_accepted
            );
        end

        if (r_count_total != 2 * total_expected) begin
            error_count++;
            $error(
                "[TB_STRESS] beat mismatch expected=%0d actual=%0d",
                2 * total_expected, r_count_total
            );
        end

        if (output_stall_cycles == 0) begin
            error_count++;
            $error("[TB_STRESS] no MSM backpressure was exercised");
        end

        if (fifo_backpressure_cycles == 0) begin
            error_count++;
            $error("[TB_STRESS] CDC FIFO never backpressured AXI R");
        end

        if (axi_r_gap_cycles == 0) begin
            error_count++;
            $error("[TB_STRESS] AXI R-channel gaps were not exercised");
        end

        if (error_count == 0) begin
            $display("");
            $display("============================================================");
            $display("[TB_STRESS] AXI DUAL-CLOCK STRESS PASSED");
            $display("[TB_STRESS] transfers                 = %0d",
                     NUM_TRANSFERS);
            $display("[TB_STRESS] logical points            = %0d",
                     total_accepted);
            $display("[TB_STRESS] total AXI beats           = %0d",
                     r_count_total);
            $display("[TB_STRESS] total AXI bursts          = %0d",
                     ar_count_total);
            $display("[TB_STRESS] AXI R gap cycles          = %0d",
                     axi_r_gap_cycles);
            $display("[TB_STRESS] MSM output stall cycles   = %0d",
                     output_stall_cycles);
            $display("[TB_STRESS] longest MSM stall run     = %0d",
                     max_output_stall_run);
            $display("[TB_STRESS] AXI FIFO backpressure     = %0d",
                     fifo_backpressure_cycles);
            $display("[TB_STRESS] verified: 4KB, CDC, ordering,");
            $display("[TB_STRESS] no loss/duplication, multi-transfer");
            $display("============================================================");
        end else begin
            $fatal(1, "[TB_STRESS] FAILED with %0d errors", error_count);
        end

        $finish;
    end

    initial begin : watchdog
        #5ms;
        $fatal(1, "[TB_STRESS] WATCHDOG TIMEOUT");
    end

endmodule