`timescale 1ns/1ps

module tb_msm_axi_affine_multiwindow_dualclk_16win_65536p_uniform_stageprofile_v3_heartbeat_fixed;

    localparam int ADDR_W           = 16;
    localparam int DATA_W           = 256;
    localparam int NUM_WINDOWS      = 16;
    localparam int WINDOW_BITS      = 16;
    localparam int ASYNC_FIFO_DEPTH = 64;
    localparam int MAX_BURST_BEATS  = 16;
    localparam int CONV_FIFO_DEPTH  = 32;
    localparam int NUM_LANES        = 8;

    localparam logic [63:0] BASE_ADDR =
        64'h0000_0000_0200_0000;

    localparam time AXI_CLK_PERIOD = 2ns;
    localparam time MSM_CLK_PERIOD = 4ns;

    `include "vectors/full_axi_affine_65536p_uniform_stageprofile_v1_expected.svh"

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
    integer active_axi_window;
    integer active_msm_window;
    integer csv_fd;

    longint unsigned axi_cycle_count;
    longint unsigned msm_cycle_count;

    localparam longint unsigned HEARTBEAT_MSM_CYCLES = 1_000_000;
    longint unsigned next_heartbeat_cycle;
    longint unsigned heartbeat_count;

    logic        slave_active_q;
    logic [63:0] slave_addr_q;
    logic [8:0]  slave_beats_left_q;
    logic [31:0] gap_lfsr_q;

    longint unsigned win_enter_cycle        [0:NUM_WINDOWS-1];
    longint unsigned win_start_cycle        [0:NUM_WINDOWS-1];
    longint unsigned win_done_cycle         [0:NUM_WINDOWS-1];
    longint unsigned next_window_cycle      [0:NUM_WINDOWS-1];

    longint unsigned axi_start_cycle        [0:NUM_WINDOWS-1];
    longint unsigned axi_first_ar_cycle     [0:NUM_WINDOWS-1];
    longint unsigned axi_first_r_cycle      [0:NUM_WINDOWS-1];
    longint unsigned axi_last_r_cycle       [0:NUM_WINDOWS-1];
    longint unsigned axi_done_cycle         [0:NUM_WINDOWS-1];

    longint unsigned conv_first_in_cycle    [0:NUM_WINDOWS-1];
    longint unsigned conv_last_in_cycle     [0:NUM_WINDOWS-1];
    longint unsigned conv_first_out_cycle   [0:NUM_WINDOWS-1];
    longint unsigned conv_last_out_cycle    [0:NUM_WINDOWS-1];

    longint unsigned build_first_cycle      [0:NUM_WINDOWS-1];
    longint unsigned build_last_cycle       [0:NUM_WINDOWS-1];
    longint unsigned build_done_cycle       [0:NUM_WINDOWS-1];
    longint unsigned reduce_start_cycle     [0:NUM_WINDOWS-1];
    longint unsigned reduce_done_cycle      [0:NUM_WINDOWS-1];

    longint unsigned ar_count_win           [0:NUM_WINDOWS-1];
    longint unsigned r_count_win            [0:NUM_WINDOWS-1];
    longint unsigned point_in_count_win     [0:NUM_WINDOWS-1];
    longint unsigned mont_out_count_win     [0:NUM_WINDOWS-1];
    longint unsigned zero_count_win         [0:NUM_WINDOWS-1];

    longint unsigned lane_points_win        [0:NUM_WINDOWS-1][0:NUM_LANES-1];
    longint unsigned lane_fifo_max_win      [0:NUM_WINDOWS-1][0:NUM_LANES-1];

    longint unsigned enqueue_base           [0:NUM_WINDOWS-1];
    longint unsigned completed_base         [0:NUM_WINDOWS-1];
    longint unsigned stall_base             [0:NUM_WINDOWS-1];
    longint unsigned direct_base            [0:NUM_WINDOWS-1];
    longint unsigned mixed_base             [0:NUM_WINDOWS-1];

    longint unsigned enqueue_delta          [0:NUM_WINDOWS-1];
    longint unsigned completed_delta        [0:NUM_WINDOWS-1];
    longint unsigned stall_delta            [0:NUM_WINDOWS-1];
    longint unsigned direct_delta           [0:NUM_WINDOWS-1];
    longint unsigned mixed_delta            [0:NUM_WINDOWS-1];

    longint unsigned double_count_win       [0:NUM_WINDOWS-1];
    longint unsigned add_count_win          [0:NUM_WINDOWS-1];

    bit seen_axi_ar                         [0:NUM_WINDOWS-1];
    bit seen_axi_r                          [0:NUM_WINDOWS-1];
    bit seen_conv_in                        [0:NUM_WINDOWS-1];
    bit seen_conv_out                       [0:NUM_WINDOWS-1];
    bit seen_build                          [0:NUM_WINDOWS-1];
    bit seen_build_done                     [0:NUM_WINDOWS-1];
    bit seen_reduce_start                   [0:NUM_WINDOWS-1];
    bit seen_reduce_done                    [0:NUM_WINDOWS-1];

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

    initial begin : init_axi_memory
        integer beat;
        for (beat = 0; beat < TOTAL_BEATS; beat = beat + 1)
            axi_mem[beat] = '0;

        $readmemh(
            "vectors/full_axi_affine_65536p_uniform_stageprofile_v1.memh",
            axi_mem
        );
    end

    assign m_axi_arready =
        !slave_active_q && (gap_lfsr_q[2:0] != 3'b000);

    always @(posedge axi_clk or negedge axi_rst_n) begin : axi_slave
        integer mem_index;
        integer w;

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
            source_done_count  <= 0;
            active_axi_window  <= NUM_WINDOWS-1;
        end else begin
            gap_lfsr_q <= {
                gap_lfsr_q[30:0],
                gap_lfsr_q[31] ^
                gap_lfsr_q[21] ^
                gap_lfsr_q[1]  ^
                gap_lfsr_q[0]
            };

            if (dut.source_start_axi_q) begin
                active_axi_window <= dut.launch_window_axi_sync2_q;
                axi_start_cycle[dut.launch_window_axi_sync2_q] <=
                    axi_cycle_count;
            end

            if (m_axi_arvalid && m_axi_arready) begin
                w = active_axi_window;

                if (!seen_axi_ar[w]) begin
                    axi_first_ar_cycle[w] <= axi_cycle_count;
                    seen_axi_ar[w] <= 1'b1;
                end

                ar_count_win[w] <= ar_count_win[w] + 1;
                ar_burst_count <= ar_burst_count + 1;

                if (m_axi_arsize !== 3'b110) begin
                    error_count++;
                    $error("[TB_STAGE65536_HB] bad ARSIZE=%0d", m_axi_arsize);
                end

                if (m_axi_arburst !== 2'b01) begin
                    error_count++;
                    $error("[TB_STAGE65536_HB] bad ARBURST=%0b", m_axi_arburst);
                end

                if (m_axi_araddr[5:0] != 0) begin
                    error_count++;
                    $error("[TB_STAGE65536_HB] unaligned ARADDR=%h", m_axi_araddr);
                end

                slave_active_q     <= 1'b1;
                slave_addr_q       <= m_axi_araddr;
                slave_beats_left_q <= {1'b0, m_axi_arlen} + 9'd1;
            end

            if (!m_axi_rvalid &&
                slave_active_q &&
                (gap_lfsr_q[5:3] != 3'b000)) begin

                mem_index = (slave_addr_q - BASE_ADDR) >> 6;

                if ((mem_index < 0) || (mem_index >= TOTAL_BEATS)) begin
                    error_count++;
                    $error("[TB_STAGE65536_HB] invalid beat index=%0d", mem_index);
                    m_axi_rdata <= '0;
                end else begin
                    m_axi_rdata <= axi_mem[mem_index];
                end

                m_axi_rresp  <= 2'b00;
                m_axi_rlast  <= (slave_beats_left_q == 1);
                m_axi_rvalid <= 1'b1;
            end

            if (m_axi_rvalid && m_axi_rready) begin
                w = active_axi_window;

                if (!seen_axi_r[w]) begin
                    axi_first_r_cycle[w] <= axi_cycle_count;
                    seen_axi_r[w] <= 1'b1;
                end

                axi_last_r_cycle[w] <= axi_cycle_count;
                r_count_win[w] <= r_count_win[w] + 1;
                r_beat_count <= r_beat_count + 1;

                if (m_axi_rlast !== (slave_beats_left_q == 1)) begin
                    error_count++;
                    $error("[TB_STAGE65536_HB] RLAST mismatch");
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
                            $error("[TB_STAGE65536_HB] invalid next beat=%0d",
                                   mem_index);
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

            if (dut.source_axi_done) begin
                axi_done_cycle[active_axi_window] <= axi_cycle_count;
                source_done_count <= source_done_count + 1;
            end
        end
    end

    always @(posedge msm_clk or negedge msm_rst_n) begin : msm_profile
        integer w;
        integer lane;
        longint unsigned occ;

        if (!msm_rst_n) begin
            last_seen_window    <= NUM_WINDOWS;
            active_msm_window   <= NUM_WINDOWS-1;
            next_heartbeat_cycle <= HEARTBEAT_MSM_CYCLES;
            heartbeat_count      <= 0;
        end else begin
            if (busy && (msm_cycle_count >= next_heartbeat_cycle)) begin
                heartbeat_count <= heartbeat_count + 1;
                next_heartbeat_cycle <=
                    next_heartbeat_cycle + HEARTBEAT_MSM_CYCLES;

                $display(
                    "[TB_STAGE65536_HB] HEARTBEAT=%0d msm_cycle=%0d window=%0d point_in=%0d mont_out=%0d build_fire=%0d completed=%0d fifo_stall=%0d build_done=%0b reduce_start=%0b reduce_done=%0b",
                    heartbeat_count + 1,
                    msm_cycle_count,
                    active_msm_window,
                    point_in_count_win[active_msm_window],
                    mont_out_count_win[active_msm_window],
                    dut.u_controller.u_window.
                        scheduler_total_enqueue_count -
                        enqueue_base[active_msm_window],
                    dut.u_controller.u_window.
                        scheduler_total_completed_count -
                        completed_base[active_msm_window],
                    dut.u_controller.u_window.
                        scheduler_total_fifo_full_stall_count -
                        stall_base[active_msm_window],
                    seen_build_done[active_msm_window],
                    seen_reduce_start[active_msm_window],
                    seen_reduce_done[active_msm_window]
                );
            end

            if (busy && (window_index != last_seen_window)) begin
                w = window_index;
                active_msm_window <= w;
                win_enter_cycle[w] <= msm_cycle_count;
                last_seen_window <= w;

                $display(
                    "[TB_STAGE65536_HB] entered window=%0d base=%h msm_cycle=%0d",
                    w,
                    BASE_ADDR + w*BYTES_PER_WINDOW,
                    msm_cycle_count
                );

                $display(
                    "[TB_STAGE65536_HB] heartbeat interval=%0d MSM cycles",
                    HEARTBEAT_MSM_CYCLES
                );
            end

            if (dut.u_controller.window_start) begin
                w = window_index;
                win_start_cycle[w] <= msm_cycle_count;

                enqueue_base[w] <=
                    dut.u_controller.u_window.scheduler_total_enqueue_count;
                completed_base[w] <=
                    dut.u_controller.u_window.scheduler_total_completed_count;
                stall_base[w] <=
                    dut.u_controller.u_window.
                        scheduler_total_fifo_full_stall_count;
                direct_base[w] <=
                    dut.u_controller.u_window.
                        scheduler_total_direct_write_count;
                mixed_base[w] <=
                    dut.u_controller.u_window.
                        scheduler_total_mixed_add_count;

                if (w < NUM_WINDOWS-1)
                    next_window_cycle[w+1] <= msm_cycle_count;
            end

            if (dut.point_valid && dut.point_ready) begin
                w = active_msm_window;
                if (!seen_conv_in[w]) begin
                    conv_first_in_cycle[w] <= msm_cycle_count;
                    seen_conv_in[w] <= 1'b1;
                end
                conv_last_in_cycle[w] <= msm_cycle_count;
                point_in_count_win[w] <= point_in_count_win[w] + 1;
            end

            if (dut.mont_valid && dut.mont_ready) begin
                w = active_msm_window;
                if (!seen_conv_out[w]) begin
                    conv_first_out_cycle[w] <= msm_cycle_count;
                    seen_conv_out[w] <= 1'b1;
                end
                conv_last_out_cycle[w] <= msm_cycle_count;
                mont_out_count_win[w] <= mont_out_count_win[w] + 1;

                if (dut.mont_bucket_id == 0)
                    zero_count_win[w] <= zero_count_win[w] + 1;
                else begin
                    lane = dut.mont_bucket_id[2:0];
                    lane_points_win[w][lane] <=
                        lane_points_win[w][lane] + 1;
                end
            end

            if (dut.u_controller.u_window.build_input_fire) begin
                w = active_msm_window;
                if (!seen_build[w]) begin
                    build_first_cycle[w] <= msm_cycle_count;
                    seen_build[w] <= 1'b1;
                end
                build_last_cycle[w] <= msm_cycle_count;
            end

            if (dut.u_controller.u_window.build_done &&
                !seen_build_done[active_msm_window]) begin
                build_done_cycle[active_msm_window] <= msm_cycle_count;
                seen_build_done[active_msm_window] <= 1'b1;
            end

            if (dut.u_controller.u_window.reduce_start &&
                !seen_reduce_start[active_msm_window]) begin
                reduce_start_cycle[active_msm_window] <= msm_cycle_count;
                seen_reduce_start[active_msm_window] <= 1'b1;
            end

            if (dut.u_controller.u_window.reduce_done &&
                !seen_reduce_done[active_msm_window]) begin
                reduce_done_cycle[active_msm_window] <= msm_cycle_count;
                seen_reduce_done[active_msm_window] <= 1'b1;
            end

            if (dut.u_controller.window_done) begin
                w = active_msm_window;
                win_done_cycle[w] <= msm_cycle_count;

                enqueue_delta[w] <=
                    dut.u_controller.u_window.scheduler_total_enqueue_count -
                    enqueue_base[w];

                completed_delta[w] <=
                    dut.u_controller.u_window.scheduler_total_completed_count -
                    completed_base[w];

                stall_delta[w] <=
                    dut.u_controller.u_window.
                        scheduler_total_fifo_full_stall_count -
                    stall_base[w];

                direct_delta[w] <=
                    dut.u_controller.u_window.
                        scheduler_total_direct_write_count -
                    direct_base[w];

                mixed_delta[w] <=
                    dut.u_controller.u_window.
                        scheduler_total_mixed_add_count -
                    mixed_base[w];
            end

            if (dut.u_controller.double_done)
                double_count_win[active_msm_window] <=
                    double_count_win[active_msm_window] + 1;

            if (dut.u_controller.add_done)
                add_count_win[active_msm_window] <=
                    add_count_win[active_msm_window] + 1;

            for (lane = 0; lane < NUM_LANES; lane = lane + 1) begin
                occ = dut.u_controller.u_window.
                    scheduler_lane_fifo_occupancy[lane];

                if (occ > lane_fifo_max_win[active_msm_window][lane])
                    lane_fifo_max_win[active_msm_window][lane] <= occ;
            end
        end
    end

    task automatic write_profile_csv;
        integer w;
        integer lane;
        longint unsigned axi_fetch_cycles;
        longint unsigned convert_cycles;
        longint unsigned build_accept_cycles;
        longint unsigned build_drain_cycles;
        longint unsigned reduce_cycles;
        longint unsigned total_window_cycles;
        longint unsigned recombine_cycles;
        begin
            csv_fd = $fopen(
                "runs/full_axi_affine_65536p_uniform_stageprofile_v3_heartbeat_fixed.csv",
                "w"
            );

            if (csv_fd == 0)
                $fatal(1, "[TB_STAGE65536_HB] could not open CSV");

            $fdisplay(
                csv_fd,
                "window,points,axi_fetch_cycles,convert_span_cycles,build_accept_span_cycles,build_drain_cycles,reduce_cycles,window_total_cycles,recombine_to_next_window_cycles,enqueue_count,completed_count,fifo_full_stalls,direct_writes,mixed_adds,zero_digits,double_count,add_count,lane0_points,lane1_points,lane2_points,lane3_points,lane4_points,lane5_points,lane6_points,lane7_points,lane0_fifo_max,lane1_fifo_max,lane2_fifo_max,lane3_fifo_max,lane4_fifo_max,lane5_fifo_max,lane6_fifo_max,lane7_fifo_max"
            );

            for (w = NUM_WINDOWS-1; w >= 0; w = w - 1) begin
                axi_fetch_cycles =
                    axi_done_cycle[w] - axi_start_cycle[w];

                convert_cycles =
                    conv_last_out_cycle[w] - conv_first_in_cycle[w] + 1;

                build_accept_cycles =
                    build_last_cycle[w] - build_first_cycle[w] + 1;

                build_drain_cycles =
                    build_done_cycle[w] - build_last_cycle[w];

                reduce_cycles =
                    reduce_done_cycle[w] - reduce_start_cycle[w] + 1;

                total_window_cycles =
                    win_done_cycle[w] - win_start_cycle[w] + 1;

                if (w > 0)
                    recombine_cycles =
                        next_window_cycle[w] - win_done_cycle[w];
                else
                    recombine_cycles =
                        msm_cycle_count - win_done_cycle[w];

                $fwrite(
                    csv_fd,
                    "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,",
                    w,
                    LOGICAL_POINTS_PER_WINDOW,
                    axi_fetch_cycles,
                    convert_cycles,
                    build_accept_cycles,
                    build_drain_cycles,
                    reduce_cycles,
                    total_window_cycles,
                    recombine_cycles,
                    enqueue_delta[w],
                    completed_delta[w],
                    stall_delta[w],
                    direct_delta[w],
                    mixed_delta[w],
                    zero_count_win[w],
                    double_count_win[w],
                    add_count_win[w]
                );

                for (lane = 0; lane < NUM_LANES; lane = lane + 1)
                    $fwrite(csv_fd, "%0d,", lane_points_win[w][lane]);

                for (lane = 0; lane < NUM_LANES; lane = lane + 1) begin
                    if (lane == NUM_LANES-1)
                        $fwrite(csv_fd, "%0d\n",
                                lane_fifo_max_win[w][lane]);
                    else
                        $fwrite(csv_fd, "%0d,",
                                lane_fifo_max_win[w][lane]);
                end
            end

            $fclose(csv_fd);
        end
    endtask

    task automatic check_profile;
        integer w;
        integer lane;
        begin
            for (w = 0; w < NUM_WINDOWS; w = w + 1) begin
                if (point_in_count_win[w] != LOGICAL_POINTS_PER_WINDOW) begin
                    error_count++;
                    $error("[TB_STAGE65536_HB] w=%0d input exp=%0d got=%0d",
                           w, LOGICAL_POINTS_PER_WINDOW,
                           point_in_count_win[w]);
                end

                if (mont_out_count_win[w] != LOGICAL_POINTS_PER_WINDOW) begin
                    error_count++;
                    $error("[TB_STAGE65536_HB] w=%0d mont exp=%0d got=%0d",
                           w, LOGICAL_POINTS_PER_WINDOW,
                           mont_out_count_win[w]);
                end

                if (r_count_win[w] != BEATS_PER_WINDOW) begin
                    error_count++;
                    $error("[TB_STAGE65536_HB] w=%0d beats exp=%0d got=%0d",
                           w, BEATS_PER_WINDOW, r_count_win[w]);
                end

                if (enqueue_delta[w] != LOGICAL_POINTS_PER_WINDOW) begin
                    error_count++;
                    $error("[TB_STAGE65536_HB] w=%0d enqueue exp=%0d got=%0d",
                           w, LOGICAL_POINTS_PER_WINDOW, enqueue_delta[w]);
                end

                if (completed_delta[w] != LOGICAL_POINTS_PER_WINDOW) begin
                    error_count++;
                    $error("[TB_STAGE65536_HB] w=%0d completed exp=%0d got=%0d",
                           w, LOGICAL_POINTS_PER_WINDOW,
                           completed_delta[w]);
                end

                if (direct_delta[w] !=
                    mw_expected_direct_write_count[w]) begin
                    error_count++;
                    $error("[TB_STAGE65536_HB] w=%0d direct exp=%0d got=%0d",
                           w, mw_expected_direct_write_count[w],
                           direct_delta[w]);
                end

                if (mixed_delta[w] !=
                    mw_expected_mixed_add_count[w]) begin
                    error_count++;
                    $error("[TB_STAGE65536_HB] w=%0d mixed exp=%0d got=%0d",
                           w, mw_expected_mixed_add_count[w],
                           mixed_delta[w]);
                end

                if (zero_count_win[w] !=
                    mw_expected_skipped_zero_count[w]) begin
                    error_count++;
                    $error("[TB_STAGE65536_HB] w=%0d zero exp=%0d got=%0d",
                           w, mw_expected_skipped_zero_count[w],
                           zero_count_win[w]);
                end

                for (lane = 0; lane < NUM_LANES; lane = lane + 1) begin
                    if (lane_points_win[w][lane] !=
                        mw_expected_lane_points[w][lane]) begin
                        error_count++;
                        $error(
                            "[TB_STAGE65536_HB] w=%0d lane=%0d exp=%0d got=%0d",
                            w, lane,
                            mw_expected_lane_points[w][lane],
                            lane_points_win[w][lane]
                        );
                    end
                end
            end
        end
    endtask

    initial begin : initialize_profile_arrays
        integer w;
        integer lane;

        for (w = 0; w < NUM_WINDOWS; w = w + 1) begin
            win_enter_cycle[w] = 0;
            win_start_cycle[w] = 0;
            win_done_cycle[w] = 0;
            next_window_cycle[w] = 0;
            axi_start_cycle[w] = 0;
            axi_first_ar_cycle[w] = 0;
            axi_first_r_cycle[w] = 0;
            axi_last_r_cycle[w] = 0;
            axi_done_cycle[w] = 0;
            conv_first_in_cycle[w] = 0;
            conv_last_in_cycle[w] = 0;
            conv_first_out_cycle[w] = 0;
            conv_last_out_cycle[w] = 0;
            build_first_cycle[w] = 0;
            build_last_cycle[w] = 0;
            build_done_cycle[w] = 0;
            reduce_start_cycle[w] = 0;
            reduce_done_cycle[w] = 0;
            ar_count_win[w] = 0;
            r_count_win[w] = 0;
            point_in_count_win[w] = 0;
            mont_out_count_win[w] = 0;
            zero_count_win[w] = 0;
            enqueue_base[w] = 0;
            completed_base[w] = 0;
            stall_base[w] = 0;
            direct_base[w] = 0;
            mixed_base[w] = 0;
            enqueue_delta[w] = 0;
            completed_delta[w] = 0;
            stall_delta[w] = 0;
            direct_delta[w] = 0;
            mixed_delta[w] = 0;
            double_count_win[w] = 0;
            add_count_win[w] = 0;
            seen_axi_ar[w] = 0;
            seen_axi_r[w] = 0;
            seen_conv_in[w] = 0;
            seen_conv_out[w] = 0;
            seen_build[w] = 0;
            seen_build_done[w] = 0;
            seen_reduce_start[w] = 0;
            seen_reduce_done[w] = 0;

            for (lane = 0; lane < NUM_LANES; lane = lane + 1) begin
                lane_points_win[w][lane] = 0;
                lane_fifo_max_win[w][lane] = 0;
            end
        end
    end

    initial begin : test_sequence
        error_count = 0;
        ar_burst_count = 0;
        r_beat_count = 0;
        source_done_count = 0;

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
            $error("[TB_STAGE65536_HB] X mismatch exp=%h got=%h",
                   MW_EXPECTED_X, result_x);
        end

        if (result_y !== MW_EXPECTED_Y) begin
            error_count++;
            $error("[TB_STAGE65536_HB] Y mismatch exp=%h got=%h",
                   MW_EXPECTED_Y, result_y);
        end

        if (result_z !== MW_EXPECTED_Z) begin
            error_count++;
            $error("[TB_STAGE65536_HB] Z mismatch exp=%h got=%h",
                   MW_EXPECTED_Z, result_z);
        end

        if (converter_busy !== 1'b0) begin
            error_count++;
            $error("[TB_STAGE65536_HB] converter not empty");
        end

        if (source_done_count != NUM_WINDOWS) begin
            error_count++;
            $error("[TB_STAGE65536_HB] source_done exp=%0d got=%0d",
                   NUM_WINDOWS, source_done_count);
        end

        if (r_beat_count != TOTAL_BEATS) begin
            error_count++;
            $error("[TB_STAGE65536_HB] total beats exp=%0d got=%0d",
                   TOTAL_BEATS, r_beat_count);
        end

        // Allow final nonblocking monitor updates to settle.
        repeat (2) @(posedge msm_clk);

        check_profile();
        write_profile_csv();

        if (error_count == 0) begin
            $display("");
            $display("============================================================");
            $display(
                "[TB_STAGE65536_HB] FULL 65536-POINT UNIFORM AXI AFFINE STAGE PROFILE PASSED"
            );
            $display("[TB_STAGE65536_HB] windows              = %0d",
                     NUM_WINDOWS);
            $display("[TB_STAGE65536_HB] original points      = %0d",
                     LOGICAL_POINTS_PER_WINDOW);
            $display("[TB_STAGE65536_HB] point-window records = %0d",
                     NUM_WINDOWS*LOGICAL_POINTS_PER_WINDOW);
            $display("[TB_STAGE65536_HB] total AXI beats      = %0d",
                     r_beat_count);
            $display("[TB_STAGE65536_HB] AXI bursts           = %0d",
                     ar_burst_count);
            $display("[TB_STAGE65536_HB] done MSM cycle       = %0d",
                     msm_cycle_count);
            $display(
                "[TB_STAGE65536_HB] CSV: runs/full_axi_affine_65536p_uniform_stageprofile_v1.csv"
            );
            $display("============================================================");
        end else begin
            $fatal(1, "[TB_STAGE65536_HB] FAILED with %0d errors",
                   error_count);
        end

        #20;
        $finish;
    end

    initial begin
        #(64'd16000000000);
        $fatal(1, "[TB_STAGE65536_HB] Watchdog timeout");
    end

endmodule