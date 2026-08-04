`timescale 1ns / 1ps

// ============================================================================
// File: rtl/mem/active/tb_msm_point_streamer_block_memmodel.sv
//
// Testbench for msm_point_streamer_block (V2 Double-Beat Packing wrapper).
//
//   STEP 1 — Infrastructure ONLY (this file):
//     * Clock / reset / waveform dump
//     * DUT instantiation with all pass-through SRAM + debug ports tied off
//     * Shared scoreboard queues (Beat1 ⊗ Beat2 pairing for phase-slip detect)
//     * Stimulus / checker helper tasks
//     * Empty main_test_sequence placeholder
//
//   STEP 2 — Directed test cases (added later) drop into main_test_sequence.
//
// Verification principles encoded here:
//   * Phase-Slip detection: every logical point pushes {X,Y,ID,Last} as a unit
//     into the scoreboard queues. A beat-assembly misalignment in the DUT
//     surfaces as an X/Y/ID mismatch at check_scheduler_output().
//   * The AXI R channel is the stimulus surface; the scheduler completion
//     stream (out_*) is the observation surface.
// ============================================================================

module tb_msm_point_streamer_block_memmodel;

    // =========================================================================
    // DUT geometry (mirror of msm_point_streamer_block defaults)
    // =========================================================================
    localparam int LANES          = 8;
    localparam int GLOBAL_ADDR_W  = 16;
    localparam int DATA_W         = 256;
    localparam int GEN_W          = 16;
    localparam int SCH_FIFO_DEPTH = 16;
    localparam int SLOT_COUNT     = 16;

    // TEST 3 backpressure: a small AXI fetch FIFO so single-lane saturation
    // fills it and m_axi_rready (= ~full) deasserts with a modest stimulus.
    localparam int AXI_FIFO_DEPTH = 8;

    localparam int MEM_ADDR_W = GLOBAL_ADDR_W - $clog2(LANES);          // 13
    localparam int FIFO_OCC_W = $clog2(SCH_FIFO_DEPTH + 1);             // 5
    localparam int SLOT_OCC_W = $clog2(SLOT_COUNT + 1);                 // 5

    // Clock: 250 MHz -> 4 ns period
    localparam time CLK_PERIOD = 4ns;

    // =========================================================================
    // Clock & reset
    // =========================================================================
    logic clk;
    logic rst_n;

    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // =========================================================================
    // Waveform dump
    // =========================================================================
    initial begin : dump_block
        $dumpfile("tb_streamer_memmodel.vcd");
        $dumpvars(0, tb_msm_point_streamer_block_memmodel);
    end

    // =========================================================================
    // DUT control / status
    // =========================================================================
    logic            start;
    logic [63:0]     base_addr;
    logic [31:0]     total_points;
    logic [GEN_W-1:0] current_gen;
    logic            fetch_busy;
    logic            fetch_done;

    // ---- AXI4 Read Address Channel (AR) — DUT is master ---------------------
    logic [63:0]     m_axi_araddr;
    logic [7:0]      m_axi_arlen;
    logic [2:0]      m_axi_arsize;
    logic [1:0]      m_axi_arburst;
    logic            m_axi_arvalid;
    logic            m_axi_arready;     // TB (slave) drives — always ready

    // ---- AXI4 Read Data Channel (R) — TB (slave) drives ---------------------
    logic [511:0]    m_axi_rdata;
    logic [1:0]      m_axi_rresp;
    logic            m_axi_rlast;
    logic            m_axi_rvalid;
    logic            m_axi_rready;      // DUT drives

    // ---- Scheduler completion stream (observation surface) ------------------
    logic                    out_valid;
    logic                    out_ready;     // TB drives (downstream backpressure)
    logic [GLOBAL_ADDR_W-1:0] out_bucket_id;
    logic                    out_skipped;
    logic                    out_direct_write;
    logic                    out_mixed_add;
    logic [DATA_W-1:0]       out_x;
    logic [DATA_W-1:0]       out_y;
    logic [DATA_W-1:0]       out_z;
    logic                    out_last_point;

    // =========================================================================
    // Pass-through SRAM bank interface + behavioral 1-cycle memory model
    //
    // The scheduler owns eight independent SRAM banks.  The previous TB tied
    // mem_rvalid permanently low, so every bucket read waited forever.  This
    // model accepts one request per bank per cycle, performs writes immediately,
    // and returns read data exactly one clock later.
    //
    // Only tag memory is cleared on reset.  X/Y/Z may retain arbitrary/stale
    // values, matching the real generation-tag architecture: coordinates are
    // meaningful only when stored_tag == current_gen.
    // =========================================================================
    logic [LANES-1:0]                       mem_valid;
    logic [LANES-1:0]                       mem_write_en;
    logic [LANES-1:0][MEM_ADDR_W-1:0]       mem_addr;
    logic [LANES-1:0][DATA_W-1:0]           mem_wdata_x;
    logic [LANES-1:0][DATA_W-1:0]           mem_wdata_y;
    logic [LANES-1:0][DATA_W-1:0]           mem_wdata_z;
    logic [LANES-1:0]                       mem_tag_write_en;
    logic [LANES-1:0][GEN_W-1:0]            mem_tag_wdata;

    logic [LANES-1:0]                       mem_ready;
    logic [LANES-1:0]                       mem_rvalid;
    logic [LANES-1:0][DATA_W-1:0]           mem_rdata_x;
    logic [LANES-1:0][DATA_W-1:0]           mem_rdata_y;
    logic [LANES-1:0][DATA_W-1:0]           mem_rdata_z;
    logic [LANES-1:0][GEN_W-1:0]            mem_tag_rdata;

    localparam int MEM_DEPTH = 1 << MEM_ADDR_W;

    logic [DATA_W-1:0] mem_model_x   [0:LANES-1][0:MEM_DEPTH-1];
    logic [DATA_W-1:0] mem_model_y   [0:LANES-1][0:MEM_DEPTH-1];
    logic [DATA_W-1:0] mem_model_z   [0:LANES-1][0:MEM_DEPTH-1];
    logic [GEN_W-1:0]  mem_model_tag [0:LANES-1][0:MEM_DEPTH-1];

    assign mem_ready = '1;

    always_ff @(posedge clk or negedge rst_n) begin : bank_memory_model
        integer lane;
        integer addr;
        if (!rst_n) begin
            mem_rvalid    <= '0;
            mem_rdata_x   <= '0;
            mem_rdata_y   <= '0;
            mem_rdata_z   <= '0;
            mem_tag_rdata <= '0;

            // A real SRAM does not reset X/Y/Z.  Clearing tags is sufficient
            // to make every bucket logically empty after reset.
            for (lane = 0; lane < LANES; lane = lane + 1) begin
                for (addr = 0; addr < MEM_DEPTH; addr = addr + 1) begin
                    mem_model_tag[lane][addr] <= '0;
                end
            end
        end else begin
            // Default: read response is a one-cycle pulse.
            mem_rvalid <= '0;

            for (lane = 0; lane < LANES; lane = lane + 1) begin
                if (mem_valid[lane] && mem_ready[lane]) begin
                    if (mem_write_en[lane]) begin
                        mem_model_x[lane][mem_addr[lane]] <= mem_wdata_x[lane];
                        mem_model_y[lane][mem_addr[lane]] <= mem_wdata_y[lane];
                        mem_model_z[lane][mem_addr[lane]] <= mem_wdata_z[lane];
                    end else begin
                        mem_rvalid[lane]    <= 1'b1;
                        mem_rdata_x[lane]   <= mem_model_x[lane][mem_addr[lane]];
                        mem_rdata_y[lane]   <= mem_model_y[lane][mem_addr[lane]];
                        mem_rdata_z[lane]   <= mem_model_z[lane][mem_addr[lane]];
                        mem_tag_rdata[lane] <= mem_model_tag[lane][mem_addr[lane]];
                    end
                end

                // Tag write is kept separate because the scheduler exposes a
                // dedicated tag write-enable alongside the coordinate write.
                if (mem_tag_write_en[lane]) begin
                    mem_model_tag[lane][mem_addr[lane]] <= mem_tag_wdata[lane];
                end
            end
        end
    end

    // =========================================================================
    // Debug / performance counters — dummy sink nets
    // =========================================================================
    logic [63:0]                       total_enqueue_count;
    logic [63:0]                       total_issue_count;
    logic [63:0]                       total_completed_count;
    logic [63:0]                       total_bypass_count;
    logic [63:0]                       total_fifo_full_stall_count;
    logic [63:0]                       total_direct_write_count;
    logic [63:0]                       total_mixed_add_count;
    logic [LANES-1:0][FIFO_OCC_W-1:0]  lane_fifo_occupancy;
    logic [LANES-1:0][SLOT_OCC_W-1:0]  lane_active_slots;

    // =========================================================================
    // DUT instantiation
    // =========================================================================
    msm_point_streamer_block #(
        .LANES            (LANES),
        .GLOBAL_ADDR_W    (GLOBAL_ADDR_W),
        .DATA_W           (DATA_W),
        .GEN_W            (GEN_W),
        .SCH_FIFO_DEPTH   (SCH_FIFO_DEPTH),
        .SLOT_COUNT       (SLOT_COUNT),
        .AXI_FIFO_DEPTH   (AXI_FIFO_DEPTH)
    ) dut (
        .clk              (clk),
        .rst_n            (rst_n),

        .start            (start),
        .base_addr        (base_addr),
        .total_points     (total_points),
        .current_gen      (current_gen),
        .fetch_busy       (fetch_busy),
        .fetch_done       (fetch_done),

        .m_axi_araddr     (m_axi_araddr),
        .m_axi_arlen      (m_axi_arlen),
        .m_axi_arsize     (m_axi_arsize),
        .m_axi_arburst    (m_axi_arburst),
        .m_axi_arvalid    (m_axi_arvalid),
        .m_axi_arready    (m_axi_arready),

        .m_axi_rdata      (m_axi_rdata),
        .m_axi_rresp      (m_axi_rresp),
        .m_axi_rlast      (m_axi_rlast),
        .m_axi_rvalid     (m_axi_rvalid),
        .m_axi_rready     (m_axi_rready),

        .out_valid        (out_valid),
        .out_ready        (out_ready),
        .out_bucket_id    (out_bucket_id),
        .out_skipped      (out_skipped),
        .out_direct_write (out_direct_write),
        .out_mixed_add    (out_mixed_add),
        .out_x            (out_x),
        .out_y            (out_y),
        .out_z            (out_z),
        .out_last_point   (out_last_point),

        .mem_valid        (mem_valid),
        .mem_write_en     (mem_write_en),
        .mem_addr         (mem_addr),
        .mem_wdata_x      (mem_wdata_x),
        .mem_wdata_y      (mem_wdata_y),
        .mem_wdata_z      (mem_wdata_z),
        .mem_tag_write_en (mem_tag_write_en),
        .mem_tag_wdata    (mem_tag_wdata),

        .mem_ready        (mem_ready),
        .mem_rvalid       (mem_rvalid),
        .mem_rdata_x      (mem_rdata_x),
        .mem_rdata_y      (mem_rdata_y),
        .mem_rdata_z      (mem_rdata_z),
        .mem_tag_rdata    (mem_tag_rdata),

        .total_enqueue_count         (total_enqueue_count),
        .total_issue_count           (total_issue_count),
        .total_completed_count       (total_completed_count),
        .total_bypass_count          (total_bypass_count),
        .total_fifo_full_stall_count (total_fifo_full_stall_count),
        .total_direct_write_count    (total_direct_write_count),
        .total_mixed_add_count       (total_mixed_add_count),
        .lane_fifo_occupancy         (lane_fifo_occupancy),
        .lane_active_slots           (lane_active_slots)
    );

    // =========================================================================
    // Shared scoreboard queues  (Beat1 ⊗ Beat2 pairing)
    //   One push per logical point; one pop per accepted scheduler output.
    // =========================================================================
    logic [DATA_W-1:0]        sb_x   [$];
    logic [DATA_W-1:0]        sb_y   [$];
    logic [GLOBAL_ADDR_W-1:0] sb_id  [$];
    logic                     sb_last[$];

    // Running tally for end-of-test reporting (STEP 2 uses these)
    int unsigned err_count = 0;
    int unsigned chk_count = 0;

    // =========================================================================
    // Completion-stream capture
    //   sched_accept_with_delay() latches the out_* bus AT the accepting edge
    //   (before the scheduler's registered outputs advance to the next beat);
    //   check_scheduler_output() then compares these captured values. Sampling
    //   after a negedge would read the NEXT beat — a classic off-by-one.
    // =========================================================================
    logic [GLOBAL_ADDR_W-1:0] cap_bucket_id;
    logic [DATA_W-1:0]        cap_x;
    logic [DATA_W-1:0]        cap_y;
    logic                     cap_skipped;

    // =========================================================================
    // out_last_point monitor (sticky counter)
    //   out_last_point pulses when the wrapper hands the LAST frame to the
    //   scheduler INPUT — decoupled by engine latency from the completion
    //   OUTPUT handshake. So it is verified here, not inside the per-point
    //   checker. Each test snapshots lp_base and checks the delta == 1.
    // =========================================================================
    int unsigned last_point_count;
    int unsigned lp_base;
    initial last_point_count = 0;
    always @(posedge clk) begin
        if (out_last_point) last_point_count <= last_point_count + 1;
    end

    // TEST 3 control
    localparam int T3_POINTS = 28;     // > saturation+fill threshold for depth-8
    logic          saw_rready_low;

    // =========================================================================
    // Deterministic, distinct point-data generators (transport-only check)
    // =========================================================================
    function automatic logic [DATA_W-1:0] mk_x(input logic [15:0] id);
        return {16{16'hA001}} ^ {{(DATA_W-16){1'b0}}, id};
    endfunction
    function automatic logic [DATA_W-1:0] mk_y(input logic [15:0] id);
        return {16{16'hB002}} ^ {{(DATA_W-16){1'b0}}, id};
    endfunction

    // =========================================================================
    // Helper task: axi_send_beat
    //   Drives ONE 512-bit beat onto the AXI R channel and completes the
    //   ready/valid handshake. `rlast` marks the final beat of an AXI burst.
    // =========================================================================
    task automatic axi_send_beat(input logic [511:0] data, input logic rlast);
        @(negedge clk);
        m_axi_rdata  = data;
        m_axi_rresp  = 2'b00;
        m_axi_rlast  = rlast;
        m_axi_rvalid = 1'b1;
        // Hold until the DUT (FIFO) accepts the beat on a rising edge.
        do @(posedge clk); while (!m_axi_rready);
        @(negedge clk);
        m_axi_rvalid = 1'b0;
        m_axi_rlast  = 1'b0;
    endtask

    // =========================================================================
    // Helper task: axi_send_point
    //   Pushes one logical point into the scoreboard, then streams its two
    //   512-bit beats per the V2 packing contract:
    //     Beat 1 = {Y[255:0], X[255:0]}
    //     Beat 2 = {.., last_point @bit16, bucket_id @[15:0]}
    //   `axi_rlast` asserts m_axi_rlast on the SECOND beat (burst boundary).
    // =========================================================================
    task automatic axi_send_point(
        input logic [DATA_W-1:0]        x,
        input logic [DATA_W-1:0]        y,
        input logic [GLOBAL_ADDR_W-1:0] id,
        input logic                     last_point,
        input logic                     axi_rlast
    );
        logic [511:0] beat1;
        logic [511:0] beat2;

        // Scoreboard: record the four fields as one atomic unit.
        sb_x.push_back(x);
        sb_y.push_back(y);
        sb_id.push_back(id);
        sb_last.push_back(last_point);

        // Beat 1: X in low half, Y in high half.
        beat1 = {y, x};

        // Beat 2: bucket_id in [15:0], last_point flag at bit GLOBAL_ADDR_W.
        beat2                       = '0;
        beat2[GLOBAL_ADDR_W-1:0]    = id;
        beat2[GLOBAL_ADDR_W]        = last_point;

        axi_send_beat(beat1, 1'b0);
        axi_send_beat(beat2, axi_rlast);
    endtask

    // =========================================================================
    // Helper task: sched_accept_with_delay
    //   Models downstream backpressure on the scheduler completion stream.
    //   Holds out_ready low for `delay` cycles, then raises it and waits for a
    //   completed out_valid/out_ready handshake before dropping it again.
    // =========================================================================
    task automatic sched_accept_with_delay(input int unsigned delay);
        out_ready = 1'b0;
        repeat (delay) @(posedge clk);
        out_ready = 1'b1;
        // Wait for the cycle where both valid and ready are asserted.
        do @(posedge clk); while (!(out_valid && out_ready));
        // Sample NOW: just after the accepting edge the scheduler's registered
        // outputs still present the beat being consumed this cycle.
        cap_bucket_id = out_bucket_id;
        cap_x         = out_x;
        cap_y         = out_y;
        cap_skipped   = out_skipped;
        out_ready = 1'b0;
        @(negedge clk);
    endtask

    // =========================================================================
    // Helper task: check_scheduler_output
    //   Pops the head of the scoreboard and compares it against the CURRENT
    //   completion-stream bus. Intended to be called on an accepted handshake.
    //   A phase-slip (mis-paired beats) shows up here as an X/Y/ID mismatch.
    // =========================================================================
    task automatic check_scheduler_output();
        logic [DATA_W-1:0]        exp_x;
        logic [DATA_W-1:0]        exp_y;
        logic [GLOBAL_ADDR_W-1:0] exp_id;

        chk_count++;

        if (sb_x.size() == 0) begin
            err_count++;
            $error("[%0t] check_scheduler_output: scoreboard EMPTY but output consumed", $time);
            return;
        end

        exp_x  = sb_x.pop_front();
        exp_y  = sb_y.pop_front();
        exp_id = sb_id.pop_front();
        void'(sb_last.pop_front());   // last-point correctness checked via the monitor

        // Compare the latched accepting-edge sample (set by sched_accept_with_delay).
        if (cap_bucket_id !== exp_id) begin
            err_count++;
            $error("[%0t] BUCKET_ID mismatch: exp=%0h got=%0h", $time, exp_id, cap_bucket_id);
        end
        if (cap_x !== exp_x) begin
            err_count++;
            $error("[%0t] X mismatch (phase-slip?): exp=%0h got=%0h", $time, exp_x, cap_x);
        end
        if (cap_y !== exp_y) begin
            err_count++;
            $error("[%0t] Y mismatch (phase-slip?): exp=%0h got=%0h", $time, exp_y, cap_y);
        end
        if (cap_skipped !== 1'b0) begin
            err_count++;
            $error("[%0t] unexpected out_skipped for bucket %0h (direct-write regime)", $time, exp_id);
        end
    endtask

    // =========================================================================
    // Helper task: drain_points
    //   Accept and check `n` consecutive completion beats, each preceded by
    //   `delay` idle cycles of downstream backpressure. 1:1 in the
    //   distinct-bucket direct-write regime (one completion per logical point).
    // =========================================================================
    task automatic drain_points(input int unsigned n, input int unsigned delay);
        for (int unsigned k = 0; k < n; k++) begin
            sched_accept_with_delay(delay);
            check_scheduler_output();
        end
    endtask

    // =========================================================================
    // Reset / default-drive sequence
    // =========================================================================
    initial begin : reset_block
        rst_n         = 1'b0;
        start         = 1'b0;
        base_addr     = '0;
        total_points  = '0;
        current_gen   = '0;
        m_axi_arready = 1'b1;   // TB slave: always ready to accept AR
        m_axi_rdata   = '0;
        m_axi_rresp   = 2'b00;
        m_axi_rlast   = 1'b0;
        m_axi_rvalid  = 1'b0;
        out_ready     = 1'b0;
        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
    end

    // =========================================================================
    // Global watchdog
    // =========================================================================
    initial begin : watchdog_block
        #100us;
        $error("[%0t] WATCHDOG TIMEOUT — simulation did not finish", $time);
        $finish;
    end

    // =========================================================================
    // Main test sequence — STEP 2 directed tests drop in here.
    // =========================================================================
    initial begin : main_test_sequence

        // Gate per the agreed rule: this DUT has no SRAM tag-init FSM, so the
        // scheduler's in_ready is the true "ready to accept" signal.
        wait (rst_n === 1'b1);
        wait (dut.u_scheduler.in_ready === 1'b1);

        // =====================================================================
        // TEST 1 — Nominal 4-point stream (distinct buckets 1..4, direct-write)
        // =====================================================================
        $display("\n[T1 %0t] === TEST 1: Nominal 4-point stream ===", $time);
        lp_base = last_point_count;
        @(negedge clk);
        base_addr    = 64'h0000_1000;
        current_gen  = 16'h0001;
        total_points = 32'd8;                 // 4 logical points x 2 beats
        start        = 1'b1;
        @(negedge clk);
        start        = 1'b0;

        fork
            begin : t1_producer
                axi_send_point(mk_x(16'd1), mk_y(16'd1), 16'd1, 1'b0, 1'b0);
                axi_send_point(mk_x(16'd2), mk_y(16'd2), 16'd2, 1'b0, 1'b0);
                axi_send_point(mk_x(16'd3), mk_y(16'd3), 16'd3, 1'b0, 1'b0);
                axi_send_point(mk_x(16'd4), mk_y(16'd4), 16'd4, 1'b1, 1'b1); // last
            end
            begin : t1_consumer
                drain_points(4, 0);
            end
        join

        if ((last_point_count - lp_base) !== 1) begin
            err_count++;
            $error("[T1] expected exactly 1 out_last_point pulse, got %0d", last_point_count - lp_base);
        end
        if (sb_x.size() != 0) begin
            err_count++;
            $error("[T1] scoreboard not drained: %0d left", sb_x.size());
        end
        $display("[T1 %0t] done (errors so far=%0d)", $time, err_count);

        // =====================================================================
        // TEST 2 — Premature last-point: FIFO told to expect 4 points, stream
        //          terminates at point 2. The remaining 4 beats are discarded.
        // =====================================================================
        repeat (20) @(posedge clk);
        $display("\n[T2 %0t] === TEST 2: Premature last-point ===", $time);
        lp_base = last_point_count;
        @(negedge clk);
        total_points = 32'd8;                 // expects 4 points...
        start        = 1'b1;
        @(negedge clk);
        start        = 1'b0;

        fork
            begin : t2_producer
                axi_send_point(mk_x(16'd5), mk_y(16'd5), 16'd5, 1'b0, 1'b0);
                axi_send_point(mk_x(16'd6), mk_y(16'd6), 16'd6, 1'b1, 1'b1); // ...stop here
            end
            begin : t2_consumer
                drain_points(2, 0);           // only 2 completions exist
            end
        join

        if ((last_point_count - lp_base) !== 1) begin
            err_count++;
            $error("[T2] expected exactly 1 out_last_point pulse, got %0d", last_point_count - lp_base);
        end
        if (sb_x.size() != 0) begin
            err_count++;
            $error("[T2] scoreboard not drained: %0d left", sb_x.size());
        end
        $display("[T2 %0t] stream terminated early after 2 points (errors so far=%0d)", $time, err_count);

        // ---- Recovery: the FIFO FSM counts beats, not last_point, so it is now
        //      hung in ACTIVE still owing 4 beats; `start` is ignored outside
        //      IDLE. Pulse reset to flush it (and the scheduler busy-map) clean.
        rst_n = 1'b0;
        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        wait (dut.u_scheduler.in_ready === 1'b1);

        // =====================================================================
        // TEST 3 — AXI backpressure: all buckets map to lane 1 (id%8==1). That
        //          one lane's 16-deep FIFO saturates, stalling the wrapper and
        //          filling the depth-8 AXI FIFO until m_axi_rready deasserts.
        // =====================================================================
        repeat (20) @(posedge clk);
        $display("\n[T3 %0t] === TEST 3: AXI backpressure stress (%0d points) ===", $time, T3_POINTS);
        lp_base        = last_point_count;
        saw_rready_low = 1'b0;
        out_ready      = 1'b0;                 // hold downstream backpressure
        @(negedge clk);
        total_points = 32'(2*T3_POINTS);
        start        = 1'b1;
        @(negedge clk);
        start        = 1'b0;

        fork
            // Producer: bursts beats; blocks naturally when m_axi_rready drops.
            begin : t3_producer
                for (int unsigned i = 0; i < T3_POINTS; i++) begin
                    automatic logic [15:0] bid = 16'(1 + 8*i);   // 1,9,17,... -> lane 1
                    axi_send_point(mk_x(bid), mk_y(bid), bid,
                                   (i == T3_POINTS-1), (i == T3_POINTS-1));
                end
            end
            // Watcher: confirm backpressure reaches the AXI data channel.
            begin : t3_watch
                wait (m_axi_rready === 1'b0);
                saw_rready_low = 1'b1;
                $display("[T3 %0t] m_axi_rready deasserted (AXI FIFO full under backpressure)", $time);
            end
            // Releaser: once backpressure is proven, open the drain and check.
            begin : t3_drain
                wait (saw_rready_low === 1'b1);
                repeat (8) @(posedge clk);     // hold the stall a while longer
                drain_points(T3_POINTS, 0);    // drain & check 100% of points
            end
        join

        if (!saw_rready_low) begin
            err_count++;
            $error("[T3] m_axi_rready never deasserted under backpressure");
        end
        if ((last_point_count - lp_base) !== 1) begin
            err_count++;
            $error("[T3] expected exactly 1 out_last_point pulse, got %0d", last_point_count - lp_base);
        end
        if (sb_x.size() != 0) begin
            err_count++;
            $error("[T3] scoreboard not drained: %0d left (dropped beats!)", sb_x.size());
        end
        $display("[T3 %0t] 100%% drained, 0 dropped (errors so far=%0d)", $time, err_count);

        // =====================================================================
        // Final verdict
        // =====================================================================
        repeat (10) @(posedge clk);
        if (err_count == 0)
            $display("\n=== ALL STEP 2 DIRECTED TESTS PASSED SUCCESSFULLY ===");
        else
            $fatal(1, "TESTS FAILED: %0d error(s) over %0d checks", err_count, chk_count);
        $finish;
    end

endmodule