`timescale 1ns / 1ps

// ============================================================================
// File: rtl/mem/active/msm_point_streamer_block.sv
//
// Integration wrapper: AXI Point Fetch FIFO + 8-Lane Bucket Update Scheduler.
// V2: Double-Beat Packing.
//
// The AXI/FIFO bus stays at standard 512 bits. The wrapper's 3-state FSM
// reads two consecutive 512-bit beats from the FIFO, assembles them into the
// 544-bit internal frame that the scheduler expects, then presents the
// registered frame on the scheduler's ready/valid input.
//
// axi_point_fetch_fifo.sv requires NO changes for this scheme.
//
// ── DMA beat-packing contract ────────────────────────────────────────────────
// Beat 1  [511:0]:
//   [255:0]   = point_x   (Montgomery affine X, 256 bits)
//   [511:256] = point_y   (Montgomery affine Y, 256 bits)
//
// Beat 2  [511:0]:
//   [GLOBAL_ADDR_W-1:0]   = bucket_id   (W=16 Pippenger digit; = [15:0])
//   [GLOBAL_ADDR_W]       = last_point  (stream termination;   = [16])
//   [511:GLOBAL_ADDR_W+1] = ignored
//
// Note: total_points passed to the FIFO must equal 2 × (number of logical
// points), because each logical point consumes two 512-bit beats.
//
// ── Assembled internal frame (FRAME_W = 544 bits) ────────────────────────────
//   [255:0]   = point_x
//   [511:256] = point_y
//   [527:512] = bucket_id
//   [528]     = last_point
//   [543:529] = 0 (padding, hardwired in ST_FETCH2)
// ============================================================================

module msm_point_streamer_block #(
    // ---- 8-Lane Scheduler -----------------------------------------------
    parameter int LANES            = 8,
    parameter int GLOBAL_ADDR_W    = 16,   // W=16 → 65,536 buckets
    parameter int DATA_W           = 256,
    parameter int GEN_W            = 16,
    parameter int SCH_FIFO_DEPTH   = 16,
    parameter int SLOT_COUNT       = 16,
    parameter int MIX_CTX_COUNT    = 40,
    parameter int MUL_LATENCY      = 16,
    parameter bit SKIP_ZERO_BUCKET = 1'b1,
    // ---- AXI Fetch FIFO -------------------------------------------------
    parameter int AXI_FIFO_DEPTH   = 64,
    parameter int MAX_BURST_BEATS  = 16
) (
    input  logic clk,
    input  logic rst_n,

    // ---- Control & Status -----------------------------------------------
    input  logic         start,
    input  logic [63:0]  base_addr,
    input  logic [31:0]  total_points,  // Total 512-bit beats = 2 × logical points
    input  logic [GEN_W-1:0] current_gen,
    output logic         fetch_busy,
    output logic         fetch_done,

    // ---- AXI4 Read Address Channel (AR) ---------------------------------
    output logic [63:0]  m_axi_araddr,
    output logic [7:0]   m_axi_arlen,
    output logic [2:0]   m_axi_arsize,
    output logic [1:0]   m_axi_arburst,
    output logic         m_axi_arvalid,
    input  logic         m_axi_arready,

    // ---- AXI4 Read Data Channel (R) — standard 512-bit ----------------
    input  logic [511:0] m_axi_rdata,
    input  logic [1:0]   m_axi_rresp,
    input  logic         m_axi_rlast,
    input  logic         m_axi_rvalid,
    output logic         m_axi_rready,

    // ---- Scheduler Completion Stream ------------------------------------
    output logic              out_valid,
    input  logic              out_ready,
    output logic [GLOBAL_ADDR_W-1:0] out_bucket_id,
    output logic              out_skipped,
    output logic              out_direct_write,
    output logic              out_mixed_add,
    output logic [DATA_W-1:0] out_x,
    output logic [DATA_W-1:0] out_y,
    output logic [DATA_W-1:0] out_z,

    // ---- Stream Termination Pulse ----------------------------------------
    // Pulses for one cycle when the scheduler accepts the beat whose
    // last_point flag is set. Wire to the window controller's last_point port.
    output logic              out_last_point,

    // ---- 8-Lane SRAM Bank Interface (pass-through from scheduler) -------
    output logic [LANES-1:0]  mem_valid,
    output logic [LANES-1:0]  mem_write_en,
    output logic [LANES-1:0][GLOBAL_ADDR_W-$clog2(LANES)-1:0] mem_addr,
    output logic [LANES-1:0][DATA_W-1:0] mem_wdata_x,
    output logic [LANES-1:0][DATA_W-1:0] mem_wdata_y,
    output logic [LANES-1:0][DATA_W-1:0] mem_wdata_z,
    output logic [LANES-1:0]  mem_tag_write_en,
    output logic [LANES-1:0][GEN_W-1:0]  mem_tag_wdata,

    input  logic [LANES-1:0]  mem_ready,
    input  logic [LANES-1:0]  mem_rvalid,
    input  logic [LANES-1:0][DATA_W-1:0] mem_rdata_x,
    input  logic [LANES-1:0][DATA_W-1:0] mem_rdata_y,
    input  logic [LANES-1:0][DATA_W-1:0] mem_rdata_z,
    input  logic [LANES-1:0][GEN_W-1:0]  mem_tag_rdata,

    // ---- Debug / Performance Counters (pass-through from scheduler) -----
    output logic [63:0] total_enqueue_count,
    output logic [63:0] total_issue_count,
    output logic [63:0] total_completed_count,
    output logic [63:0] total_bypass_count,
    output logic [63:0] total_fifo_full_stall_count,
    output logic [63:0] total_direct_write_count,
    output logic [63:0] total_mixed_add_count,
    output logic [LANES-1:0][$clog2(SCH_FIFO_DEPTH+1)-1:0] lane_fifo_occupancy,
    output logic [LANES-1:0][$clog2(SLOT_COUNT+1)-1:0]     lane_active_slots
);

    // =========================================================================
    // Internal Frame Layout Constants
    // =========================================================================
    localparam int FIFO_W  = 512;
    localparam int FRAME_W = 2 * DATA_W + GLOBAL_ADDR_W + 16; // 544

    // Bit-slice anchors inside the assembled 544-bit frame
    localparam int X_HI   = DATA_W - 1;                       // 255
    localparam int Y_LO   = DATA_W;                            // 256
    localparam int Y_HI   = 2 * DATA_W - 1;                   // 511
    localparam int ID_LO  = 2 * DATA_W;                        // 512
    localparam int ID_HI  = 2 * DATA_W + GLOBAL_ADDR_W - 1;   // 527
    localparam int LAST_B = 2 * DATA_W + GLOBAL_ADDR_W;        // 528

    // =========================================================================
    // FSM State Encoding
    // =========================================================================
    typedef enum logic [1:0] {
        ST_FETCH1  = 2'b00,  // Waiting for / capturing beat 1 (X, Y)
        ST_FETCH2  = 2'b01,  // Waiting for / capturing beat 2 (bucket_id, last)
        ST_PRESENT = 2'b10   // Frame registered; waiting for scheduler acceptance
    } state_e;

    state_e state_q;

    // =========================================================================
    // Internal Data Registers
    // =========================================================================
    logic [FIFO_W-1:0]  beat1_q;  // Holds beat 1 while beat 2 is being fetched
    logic [FRAME_W-1:0] frame_q;  // Fully assembled registered frame, fed to scheduler

    // =========================================================================
    // FIFO Interface Wires
    // =========================================================================
    logic         fifo_pop;
    logic [511:0] fifo_dout;
    logic         fifo_empty;
    logic         fifo_full;        // Consumed internally by FIFO to gate m_axi_rready
    logic         fifo_almost_full;

    // =========================================================================
    // Scheduler Handshake
    // =========================================================================
    logic sched_in_valid;
    logic sched_in_ready;

    // =========================================================================
    // Beat-Assembler FSM
    //
    // ST_FETCH1: Pop FIFO as soon as data is available. Register beat 1.
    //
    // ST_FETCH2: Pop FIFO as soon as data is available. On the same clock
    //            edge, assemble the 544-bit frame from beat1_q (registered)
    //            and the current fifo_dout (combinatorial FWFT output), then
    //            register it into frame_q. Transition to ST_PRESENT.
    //
    // ST_PRESENT: Drive sched_in_valid=1. The FIFO is frozen (fifo_pop=0).
    //             The scheduler's sched_in_ready is the sole exit condition,
    //             which provides natural backpressure to the FIFO fetch path.
    //             On acceptance, return to ST_FETCH1.
    //
    // start: synchronises with the FIFO's auto-flush (pointer reset on start).
    //        Any partially assembled frame is discarded by returning to FETCH1.
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_FETCH1;
            beat1_q <= '0;
            frame_q <= '0;

        end else if (start) begin
            // The FIFO resets its rd/wr pointers on this same posedge.
            // Discard any partial assembly in progress.
            state_q <= ST_FETCH1;

        end else begin
            case (state_q)

                // ---- State 0: latch beat 1 (X, Y) ----------------------
                ST_FETCH1: begin
                    if (!fifo_empty) begin
                        beat1_q <= fifo_dout;
                        state_q <= ST_FETCH2;
                    end
                end

                // ---- State 1: latch beat 2 & assemble frame -------------
                ST_FETCH2: begin
                    if (!fifo_empty) begin
                        // X and Y sourced from the already-registered beat1_q.
                        frame_q[X_HI  -: DATA_W]         <= beat1_q[X_HI  -: DATA_W];
                        frame_q[Y_HI  -: DATA_W]         <= beat1_q[Y_HI  -: DATA_W];
                        // bucket_id and last_point from the current FWFT output.
                        // fifo_dout is stable (rd_ptr unchanged until posedge).
                        frame_q[ID_HI -: GLOBAL_ADDR_W]  <= fifo_dout[GLOBAL_ADDR_W-1:0];
                        frame_q[LAST_B]                  <= fifo_dout[GLOBAL_ADDR_W];
                        frame_q[FRAME_W-1 -: 15]         <= 15'b0;  // padding
                        state_q <= ST_PRESENT;
                    end
                end

                // ---- State 2: present registered frame to scheduler -----
                ST_PRESENT: begin
                    if (sched_in_ready) begin
                        state_q <= ST_FETCH1;
                    end
                end

                default: state_q <= ST_FETCH1;

            endcase
        end
    end

    // =========================================================================
    // Combinatorial Control
    // =========================================================================

    // FIFO is active in both fetch states; frozen while scheduler is consuming.
    always_comb begin
        unique case (state_q)
            ST_FETCH1:  fifo_pop = !fifo_empty;
            ST_FETCH2:  fifo_pop = !fifo_empty;
            default:    fifo_pop = 1'b0;          // ST_PRESENT: hold fetch
        endcase
    end

    // sched_in_valid is a pure decode of the 2-bit registered state — no
    // datapath signal in the cone, so the scheduler sees a clean registered
    // fanout.
    assign sched_in_valid = (state_q == ST_PRESENT);

    // out_last_point: one-cycle pulse when the scheduler accepts the frame
    // whose last_point flag is set. Both state_q and frame_q are registers.
    assign out_last_point = (state_q == ST_PRESENT) && sched_in_ready && frame_q[LAST_B];

    // =========================================================================
    // AXI Point Fetch FIFO  (standard 512-bit DATA_W — no modifications needed)
    // =========================================================================
    axi_point_fetch_fifo #(
        .FIFO_DEPTH     (AXI_FIFO_DEPTH),
        .MAX_BURST_BEATS(MAX_BURST_BEATS)
    ) u_fetch_fifo (
        .clk            (clk),
        .rst_n          (rst_n),

        .start          (start),
        .base_addr      (base_addr),
        .total_points   (total_points),
        .busy           (fetch_busy),
        .done           (fetch_done),

        .m_axi_araddr   (m_axi_araddr),
        .m_axi_arlen    (m_axi_arlen),
        .m_axi_arsize   (m_axi_arsize),
        .m_axi_arburst  (m_axi_arburst),
        .m_axi_arvalid  (m_axi_arvalid),
        .m_axi_arready  (m_axi_arready),

        .m_axi_rdata    (m_axi_rdata),
        .m_axi_rresp    (m_axi_rresp),
        .m_axi_rlast    (m_axi_rlast),
        .m_axi_rvalid   (m_axi_rvalid),
        .m_axi_rready   (m_axi_rready),

        .pop            (fifo_pop),
        .dout           (fifo_dout),
        .empty          (fifo_empty),
        .full           (fifo_full),
        .almost_full    (fifo_almost_full)
    );

    // =========================================================================
    // 8-Lane Bucket Update Scheduler
    //
    // All data inputs are driven from the registered frame_q.
    // sched_in_valid is a registered-state decode.
    // Together these guarantee no long combinatorial paths reach the
    // scheduler's lane-select and scoreboard logic.
    // =========================================================================
    bucket_update_scheduler_8lane_v1 #(
        .LANES            (LANES),
        .GLOBAL_ADDR_W    (GLOBAL_ADDR_W),
        .DATA_W           (DATA_W),
        .GEN_W            (GEN_W),
        .FIFO_DEPTH       (SCH_FIFO_DEPTH),
        .SLOT_COUNT       (SLOT_COUNT),
        .MIX_CTX_COUNT    (MIX_CTX_COUNT),
        .MUL_LATENCY      (MUL_LATENCY),
        .SKIP_ZERO_BUCKET (SKIP_ZERO_BUCKET)
    ) u_scheduler (
        .clk              (clk),
        .rst_n            (rst_n),

        .in_valid         (sched_in_valid),
        .in_ready         (sched_in_ready),
        .current_gen      (current_gen),
        .in_bucket_id     (frame_q[ID_HI -: GLOBAL_ADDR_W]),
        .in_point_x       (frame_q[X_HI  -: DATA_W]),
        .in_point_y       (frame_q[Y_HI  -: DATA_W]),

        .out_valid        (out_valid),
        .out_ready        (out_ready),
        .out_bucket_id    (out_bucket_id),
        .out_skipped      (out_skipped),
        .out_direct_write (out_direct_write),
        .out_mixed_add    (out_mixed_add),
        .out_x            (out_x),
        .out_y            (out_y),
        .out_z            (out_z),

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

        .total_enqueue_count        (total_enqueue_count),
        .total_issue_count          (total_issue_count),
        .total_completed_count      (total_completed_count),
        .total_bypass_count         (total_bypass_count),
        .total_fifo_full_stall_count(total_fifo_full_stall_count),
        .total_direct_write_count   (total_direct_write_count),
        .total_mixed_add_count      (total_mixed_add_count),
        .lane_fifo_occupancy        (lane_fifo_occupancy),
        .lane_active_slots          (lane_active_slots)
    );

endmodule
