// ============================================================================
// File: rtl/mem/active/bucket_update_scheduler_v1.sv
// ============================================================================
// Small out-of-order input queue placed in front of bucket_update_pipeline_v1.
//
// Purpose:
//   Remove head-of-line blocking caused by repeated updates to the same bucket.
//
// Operation:
//   * Input updates are enqueued into a small register-based queue.
//   * Each entry receives a monotonically increasing sequence number.
//   * The scheduler issues the oldest entry whose bucket is not busy.
//   * A local busy map mirrors updates accepted by bucket_update_pipeline_v1.
//   * The busy bit is cleared only when that bucket's completion is consumed.
//   * Therefore later independent buckets may bypass an older blocked entry,
//     while updates belonging to the same bucket remain ordered.
//
// Notes:
//   * Bucket zero is always eligible and never marks a busy bit.
//   * current_gen is stored per queue entry.
//   * This v1 scheduler uses a register queue and register busy map.
// ============================================================================

`timescale 1ns/1ps

module bucket_update_scheduler_v1 #(
    parameter int ADDR_W           = 8,
    parameter int DATA_W           = 256,
    parameter int DEPTH            = (1 << ADDR_W),
    parameter int GEN_W            = 16,
    parameter int FIFO_DEPTH       = 16,
    parameter int SLOT_COUNT       = 16,
    parameter int MIX_CTX_COUNT    = 40,
    parameter int MUL_LATENCY      = 16,
    parameter bit SKIP_ZERO_BUCKET = 1'b1
) (
    input  logic                   clk,
    input  logic                   rst_n,

    // External input stream
    input  logic                   in_valid,
    output logic                   in_ready,
    input  logic [GEN_W-1:0]       current_gen,
    input  logic [ADDR_W-1:0]      in_bucket_id,
    input  logic [DATA_W-1:0]      in_point_x,
    input  logic [DATA_W-1:0]      in_point_y,

    // Completion stream from the bucket engine
    output logic                   out_valid,
    input  logic                   out_ready,
    output logic [ADDR_W-1:0]      out_bucket_id,
    output logic                   out_skipped,
    output logic                   out_direct_write,
    output logic                   out_mixed_add,
    output logic [DATA_W-1:0]      out_x,
    output logic [DATA_W-1:0]      out_y,
    output logic [DATA_W-1:0]      out_z,

    // Shared bucket-memory interface
    output logic                   mem_valid,
    output logic                   mem_write_en,
    output logic [ADDR_W-1:0]      mem_addr,
    output logic [DATA_W-1:0]      mem_wdata_x,
    output logic [DATA_W-1:0]      mem_wdata_y,
    output logic [DATA_W-1:0]      mem_wdata_z,
    output logic                   mem_tag_write_en,
    output logic [GEN_W-1:0]       mem_tag_wdata,

    input  logic                   mem_ready,
    input  logic                   mem_rvalid,
    input  logic [DATA_W-1:0]      mem_rdata_x,
    input  logic [DATA_W-1:0]      mem_rdata_y,
    input  logic [DATA_W-1:0]      mem_rdata_z,
    input  logic [GEN_W-1:0]       mem_tag_rdata,

    // Scheduler debug / performance
    output logic [$clog2(FIFO_DEPTH+1)-1:0] fifo_occupancy,
    output logic [63:0]            enqueue_count,
    output logic [63:0]            issue_count,
    output logic [63:0]            bypass_count,
    output logic [63:0]            fifo_full_stall_count,

    output logic                   issue_pulse,
    output logic [ADDR_W-1:0]      issue_bucket_id,

    // Downstream engine debug / performance
    output logic [$clog2(SLOT_COUNT+1)-1:0] active_slots,
    output logic [63:0]            accepted_count,
    output logic [63:0]            completed_count,
    output logic [63:0]            downstream_same_bucket_stall_count,
    output logic [63:0]            direct_write_count,
    output logic [63:0]            mixed_add_count
);

    localparam int FIFO_W =
        (FIFO_DEPTH <= 1) ? 1 : $clog2(FIFO_DEPTH);

    logic fifo_valid [0:FIFO_DEPTH-1];

    logic [63:0]       fifo_seq      [0:FIFO_DEPTH-1];
    logic [GEN_W-1:0]  fifo_gen      [0:FIFO_DEPTH-1];
    logic [ADDR_W-1:0] fifo_bucket   [0:FIFO_DEPTH-1];
    logic [DATA_W-1:0] fifo_point_x  [0:FIFO_DEPTH-1];
    logic [DATA_W-1:0] fifo_point_y  [0:FIFO_DEPTH-1];

    logic [DEPTH-1:0] sched_busy_map;

    logic [63:0] next_sequence;

    logic alloc_found;
    logic [FIFO_W-1:0] alloc_idx;

    logic eligible_found;
    logic [FIFO_W-1:0] eligible_idx;
    logic [63:0] eligible_seq;

    logic older_blocked_exists;

    logic hold_valid;
    logic [FIFO_W-1:0] hold_idx;

    logic selected_valid;
    logic [FIFO_W-1:0] selected_idx;

    logic engine_in_valid;
    logic engine_in_ready;
    logic [GEN_W-1:0] engine_current_gen;
    logic [ADDR_W-1:0] engine_bucket_id;
    logic [DATA_W-1:0] engine_point_x;
    logic [DATA_W-1:0] engine_point_y;

    logic enqueue_fire;
    logic issue_fire;
    logic completion_fire;

    integer alloc_i;
    integer select_i;
    integer bypass_i;
    integer occupancy_i;

    // ------------------------------------------------------------------------
    // Find one free queue entry.
    // ------------------------------------------------------------------------
    always_comb begin
        alloc_found = 1'b0;
        alloc_idx   = '0;

        for (alloc_i = 0; alloc_i < FIFO_DEPTH; alloc_i = alloc_i + 1) begin
            if (!alloc_found && !fifo_valid[alloc_i]) begin
                alloc_found = 1'b1;
                alloc_idx   = alloc_i[FIFO_W-1:0];
            end
        end
    end

    assign in_ready    = alloc_found;
    assign enqueue_fire = in_valid && in_ready;

    // ------------------------------------------------------------------------
    // Select the oldest eligible entry.
    //
    // An entry is eligible when:
    //   * its bucket is zero and zero-bucket skipping is enabled, or
    //   * its bucket is not marked busy.
    // ------------------------------------------------------------------------
    always_comb begin
        eligible_found = 1'b0;
        eligible_idx   = '0;
        eligible_seq   = 64'hFFFF_FFFF_FFFF_FFFF;

        for (select_i = 0; select_i < FIFO_DEPTH; select_i = select_i + 1) begin
            if (fifo_valid[select_i] &&
                ((SKIP_ZERO_BUCKET && (fifo_bucket[select_i] == '0)) ||
                 !sched_busy_map[fifo_bucket[select_i]]) &&
                (!eligible_found || (fifo_seq[select_i] < eligible_seq))) begin

                eligible_found = 1'b1;
                eligible_idx   = select_i[FIFO_W-1:0];
                eligible_seq   = fifo_seq[select_i];
            end
        end
    end

    // Hold a selected entry stable while the downstream engine backpressures.
    always_comb begin
        if (hold_valid) begin
            selected_valid = 1'b1;
            selected_idx   = hold_idx;
        end else begin
            selected_valid = eligible_found;
            selected_idx   = eligible_idx;
        end
    end

    assign engine_in_valid    = selected_valid;
    assign engine_current_gen = selected_valid ? fifo_gen[selected_idx]     : '0;
    assign engine_bucket_id   = selected_valid ? fifo_bucket[selected_idx]  : '0;
    assign engine_point_x     = selected_valid ? fifo_point_x[selected_idx] : '0;
    assign engine_point_y     = selected_valid ? fifo_point_y[selected_idx] : '0;

    assign issue_fire = engine_in_valid && engine_in_ready;

    // A bypass occurred when the issued entry is younger than at least one
    // valid blocked entry still in the queue.
    always_comb begin
        older_blocked_exists = 1'b0;

        if (selected_valid) begin
            for (bypass_i = 0; bypass_i < FIFO_DEPTH; bypass_i = bypass_i + 1) begin
                if (fifo_valid[bypass_i] &&
                    (fifo_seq[bypass_i] < fifo_seq[selected_idx]) &&
                    !(SKIP_ZERO_BUCKET && (fifo_bucket[bypass_i] == '0)) &&
                    sched_busy_map[fifo_bucket[bypass_i]]) begin

                    older_blocked_exists = 1'b1;
                end
            end
        end
    end

    assign completion_fire = out_valid && out_ready;

    // ------------------------------------------------------------------------
    // Queue occupancy
    // ------------------------------------------------------------------------
    always_comb begin
        fifo_occupancy = '0;

        for (occupancy_i = 0;
             occupancy_i < FIFO_DEPTH;
             occupancy_i = occupancy_i + 1) begin

            if (fifo_valid[occupancy_i])
                fifo_occupancy = fifo_occupancy + 1'b1;
        end
    end

    // ------------------------------------------------------------------------
    // Sequential queue, hold register, mirrored busy map, and counters
    // ------------------------------------------------------------------------
    integer i;
    integer b;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            next_sequence        <= 64'd0;
            hold_valid           <= 1'b0;
            hold_idx             <= '0;

            enqueue_count        <= 64'd0;
            issue_count          <= 64'd0;
            bypass_count         <= 64'd0;
            fifo_full_stall_count <= 64'd0;

            issue_pulse          <= 1'b0;
            issue_bucket_id      <= '0;

            for (i = 0; i < FIFO_DEPTH; i = i + 1) begin
                fifo_valid[i]   <= 1'b0;
                fifo_seq[i]     <= 64'd0;
                fifo_gen[i]     <= '0;
                fifo_bucket[i]  <= '0;
                fifo_point_x[i] <= '0;
                fifo_point_y[i] <= '0;
            end

            for (b = 0; b < DEPTH; b = b + 1)
                sched_busy_map[b] <= 1'b0;
        end else begin
            issue_pulse <= 1'b0;

            if (in_valid && !in_ready)
                fifo_full_stall_count <= fifo_full_stall_count + 64'd1;

            if (enqueue_fire) begin
                fifo_valid[alloc_idx]   <= 1'b1;
                fifo_seq[alloc_idx]     <= next_sequence;
                fifo_gen[alloc_idx]     <= current_gen;
                fifo_bucket[alloc_idx]  <= in_bucket_id;
                fifo_point_x[alloc_idx] <= in_point_x;
                fifo_point_y[alloc_idx] <= in_point_y;

                next_sequence <= next_sequence + 64'd1;
                enqueue_count <= enqueue_count + 64'd1;
            end

            // Capture a selected item if the engine cannot accept it yet.
            if (!hold_valid && eligible_found && !engine_in_ready) begin
                hold_valid <= 1'b1;
                hold_idx   <= eligible_idx;
            end

            if (issue_fire) begin
                fifo_valid[selected_idx] <= 1'b0;
                hold_valid               <= 1'b0;

                issue_count     <= issue_count + 64'd1;
                issue_pulse     <= 1'b1;
                issue_bucket_id <= fifo_bucket[selected_idx];

                if (older_blocked_exists)
                    bypass_count <= bypass_count + 64'd1;

                if (!(SKIP_ZERO_BUCKET &&
                      (fifo_bucket[selected_idx] == '0))) begin
                    sched_busy_map[fifo_bucket[selected_idx]] <= 1'b1;
                end
            end

            // Keep the scheduler's busy map aligned with the downstream
            // scoreboard. A bucket is released only when completion is
            // consumed by the external sink.
            if (completion_fire &&
                !(SKIP_ZERO_BUCKET && (out_bucket_id == '0))) begin
                sched_busy_map[out_bucket_id] <= 1'b0;
            end
        end
    end

    // ------------------------------------------------------------------------
    // Existing pipelined bucket-update engine
    // ------------------------------------------------------------------------
    bucket_update_pipeline_v1 #(
        .ADDR_W(ADDR_W),
        .DATA_W(DATA_W),
        .DEPTH(DEPTH),
        .GEN_W(GEN_W),
        .SLOT_COUNT(SLOT_COUNT),
        .MIX_CTX_COUNT(MIX_CTX_COUNT),
        .MUL_LATENCY(MUL_LATENCY),
        .SKIP_ZERO_BUCKET(SKIP_ZERO_BUCKET)
    ) u_bucket_update_pipeline (
        .clk(clk),
        .rst_n(rst_n),

        .in_valid(engine_in_valid),
        .in_ready(engine_in_ready),
        .current_gen(engine_current_gen),
        .in_bucket_id(engine_bucket_id),
        .in_point_x(engine_point_x),
        .in_point_y(engine_point_y),

        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_bucket_id(out_bucket_id),
        .out_skipped(out_skipped),
        .out_direct_write(out_direct_write),
        .out_mixed_add(out_mixed_add),
        .out_x(out_x),
        .out_y(out_y),
        .out_z(out_z),

        .mem_valid(mem_valid),
        .mem_write_en(mem_write_en),
        .mem_addr(mem_addr),
        .mem_wdata_x(mem_wdata_x),
        .mem_wdata_y(mem_wdata_y),
        .mem_wdata_z(mem_wdata_z),
        .mem_tag_write_en(mem_tag_write_en),
        .mem_tag_wdata(mem_tag_wdata),

        .mem_ready(mem_ready),
        .mem_rvalid(mem_rvalid),
        .mem_rdata_x(mem_rdata_x),
        .mem_rdata_y(mem_rdata_y),
        .mem_rdata_z(mem_rdata_z),
        .mem_tag_rdata(mem_tag_rdata),

        .active_slots(active_slots),
        .accepted_count(accepted_count),
        .completed_count(completed_count),
        .same_bucket_stall_count(downstream_same_bucket_stall_count),
        .direct_write_count(direct_write_count),
        .mixed_add_count(mixed_add_count)
    );

endmodule