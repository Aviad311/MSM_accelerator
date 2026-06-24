// ============================================================================
// File: rtl/mem/active/bucket_update_pipeline_v1.sv
// ============================================================================
// One-lane pipelined bucket-update engine for the MSM Pippenger build stage.
//
// Architecture:
//   1. Accept independent bucket updates through ready/valid.
//   2. Block a new update when the same bucket is already in flight.
//   3. Read one bucket at a time from the 1RW SRAM interface.
//   4. Empty-generation buckets use a direct affine write.
//   5. Occupied buckets enter jacobian_mixed_add_pipeline_v2.
//   6. Results may finish out of order and are matched by slot tag.
//   7. Completed values are written back through the shared 1RW port.
//
// This is intentionally a one-lane prototype. It validates:
//   * multiple independent bucket updates in flight,
//   * same-bucket scoreboard protection,
//   * out-of-order MixedAdd completion,
//   * SRAM read / compute / write overlap.
//
// Important:
//   * current_gen must remain stable while operations are in flight.
//   * The scoreboard is implemented as registers in v1. For a very large
//     bucket space it should later become a bitmap RAM or a smaller CAM.
//   * The SRAM interface has no request tag, so v1 allows only one outstanding
//     SRAM read. MixedAdd computation is still overlapped across many slots.
// ============================================================================

`timescale 1ns/1ps

module bucket_update_pipeline_v1 #(
    parameter int ADDR_W           = 8,
    parameter int DATA_W           = 256,
    parameter int DEPTH            = (1 << ADDR_W),
    parameter int GEN_W            = 16,
    parameter int SLOT_COUNT       = 16,
    parameter int MIX_CTX_COUNT    = 40,
    parameter int MUL_LATENCY      = 16,
    parameter bit SKIP_ZERO_BUCKET = 1'b1
) (
    input  logic                   clk,
    input  logic                   rst_n,

    // Input update stream
    input  logic                   in_valid,
    output logic                   in_ready,
    input  logic [GEN_W-1:0]       current_gen,
    input  logic [ADDR_W-1:0]      in_bucket_id,
    input  logic [DATA_W-1:0]      in_point_x,
    input  logic [DATA_W-1:0]      in_point_y,

    // Completion stream
    output logic                   out_valid,
    input  logic                   out_ready,
    output logic [ADDR_W-1:0]      out_bucket_id,
    output logic                   out_skipped,
    output logic                   out_direct_write,
    output logic                   out_mixed_add,
    output logic [DATA_W-1:0]      out_x,
    output logic [DATA_W-1:0]      out_y,
    output logic [DATA_W-1:0]      out_z,

    // Shared 1RW bucket-memory interface
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

    // Debug / performance
    output logic [$clog2(SLOT_COUNT+1)-1:0] active_slots,
    output logic [63:0]            accepted_count,
    output logic [63:0]            completed_count,
    output logic [63:0]            same_bucket_stall_count,
    output logic [63:0]            direct_write_count,
    output logic [63:0]            mixed_add_count
);

    localparam int SLOT_W =
        (SLOT_COUNT <= 1) ? 1 : $clog2(SLOT_COUNT);

    localparam logic [255:0] ZERO =
        256'h0000000000000000000000000000000000000000000000000000000000000000;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    typedef enum logic [2:0] {
        SLOT_FREE,
        SLOT_READ_READY,
        SLOT_READ_WAIT,
        SLOT_ADD_READY,
        SLOT_ADD_WAIT,
        SLOT_WRITE_READY,
        SLOT_DONE
    } slot_state_t;

    slot_state_t slot_state [0:SLOT_COUNT-1];

    logic [ADDR_W-1:0] slot_bucket [0:SLOT_COUNT-1];
    logic [GEN_W-1:0]  slot_gen    [0:SLOT_COUNT-1];

    logic [DATA_W-1:0] slot_point_x [0:SLOT_COUNT-1];
    logic [DATA_W-1:0] slot_point_y [0:SLOT_COUNT-1];

    logic [DATA_W-1:0] slot_bucket_x [0:SLOT_COUNT-1];
    logic [DATA_W-1:0] slot_bucket_y [0:SLOT_COUNT-1];
    logic [DATA_W-1:0] slot_bucket_z [0:SLOT_COUNT-1];

    logic [DATA_W-1:0] slot_result_x [0:SLOT_COUNT-1];
    logic [DATA_W-1:0] slot_result_y [0:SLOT_COUNT-1];
    logic [DATA_W-1:0] slot_result_z [0:SLOT_COUNT-1];

    logic slot_skipped [0:SLOT_COUNT-1];
    logic slot_direct  [0:SLOT_COUNT-1];
    logic slot_mixed   [0:SLOT_COUNT-1];

    // v1 register scoreboard: one bit per bucket.
    logic [DEPTH-1:0] bucket_busy_map;

    logic [SLOT_W-1:0] alloc_rr;
    logic [SLOT_W-1:0] read_rr;
    logic [SLOT_W-1:0] add_rr;
    logic [SLOT_W-1:0] write_rr;
    logic [SLOT_W-1:0] output_rr;

    logic alloc_found;
    logic [SLOT_W-1:0] alloc_idx;

    logic read_found;
    logic [SLOT_W-1:0] read_idx;

    logic add_found;
    logic [SLOT_W-1:0] add_idx;

    logic write_found;
    logic [SLOT_W-1:0] write_idx;

    logic output_found;
    logic [SLOT_W-1:0] output_idx;

    logic read_pending_valid;
    logic [SLOT_W-1:0] read_pending_idx;

    logic input_fire;
    logic output_fire;
    logic read_fire;
    logic write_fire;
    logic add_input_fire;

    // MixedAdd pipeline interface
    logic mix_in_valid;
    logic mix_in_ready;
    logic [SLOT_W-1:0] mix_in_tag;

    logic mix_out_valid;
    logic mix_out_ready;
    logic [SLOT_W-1:0] mix_out_tag;
    logic [DATA_W-1:0] mix_out_x;
    logic [DATA_W-1:0] mix_out_y;
    logic [DATA_W-1:0] mix_out_z;
    logic mix_out_special;

    logic [$clog2(MIX_CTX_COUNT+1)-1:0] mix_active_contexts;

    integer alloc_scan_i;
    integer alloc_scan_idx;
    integer read_scan_i;
    integer read_scan_idx;
    integer add_scan_i;
    integer add_scan_idx;
    integer write_scan_i;
    integer write_scan_idx;
    integer output_scan_i;
    integer output_scan_idx;
    integer active_count_i;

    // ------------------------------------------------------------------------
    // Slot allocation
    // ------------------------------------------------------------------------
    always_comb begin
        alloc_found = 1'b0;
        alloc_idx   = '0;

        for (alloc_scan_i = 0;
             alloc_scan_i < SLOT_COUNT;
             alloc_scan_i = alloc_scan_i + 1) begin

            alloc_scan_idx = alloc_rr + alloc_scan_i;
            if (alloc_scan_idx >= SLOT_COUNT)
                alloc_scan_idx = alloc_scan_idx - SLOT_COUNT;

            if (!alloc_found &&
                slot_state[alloc_scan_idx] == SLOT_FREE) begin
                alloc_found = 1'b1;
                alloc_idx   = alloc_scan_idx[SLOT_W-1:0];
            end
        end
    end

    always_comb begin
        if (!alloc_found) begin
            in_ready = 1'b0;
        end else if (SKIP_ZERO_BUCKET && (in_bucket_id == '0)) begin
            in_ready = 1'b1;
        end else begin
            in_ready = !bucket_busy_map[in_bucket_id];
        end
    end

    assign input_fire = in_valid && in_ready;

    // ------------------------------------------------------------------------
    // Read scheduler: only one outstanding untagged SRAM read in v1
    // ------------------------------------------------------------------------
    always_comb begin
        read_found = 1'b0;
        read_idx   = '0;

        for (read_scan_i = 0;
             read_scan_i < SLOT_COUNT;
             read_scan_i = read_scan_i + 1) begin

            read_scan_idx = read_rr + read_scan_i;
            if (read_scan_idx >= SLOT_COUNT)
                read_scan_idx = read_scan_idx - SLOT_COUNT;

            if (!read_found &&
                slot_state[read_scan_idx] == SLOT_READ_READY) begin
                read_found = 1'b1;
                read_idx   = read_scan_idx[SLOT_W-1:0];
            end
        end
    end

    // ------------------------------------------------------------------------
    // MixedAdd input scheduler
    // ------------------------------------------------------------------------
    always_comb begin
        add_found = 1'b0;
        add_idx   = '0;

        for (add_scan_i = 0;
             add_scan_i < SLOT_COUNT;
             add_scan_i = add_scan_i + 1) begin

            add_scan_idx = add_rr + add_scan_i;
            if (add_scan_idx >= SLOT_COUNT)
                add_scan_idx = add_scan_idx - SLOT_COUNT;

            if (!add_found &&
                slot_state[add_scan_idx] == SLOT_ADD_READY) begin
                add_found = 1'b1;
                add_idx   = add_scan_idx[SLOT_W-1:0];
            end
        end
    end

    assign mix_in_valid = add_found;
    assign mix_in_tag   = add_idx;

    assign add_input_fire = mix_in_valid && mix_in_ready;

    // The slot table can always absorb a returning MixedAdd result.
    assign mix_out_ready = 1'b1;

    // ------------------------------------------------------------------------
    // Write scheduler
    // ------------------------------------------------------------------------
    always_comb begin
        write_found = 1'b0;
        write_idx   = '0;

        for (write_scan_i = 0;
             write_scan_i < SLOT_COUNT;
             write_scan_i = write_scan_i + 1) begin

            write_scan_idx = write_rr + write_scan_i;
            if (write_scan_idx >= SLOT_COUNT)
                write_scan_idx = write_scan_idx - SLOT_COUNT;

            if (!write_found &&
                slot_state[write_scan_idx] == SLOT_WRITE_READY) begin
                write_found = 1'b1;
                write_idx   = write_scan_idx[SLOT_W-1:0];
            end
        end
    end

    // ------------------------------------------------------------------------
    // Completion scheduler
    // ------------------------------------------------------------------------
    always_comb begin
        output_found = 1'b0;
        output_idx   = '0;

        for (output_scan_i = 0;
             output_scan_i < SLOT_COUNT;
             output_scan_i = output_scan_i + 1) begin

            output_scan_idx = output_rr + output_scan_i;
            if (output_scan_idx >= SLOT_COUNT)
                output_scan_idx = output_scan_idx - SLOT_COUNT;

            if (!output_found &&
                slot_state[output_scan_idx] == SLOT_DONE) begin
                output_found = 1'b1;
                output_idx   = output_scan_idx[SLOT_W-1:0];
            end
        end
    end

    assign out_valid        = output_found;
    assign out_bucket_id    = output_found ? slot_bucket[output_idx]   : '0;
    assign out_skipped      = output_found ? slot_skipped[output_idx]  : 1'b0;
    assign out_direct_write = output_found ? slot_direct[output_idx]   : 1'b0;
    assign out_mixed_add    = output_found ? slot_mixed[output_idx]    : 1'b0;
    assign out_x            = output_found ? slot_result_x[output_idx] : ZERO;
    assign out_y            = output_found ? slot_result_y[output_idx] : ONE_M;
    assign out_z            = output_found ? slot_result_z[output_idx] : ZERO;

    assign output_fire = out_valid && out_ready;

    // ------------------------------------------------------------------------
    // Shared 1RW memory arbitration
    //
    // Writes have priority so completed results do not accumulate forever.
    // ------------------------------------------------------------------------
    always_comb begin
        mem_valid        = 1'b0;
        mem_write_en     = 1'b0;
        mem_addr         = '0;
        mem_wdata_x      = ZERO;
        mem_wdata_y      = ONE_M;
        mem_wdata_z      = ZERO;
        mem_tag_write_en = 1'b0;
        mem_tag_wdata    = '0;

        if (write_found) begin
            mem_valid        = 1'b1;
            mem_write_en     = 1'b1;
            mem_addr         = slot_bucket[write_idx];
            mem_wdata_x      = slot_result_x[write_idx];
            mem_wdata_y      = slot_result_y[write_idx];
            mem_wdata_z      = slot_result_z[write_idx];
            mem_tag_write_en = 1'b1;
            mem_tag_wdata    = slot_gen[write_idx];
        end else if (read_found && !read_pending_valid) begin
            mem_valid    = 1'b1;
            mem_write_en = 1'b0;
            mem_addr     = slot_bucket[read_idx];
        end
    end

    assign write_fire =
        mem_valid && mem_write_en && mem_ready;

    assign read_fire =
        mem_valid && !mem_write_en && mem_ready;

    // ------------------------------------------------------------------------
    // Active-slot count
    // ------------------------------------------------------------------------
    always_comb begin
        active_slots = '0;

        for (active_count_i = 0;
             active_count_i < SLOT_COUNT;
             active_count_i = active_count_i + 1) begin

            if (slot_state[active_count_i] != SLOT_FREE)
                active_slots = active_slots + 1'b1;
        end
    end

    // ------------------------------------------------------------------------
    // Sequential state updates
    // ------------------------------------------------------------------------
    integer i;
    integer b;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            alloc_rr          <= '0;
            read_rr           <= '0;
            add_rr            <= '0;
            write_rr          <= '0;
            output_rr         <= '0;
            read_pending_valid <= 1'b0;
            read_pending_idx   <= '0;

            accepted_count          <= 64'd0;
            completed_count         <= 64'd0;
            same_bucket_stall_count <= 64'd0;
            direct_write_count      <= 64'd0;
            mixed_add_count         <= 64'd0;

            for (i = 0; i < SLOT_COUNT; i = i + 1) begin
                slot_state[i] <= SLOT_FREE;
                slot_bucket[i] <= '0;
                slot_gen[i] <= '0;

                slot_point_x[i] <= ZERO;
                slot_point_y[i] <= ZERO;

                slot_bucket_x[i] <= ZERO;
                slot_bucket_y[i] <= ONE_M;
                slot_bucket_z[i] <= ZERO;

                slot_result_x[i] <= ZERO;
                slot_result_y[i] <= ONE_M;
                slot_result_z[i] <= ZERO;

                slot_skipped[i] <= 1'b0;
                slot_direct[i]  <= 1'b0;
                slot_mixed[i]   <= 1'b0;
            end

            for (b = 0; b < DEPTH; b = b + 1)
                bucket_busy_map[b] <= 1'b0;
        end else begin
            // Count only true same-bucket hazards.
            if (in_valid &&
                !(SKIP_ZERO_BUCKET && (in_bucket_id == '0)) &&
                bucket_busy_map[in_bucket_id]) begin
                same_bucket_stall_count <=
                    same_bucket_stall_count + 64'd1;
            end

            // Accept one new update.
            if (input_fire) begin
                slot_bucket[alloc_idx] <= in_bucket_id;
                slot_gen[alloc_idx]    <= current_gen;
                slot_point_x[alloc_idx] <= in_point_x;
                slot_point_y[alloc_idx] <= in_point_y;

                slot_skipped[alloc_idx] <= 1'b0;
                slot_direct[alloc_idx]  <= 1'b0;
                slot_mixed[alloc_idx]   <= 1'b0;

                accepted_count <= accepted_count + 64'd1;
                alloc_rr       <= alloc_idx + 1'b1;

                if (SKIP_ZERO_BUCKET && (in_bucket_id == '0)) begin
                    slot_result_x[alloc_idx] <= ZERO;
                    slot_result_y[alloc_idx] <= ONE_M;
                    slot_result_z[alloc_idx] <= ZERO;
                    slot_skipped[alloc_idx]  <= 1'b1;
                    slot_state[alloc_idx]    <= SLOT_DONE;
                end else begin
                    bucket_busy_map[in_bucket_id] <= 1'b1;
                    slot_state[alloc_idx] <= SLOT_READ_READY;
                end
            end

            // Launch one SRAM read.
            if (read_fire) begin
                slot_state[read_idx] <= SLOT_READ_WAIT;
                read_pending_valid   <= 1'b1;
                read_pending_idx     <= read_idx;
                read_rr              <= read_idx + 1'b1;
            end

            // Consume the one outstanding SRAM read response.
            if (mem_rvalid && read_pending_valid) begin
                read_pending_valid <= 1'b0;

                if (mem_tag_rdata !== slot_gen[read_pending_idx]) begin
                    // Empty in this generation: direct affine write.
                    slot_result_x[read_pending_idx] <=
                        slot_point_x[read_pending_idx];
                    slot_result_y[read_pending_idx] <=
                        slot_point_y[read_pending_idx];
                    slot_result_z[read_pending_idx] <= ONE_M;

                    slot_direct[read_pending_idx] <= 1'b1;
                    slot_state[read_pending_idx]  <= SLOT_WRITE_READY;

                    direct_write_count <= direct_write_count + 64'd1;
                end else begin
                    slot_bucket_x[read_pending_idx] <= mem_rdata_x;
                    slot_bucket_y[read_pending_idx] <= mem_rdata_y;
                    slot_bucket_z[read_pending_idx] <= mem_rdata_z;

                    slot_mixed[read_pending_idx] <= 1'b1;
                    slot_state[read_pending_idx] <= SLOT_ADD_READY;

                    mixed_add_count <= mixed_add_count + 64'd1;
                end
            end

            // Send one occupied bucket to the streaming MixedAdd engine.
            if (add_input_fire) begin
                slot_state[add_idx] <= SLOT_ADD_WAIT;
                add_rr              <= add_idx + 1'b1;
            end

            // Capture an out-of-order MixedAdd result by slot tag.
            if (mix_out_valid && mix_out_ready) begin
                slot_result_x[mix_out_tag] <= mix_out_x;
                slot_result_y[mix_out_tag] <= mix_out_y;
                slot_result_z[mix_out_tag] <= mix_out_z;
                slot_state[mix_out_tag]    <= SLOT_WRITE_READY;
            end

            // Commit one result to SRAM.
            if (write_fire) begin
                slot_state[write_idx] <= SLOT_DONE;
                write_rr              <= write_idx + 1'b1;
            end

            // Retire one completed update.
            if (output_fire) begin
                if (!(SKIP_ZERO_BUCKET &&
                      (slot_bucket[output_idx] == '0))) begin
                    bucket_busy_map[slot_bucket[output_idx]] <= 1'b0;
                end

                slot_state[output_idx] <= SLOT_FREE;
                output_rr             <= output_idx + 1'b1;
                completed_count       <= completed_count + 64'd1;
            end
        end
    end

    // ------------------------------------------------------------------------
    // Streaming MixedAdd v2
    // ------------------------------------------------------------------------
    jacobian_mixed_add_pipeline_v2 #(
        .WIDTH(DATA_W),
        .TAG_W(SLOT_W),
        .CTX_COUNT(MIX_CTX_COUNT),
        .MUL_LATENCY(MUL_LATENCY)
    ) u_mixed_add_pipeline (
        .clk(clk),
        .rst_n(rst_n),

        .in_valid(mix_in_valid),
        .in_ready(mix_in_ready),
        .in_tag(mix_in_tag),

        .in_X1(slot_bucket_x[add_idx]),
        .in_Y1(slot_bucket_y[add_idx]),
        .in_Z1(slot_bucket_z[add_idx]),
        .in_X2(slot_point_x[add_idx]),
        .in_Y2(slot_point_y[add_idx]),

        .out_valid(mix_out_valid),
        .out_ready(mix_out_ready),
        .out_tag(mix_out_tag),
        .out_X3(mix_out_x),
        .out_Y3(mix_out_y),
        .out_Z3(mix_out_z),
        .out_special(mix_out_special),

        .active_contexts(mix_active_contexts)
    );

endmodule