// ============================================================================
// File: rtl/mem/active/bucket_update_scheduler_8lane_v1.sv
// ============================================================================
// Eight fully independent bucket-update lanes.
//
// Mapping:
//   lane_id    = global_bucket_id[LANE_W-1:0]   (bucket_id modulo 8)
//   local_addr = global_bucket_id[GLOBAL_ADDR_W-1:LANE_W]
//
// Each lane contains:
//   * local FIFO and eligible scheduler
//   * local same-bucket scoreboard
//   * local SRAM interface
//   * one jacobian_mixed_add_pipeline_v2
//   * four pipelined Montgomery multipliers
//
// Total arithmetic resources:
//   8 lanes x 4 Montgomery multipliers = 32 multipliers.
//
// A single global input stream is dispatched to its owning lane.
// Lane completions are merged through round-robin arbitration.
//
// Important zero-bucket rule:
//   Only global bucket 0 is skipped. Local address 0 in lanes 1..7 represents
//   valid global buckets 1..7 and must not be treated as bucket zero.
// ============================================================================

`timescale 1ns/1ps

module bucket_update_scheduler_8lane_v1 #(
    parameter int LANES            = 8,
    parameter int GLOBAL_ADDR_W    = 8,
    parameter int DATA_W           = 256,
    parameter int GEN_W            = 16,
    parameter int FIFO_DEPTH       = 16,
    parameter int SLOT_COUNT       = 16,
    parameter int MIX_CTX_COUNT    = 40,
    parameter int MUL_LATENCY      = 16,
    parameter bit SKIP_ZERO_BUCKET = 1'b1
) (
    input  logic                         clk,
    input  logic                         rst_n,

    // Global input update stream
    input  logic                         in_valid,
    output logic                         in_ready,
    input  logic [GEN_W-1:0]             current_gen,
    input  logic [GLOBAL_ADDR_W-1:0]     in_bucket_id,
    input  logic [DATA_W-1:0]            in_point_x,
    input  logic [DATA_W-1:0]            in_point_y,

    // Merged global completion stream
    output logic                         out_valid,
    input  logic                         out_ready,
    output logic [GLOBAL_ADDR_W-1:0]     out_bucket_id,
    output logic                         out_skipped,
    output logic                         out_direct_write,
    output logic                         out_mixed_add,
    output logic [DATA_W-1:0]            out_x,
    output logic [DATA_W-1:0]            out_y,
    output logic [DATA_W-1:0]            out_z,

    // Eight independent local SRAM-bank interfaces
    output logic [LANES-1:0]             mem_valid,
    output logic [LANES-1:0]             mem_write_en,
    output logic [LANES-1:0][GLOBAL_ADDR_W-$clog2(LANES)-1:0] mem_addr,
    output logic [LANES-1:0][DATA_W-1:0] mem_wdata_x,
    output logic [LANES-1:0][DATA_W-1:0] mem_wdata_y,
    output logic [LANES-1:0][DATA_W-1:0] mem_wdata_z,
    output logic [LANES-1:0]             mem_tag_write_en,
    output logic [LANES-1:0][GEN_W-1:0]  mem_tag_wdata,

    input  logic [LANES-1:0]             mem_ready,
    input  logic [LANES-1:0]             mem_rvalid,
    input  logic [LANES-1:0][DATA_W-1:0] mem_rdata_x,
    input  logic [LANES-1:0][DATA_W-1:0] mem_rdata_y,
    input  logic [LANES-1:0][DATA_W-1:0] mem_rdata_z,
    input  logic [LANES-1:0][GEN_W-1:0]  mem_tag_rdata,

    // Aggregate debug/performance
    output logic [63:0]                  total_enqueue_count,
    output logic [63:0]                  total_issue_count,
    output logic [63:0]                  total_completed_count,
    output logic [63:0]                  total_bypass_count,
    output logic [63:0]                  total_fifo_full_stall_count,
    output logic [63:0]                  total_direct_write_count,
    output logic [63:0]                  total_mixed_add_count,
    output logic [LANES-1:0][$clog2(FIFO_DEPTH+1)-1:0] lane_fifo_occupancy,
    output logic [LANES-1:0][$clog2(SLOT_COUNT+1)-1:0] lane_active_slots
);

    localparam int LANE_W       = $clog2(LANES);
    localparam int LOCAL_ADDR_W = GLOBAL_ADDR_W - LANE_W;
    localparam int LOCAL_DEPTH  = (1 << LOCAL_ADDR_W);

    initial begin
        if (LANES != 8)
            $error("bucket_update_scheduler_8lane_v1 currently expects LANES=8");
        if ((1 << LANE_W) != LANES)
            $error("LANES must be a power of two");
        if (GLOBAL_ADDR_W <= LANE_W)
            $error("GLOBAL_ADDR_W must be larger than LANE_W");
    end

    logic [LANE_W-1:0] selected_input_lane;
    logic [LOCAL_ADDR_W-1:0] selected_local_addr;

    logic [LANES-1:0] lane_in_valid;
    logic [LANES-1:0] lane_in_ready;

    logic [LANES-1:0] lane_out_valid;
    logic [LANES-1:0] lane_out_ready;
    logic [LANES-1:0][LOCAL_ADDR_W-1:0] lane_out_bucket;
    logic [LANES-1:0] lane_out_skipped;
    logic [LANES-1:0] lane_out_direct;
    logic [LANES-1:0] lane_out_mixed;
    logic [LANES-1:0][DATA_W-1:0] lane_out_x;
    logic [LANES-1:0][DATA_W-1:0] lane_out_y;
    logic [LANES-1:0][DATA_W-1:0] lane_out_z;

    logic [LANES-1:0][63:0] lane_enqueue_count;
    logic [LANES-1:0][63:0] lane_issue_count;
    logic [LANES-1:0][63:0] lane_bypass_count;
    logic [LANES-1:0][63:0] lane_fifo_full_stall_count;
    logic [LANES-1:0][63:0] lane_accepted_count;
    logic [LANES-1:0][63:0] lane_completed_count;
    logic [LANES-1:0][63:0] lane_downstream_same_stall_count;
    logic [LANES-1:0][63:0] lane_direct_write_count;
    logic [LANES-1:0][63:0] lane_mixed_add_count;

    logic [LANES-1:0] lane_issue_pulse;
    logic [LANES-1:0][LOCAL_ADDR_W-1:0] lane_issue_bucket;

    logic [LANE_W-1:0] output_rr;
    logic output_found;
    logic [LANE_W-1:0] output_lane;

    integer arb_i;
    integer arb_idx;
    integer sum_i;

    assign selected_input_lane = in_bucket_id[LANE_W-1:0];
    assign selected_local_addr = in_bucket_id[GLOBAL_ADDR_W-1:LANE_W];

    always_comb begin
        lane_in_valid = '0;
        lane_in_valid[selected_input_lane] = in_valid;
        in_ready = lane_in_ready[selected_input_lane];
    end

    // Round-robin completion merge.
    always_comb begin
        output_found = 1'b0;
        output_lane  = '0;

        for (arb_i = 0; arb_i < LANES; arb_i = arb_i + 1) begin
            arb_idx = output_rr + arb_i;
            if (arb_idx >= LANES)
                arb_idx = arb_idx - LANES;

            if (!output_found && lane_out_valid[arb_idx]) begin
                output_found = 1'b1;
                output_lane  = arb_idx[LANE_W-1:0];
            end
        end
    end

    always_comb begin
        lane_out_ready = '0;

        out_valid        = output_found;
        out_bucket_id    = '0;
        out_skipped      = 1'b0;
        out_direct_write = 1'b0;
        out_mixed_add    = 1'b0;
        out_x            = '0;
        out_y            = '0;
        out_z            = '0;

        if (output_found) begin
            lane_out_ready[output_lane] = out_ready;

            out_bucket_id = {
                lane_out_bucket[output_lane],
                output_lane
            };

            out_skipped      = lane_out_skipped[output_lane];
            out_direct_write = lane_out_direct[output_lane];
            out_mixed_add    = lane_out_mixed[output_lane];
            out_x            = lane_out_x[output_lane];
            out_y            = lane_out_y[output_lane];
            out_z            = lane_out_z[output_lane];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            output_rr <= '0;
        else if (out_valid && out_ready)
            output_rr <= output_lane + 1'b1;
    end

    always_comb begin
        total_enqueue_count         = 64'd0;
        total_issue_count           = 64'd0;
        total_completed_count       = 64'd0;
        total_bypass_count          = 64'd0;
        total_fifo_full_stall_count = 64'd0;
        total_direct_write_count    = 64'd0;
        total_mixed_add_count       = 64'd0;

        for (sum_i = 0; sum_i < LANES; sum_i = sum_i + 1) begin
            total_enqueue_count         += lane_enqueue_count[sum_i];
            total_issue_count           += lane_issue_count[sum_i];
            total_completed_count       += lane_completed_count[sum_i];
            total_bypass_count          += lane_bypass_count[sum_i];
            total_fifo_full_stall_count += lane_fifo_full_stall_count[sum_i];
            total_direct_write_count    += lane_direct_write_count[sum_i];
            total_mixed_add_count       += lane_mixed_add_count[sum_i];
        end
    end

    genvar lane;
    generate
        for (lane = 0; lane < LANES; lane = lane + 1) begin : G_LANE
            bucket_update_scheduler_v1 #(
                .ADDR_W(LOCAL_ADDR_W),
                .DATA_W(DATA_W),
                .DEPTH(LOCAL_DEPTH),
                .GEN_W(GEN_W),
                .FIFO_DEPTH(FIFO_DEPTH),
                .SLOT_COUNT(SLOT_COUNT),
                .MIX_CTX_COUNT(MIX_CTX_COUNT),
                .MUL_LATENCY(MUL_LATENCY),
                .SKIP_ZERO_BUCKET(SKIP_ZERO_BUCKET && (lane == 0))
            ) u_lane (
                .clk(clk),
                .rst_n(rst_n),

                .in_valid(lane_in_valid[lane]),
                .in_ready(lane_in_ready[lane]),
                .current_gen(current_gen),
                .in_bucket_id(selected_local_addr),
                .in_point_x(in_point_x),
                .in_point_y(in_point_y),

                .out_valid(lane_out_valid[lane]),
                .out_ready(lane_out_ready[lane]),
                .out_bucket_id(lane_out_bucket[lane]),
                .out_skipped(lane_out_skipped[lane]),
                .out_direct_write(lane_out_direct[lane]),
                .out_mixed_add(lane_out_mixed[lane]),
                .out_x(lane_out_x[lane]),
                .out_y(lane_out_y[lane]),
                .out_z(lane_out_z[lane]),

                .mem_valid(mem_valid[lane]),
                .mem_write_en(mem_write_en[lane]),
                .mem_addr(mem_addr[lane]),
                .mem_wdata_x(mem_wdata_x[lane]),
                .mem_wdata_y(mem_wdata_y[lane]),
                .mem_wdata_z(mem_wdata_z[lane]),
                .mem_tag_write_en(mem_tag_write_en[lane]),
                .mem_tag_wdata(mem_tag_wdata[lane]),

                .mem_ready(mem_ready[lane]),
                .mem_rvalid(mem_rvalid[lane]),
                .mem_rdata_x(mem_rdata_x[lane]),
                .mem_rdata_y(mem_rdata_y[lane]),
                .mem_rdata_z(mem_rdata_z[lane]),
                .mem_tag_rdata(mem_tag_rdata[lane]),

                .fifo_occupancy(lane_fifo_occupancy[lane]),
                .enqueue_count(lane_enqueue_count[lane]),
                .issue_count(lane_issue_count[lane]),
                .bypass_count(lane_bypass_count[lane]),
                .fifo_full_stall_count(lane_fifo_full_stall_count[lane]),
                .issue_pulse(lane_issue_pulse[lane]),
                .issue_bucket_id(lane_issue_bucket[lane]),

                .active_slots(lane_active_slots[lane]),
                .accepted_count(lane_accepted_count[lane]),
                .completed_count(lane_completed_count[lane]),
                .downstream_same_bucket_stall_count(
                    lane_downstream_same_stall_count[lane]
                ),
                .direct_write_count(lane_direct_write_count[lane]),
                .mixed_add_count(lane_mixed_add_count[lane])
            );
        end
    endgenerate

endmodule