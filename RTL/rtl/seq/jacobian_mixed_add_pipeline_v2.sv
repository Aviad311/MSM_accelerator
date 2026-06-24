// ============================================================================
// File: rtl/seq/jacobian_mixed_add_pipeline_v2.sv
// ============================================================================
// Streaming normal-path Jacobian mixed-add prototype for secp256k1.
//
// Main idea:
//   * Uses the verified secp256k1_montgomery_mul directly.
//   * Four multiplier pipelines are instantiated.
//   * A bank of independent contexts hides the 16-cycle multiplier latency.
//   * One ready context stage can be issued every clock.
//   * Each normal mixed-add requires six issue bundles, so the theoretical
//     steady-state initiation interval is approximately 6 clocks/result,
//     provided enough independent contexts are available.
//
// Supported:
//   * Normal Jacobian + affine mixed-add path.
//   * Z1 == 0 shortcut: returns the affine input with Z = ONE_M.
//
// Fully handled special paths:
//   * H == 0, R == 0  -> Jacobian doubling through jacobian_double_seq.
//   * H == 0, R != 0  -> point at infinity.
//
// out_special is retained for interface compatibility. In v2 it is zero for
// every completed, valid result.
//
// IMPORTANT SYSTEM-LEVEL RESTRICTION:
//   The caller must not allow two in-flight operations to update the same
//   bucket unless a scoreboard/forwarding mechanism is present.
// ============================================================================

`timescale 1ns/1ps

module jacobian_mixed_add_pipeline_v2 #(
    parameter int WIDTH       = 256,
    parameter int TAG_W       = 16,
    parameter int CTX_COUNT   = 24,
    parameter int MUL_LATENCY = 16
) (
    input  logic                 clk,
    input  logic                 rst_n,

    input  logic                 in_valid,
    output logic                 in_ready,
    input  logic [TAG_W-1:0]     in_tag,

    input  logic [WIDTH-1:0]     in_X1,
    input  logic [WIDTH-1:0]     in_Y1,
    input  logic [WIDTH-1:0]     in_Z1,
    input  logic [WIDTH-1:0]     in_X2,
    input  logic [WIDTH-1:0]     in_Y2,

    output logic                 out_valid,
    input  logic                 out_ready,
    output logic [TAG_W-1:0]     out_tag,
    output logic [WIDTH-1:0]     out_X3,
    output logic [WIDTH-1:0]     out_Y3,
    output logic [WIDTH-1:0]     out_Z3,
    output logic                 out_special,

    output logic [$clog2(CTX_COUNT+1)-1:0] active_contexts
);

    localparam int CTX_W = (CTX_COUNT <= 1) ? 1 : $clog2(CTX_COUNT);

    localparam logic [255:0] P =
        256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    function automatic logic [255:0] field_add_mod(
        input logic [255:0] a,
        input logic [255:0] b
    );
        logic [256:0] sum;
        begin
            sum = {1'b0, a} + {1'b0, b};
            if (sum >= {1'b0, P})
                field_add_mod = sum[255:0] - P;
            else
                field_add_mod = sum[255:0];
        end
    endfunction

    function automatic logic [255:0] field_sub_mod(
        input logic [255:0] a,
        input logic [255:0] b
    );
        logic [256:0] diff;
        begin
            if (a >= b) begin
                field_sub_mod = a - b;
            end else begin
                diff = {1'b0, P} + {1'b0, a} - {1'b0, b};
                field_sub_mod = diff[255:0];
            end
        end
    endfunction

    function automatic logic [255:0] field_double_mod(
        input logic [255:0] a
    );
        begin
            field_double_mod = field_add_mod(a, a);
        end
    endfunction

    typedef enum logic [3:0] {
        C_FREE,
        C_R1_READY,
        C_R1_WAIT,
        C_R2_READY,
        C_R2_WAIT,
        C_R3_READY,
        C_R3_WAIT,
        C_R4_READY,
        C_R4_WAIT,
        C_R5_READY,
        C_R5_WAIT,
        C_R6_READY,
        C_R6_WAIT,
        C_DOUBLE_READY,
        C_DOUBLE_WAIT,
        C_DONE
    } ctx_state_t;

    typedef enum logic [2:0] {
        STAGE_NONE,
        STAGE_R1,
        STAGE_R2,
        STAGE_R3,
        STAGE_R4,
        STAGE_R5,
        STAGE_R6
    } stage_t;

    ctx_state_t ctx_state [0:CTX_COUNT-1];

    logic [TAG_W-1:0] ctx_tag [0:CTX_COUNT-1];
    logic             ctx_special [0:CTX_COUNT-1];

    logic [WIDTH-1:0] ctx_X1 [0:CTX_COUNT-1];
    logic [WIDTH-1:0] ctx_Y1 [0:CTX_COUNT-1];
    logic [WIDTH-1:0] ctx_Z1 [0:CTX_COUNT-1];
    logic [WIDTH-1:0] ctx_X2 [0:CTX_COUNT-1];
    logic [WIDTH-1:0] ctx_Y2 [0:CTX_COUNT-1];

    logic [WIDTH-1:0] ctx_Z1Z1 [0:CTX_COUNT-1];
    logic [WIDTH-1:0] ctx_U2   [0:CTX_COUNT-1];
    logic [WIDTH-1:0] ctx_Z1C  [0:CTX_COUNT-1];
    logic [WIDTH-1:0] ctx_S2   [0:CTX_COUNT-1];
    logic [WIDTH-1:0] ctx_H    [0:CTX_COUNT-1];
    logic [WIDTH-1:0] ctx_Rr   [0:CTX_COUNT-1];
    logic [WIDTH-1:0] ctx_HH   [0:CTX_COUNT-1];
    logic [WIDTH-1:0] ctx_RR   [0:CTX_COUNT-1];
    logic [WIDTH-1:0] ctx_HHH  [0:CTX_COUNT-1];
    logic [WIDTH-1:0] ctx_V    [0:CTX_COUNT-1];

    logic [WIDTH-1:0] ctx_X3 [0:CTX_COUNT-1];
    logic [WIDTH-1:0] ctx_Y3 [0:CTX_COUNT-1];
    logic [WIDTH-1:0] ctx_Z3 [0:CTX_COUNT-1];

    logic [CTX_W-1:0] alloc_rr;
    logic [CTX_W-1:0] sched_rr;
    logic [CTX_W-1:0] output_rr;

    logic             alloc_found;
    logic [CTX_W-1:0] alloc_idx;

    logic             sched_found;
    logic [CTX_W-1:0] sched_idx;
    stage_t           sched_stage;

    logic             output_found;
    logic [CTX_W-1:0] output_idx;

    logic             double_found;
    logic [CTX_W-1:0] double_idx;
    logic [CTX_W-1:0] double_rr;
    logic             double_start;
    logic             double_busy;
    logic             double_done;
    logic [CTX_W-1:0] double_active_ctx;
    logic [WIDTH-1:0] double_X3;
    logic [WIDTH-1:0] double_Y3;
    logic [WIDTH-1:0] double_Z3;

    logic             input_fire;
    logic             issue_fire;
    logic             output_fire;

    logic [3:0]       mul_in_valid;
    logic [WIDTH-1:0] mul_op_a [0:3];
    logic [WIDTH-1:0] mul_op_b [0:3];

    logic [3:0]       mul_ready;
    logic [3:0]       mul_out_valid;
    logic [WIDTH-1:0] mul_result [0:3];

    logic [3:0]       required_mul_mask;
    logic             required_muls_ready;

    // Metadata must be aligned to the multiplier's measured 16-cycle latency.
    // An issue accepted at cycle N is consumed with out_valid at cycle N+16.
    logic             meta_valid [0:MUL_LATENCY-1];
    logic [CTX_W-1:0] meta_ctx   [0:MUL_LATENCY-1];
    stage_t           meta_stage [0:MUL_LATENCY-1];

    integer alloc_scan_i;
    integer alloc_scan_idx;
    integer sched_scan_i;
    integer sched_scan_idx;
    integer output_scan_i;
    integer output_scan_idx;
    integer double_scan_i;
    integer double_scan_idx;
    integer count_i;

    // ------------------------------------------------------------------------
    // Free-context allocation
    // ------------------------------------------------------------------------
    always_comb begin
        alloc_found = 1'b0;
        alloc_idx   = '0;

        for (alloc_scan_i = 0; alloc_scan_i < CTX_COUNT; alloc_scan_i = alloc_scan_i + 1) begin
            alloc_scan_idx = alloc_rr + alloc_scan_i;
            if (alloc_scan_idx >= CTX_COUNT)
                alloc_scan_idx = alloc_scan_idx - CTX_COUNT;

            if (!alloc_found && ctx_state[alloc_scan_idx] == C_FREE) begin
                alloc_found = 1'b1;
                alloc_idx   = alloc_scan_idx[CTX_W-1:0];
            end
        end
    end

    assign in_ready  = alloc_found;
    assign input_fire = in_valid && in_ready;

    // ------------------------------------------------------------------------
    // Ready-stage scheduler
    // ------------------------------------------------------------------------
    always_comb begin
        sched_found = 1'b0;
        sched_idx   = '0;
        sched_stage = STAGE_NONE;

        for (sched_scan_i = 0; sched_scan_i < CTX_COUNT; sched_scan_i = sched_scan_i + 1) begin
            sched_scan_idx = sched_rr + sched_scan_i;
            if (sched_scan_idx >= CTX_COUNT)
                sched_scan_idx = sched_scan_idx - CTX_COUNT;

            if (!sched_found) begin
                case (ctx_state[sched_scan_idx])
                    C_R1_READY: begin
                        sched_found = 1'b1;
                        sched_idx   = sched_scan_idx[CTX_W-1:0];
                        sched_stage = STAGE_R1;
                    end
                    C_R2_READY: begin
                        sched_found = 1'b1;
                        sched_idx   = sched_scan_idx[CTX_W-1:0];
                        sched_stage = STAGE_R2;
                    end
                    C_R3_READY: begin
                        sched_found = 1'b1;
                        sched_idx   = sched_scan_idx[CTX_W-1:0];
                        sched_stage = STAGE_R3;
                    end
                    C_R4_READY: begin
                        sched_found = 1'b1;
                        sched_idx   = sched_scan_idx[CTX_W-1:0];
                        sched_stage = STAGE_R4;
                    end
                    C_R5_READY: begin
                        sched_found = 1'b1;
                        sched_idx   = sched_scan_idx[CTX_W-1:0];
                        sched_stage = STAGE_R5;
                    end
                    C_R6_READY: begin
                        sched_found = 1'b1;
                        sched_idx   = sched_scan_idx[CTX_W-1:0];
                        sched_stage = STAGE_R6;
                    end
                    default: begin
                    end
                endcase
            end
        end
    end

    always_comb begin
        required_mul_mask = 4'b0000;

        case (sched_stage)
            STAGE_R1: required_mul_mask = 4'b0001;
            STAGE_R2: required_mul_mask = 4'b0011;
            STAGE_R3: required_mul_mask = 4'b0001;
            STAGE_R4: required_mul_mask = 4'b0111;
            STAGE_R5: required_mul_mask = 4'b0011;
            STAGE_R6: required_mul_mask = 4'b0011;
            default:  required_mul_mask = 4'b0000;
        endcase
    end

    always_comb begin
        required_muls_ready = 1'b1;
        for (int m = 0; m < 4; m++) begin
            if (required_mul_mask[m] && !mul_ready[m])
                required_muls_ready = 1'b0;
        end
    end

    assign issue_fire = sched_found && required_muls_ready;

    // ------------------------------------------------------------------------
    // Multiplier issue bundle
    // ------------------------------------------------------------------------
    always_comb begin
        mul_in_valid = 4'b0000;

        for (int m = 0; m < 4; m++) begin
            mul_op_a[m] = '0;
            mul_op_b[m] = '0;
        end

        if (issue_fire) begin
            case (sched_stage)
                STAGE_R1: begin
                    mul_in_valid[0] = 1'b1;
                    mul_op_a[0] = ctx_Z1[sched_idx];
                    mul_op_b[0] = ctx_Z1[sched_idx];
                end

                STAGE_R2: begin
                    mul_in_valid[0] = 1'b1;
                    mul_op_a[0] = ctx_X2[sched_idx];
                    mul_op_b[0] = ctx_Z1Z1[sched_idx];

                    mul_in_valid[1] = 1'b1;
                    mul_op_a[1] = ctx_Z1[sched_idx];
                    mul_op_b[1] = ctx_Z1Z1[sched_idx];
                end

                STAGE_R3: begin
                    mul_in_valid[0] = 1'b1;
                    mul_op_a[0] = ctx_Y2[sched_idx];
                    mul_op_b[0] = ctx_Z1C[sched_idx];
                end

                STAGE_R4: begin
                    mul_in_valid[0] = 1'b1;
                    mul_op_a[0] = ctx_H[sched_idx];
                    mul_op_b[0] = ctx_H[sched_idx];

                    mul_in_valid[1] = 1'b1;
                    mul_op_a[1] = ctx_Rr[sched_idx];
                    mul_op_b[1] = ctx_Rr[sched_idx];

                    mul_in_valid[2] = 1'b1;
                    mul_op_a[2] = ctx_Z1[sched_idx];
                    mul_op_b[2] = ctx_H[sched_idx];
                end

                STAGE_R5: begin
                    mul_in_valid[0] = 1'b1;
                    mul_op_a[0] = ctx_H[sched_idx];
                    mul_op_b[0] = ctx_HH[sched_idx];

                    mul_in_valid[1] = 1'b1;
                    mul_op_a[1] = ctx_X1[sched_idx];
                    mul_op_b[1] = ctx_HH[sched_idx];
                end

                STAGE_R6: begin
                    mul_in_valid[0] = 1'b1;
                    mul_op_a[0] = ctx_Rr[sched_idx];
                    mul_op_b[0] = field_sub_mod(
                        ctx_V[sched_idx],
                        ctx_X3[sched_idx]
                    );

                    mul_in_valid[1] = 1'b1;
                    mul_op_a[1] = ctx_Y1[sched_idx];
                    mul_op_b[1] = ctx_HHH[sched_idx];
                end

                default: begin
                end
            endcase
        end
    end

    // ------------------------------------------------------------------------
    // Doubling scheduler
    //
    // v2 uses one verified jacobian_double_seq engine. Multiple contexts may
    // wait in C_DOUBLE_READY; they are served round-robin.
    // ------------------------------------------------------------------------
    always_comb begin
        double_found = 1'b0;
        double_idx   = '0;

        for (double_scan_i = 0;
             double_scan_i < CTX_COUNT;
             double_scan_i = double_scan_i + 1) begin

            double_scan_idx = double_rr + double_scan_i;
            if (double_scan_idx >= CTX_COUNT)
                double_scan_idx = double_scan_idx - CTX_COUNT;

            if (!double_found &&
                ctx_state[double_scan_idx] == C_DOUBLE_READY) begin
                double_found = 1'b1;
                double_idx   = double_scan_idx[CTX_W-1:0];
            end
        end
    end

    // Do not launch a new doubling operation in the same cycle in which the
    // previous operation asserts done. jacobian_double_seq does not accept a
    // new start while it is completing its S_DONE cycle. Without this guard,
    // the wrapper can mark the next context as C_DOUBLE_WAIT even though the
    // doubler ignored its start pulse.
    assign double_start = double_found && !double_busy && !double_done;

    // ------------------------------------------------------------------------
    // Completed-result output arbitration
    // ------------------------------------------------------------------------
    always_comb begin
        output_found = 1'b0;
        output_idx   = '0;

        for (output_scan_i = 0; output_scan_i < CTX_COUNT; output_scan_i = output_scan_i + 1) begin
            output_scan_idx = output_rr + output_scan_i;
            if (output_scan_idx >= CTX_COUNT)
                output_scan_idx = output_scan_idx - CTX_COUNT;

            if (!output_found && ctx_state[output_scan_idx] == C_DONE) begin
                output_found = 1'b1;
                output_idx   = output_scan_idx[CTX_W-1:0];
            end
        end
    end

    assign out_valid   = output_found;
    assign out_tag     = output_found ? ctx_tag[output_idx]     : '0;
    assign out_X3      = output_found ? ctx_X3[output_idx]      : '0;
    assign out_Y3      = output_found ? ctx_Y3[output_idx]      : '0;
    assign out_Z3      = output_found ? ctx_Z3[output_idx]      : '0;
    assign out_special = output_found ? ctx_special[output_idx] : 1'b0;

    assign output_fire = out_valid && out_ready;

    always_comb begin
        active_contexts = '0;
        for (count_i = 0; count_i < CTX_COUNT; count_i = count_i + 1) begin
            if (ctx_state[count_i] != C_FREE)
                active_contexts = active_contexts + 1'b1;
        end
    end

    // ------------------------------------------------------------------------
    // State, context, and metadata updates
    // ------------------------------------------------------------------------
    integer i;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            alloc_rr          <= '0;
            sched_rr          <= '0;
            output_rr         <= '0;
            double_rr         <= '0;
            double_active_ctx <= '0;

            for (i = 0; i < CTX_COUNT; i = i + 1) begin
                ctx_state[i]   <= C_FREE;
                ctx_tag[i]     <= '0;
                ctx_special[i] <= 1'b0;

                ctx_X1[i] <= '0;
                ctx_Y1[i] <= '0;
                ctx_Z1[i] <= '0;
                ctx_X2[i] <= '0;
                ctx_Y2[i] <= '0;

                ctx_Z1Z1[i] <= '0;
                ctx_U2[i]   <= '0;
                ctx_Z1C[i]  <= '0;
                ctx_S2[i]   <= '0;
                ctx_H[i]    <= '0;
                ctx_Rr[i]   <= '0;
                ctx_HH[i]   <= '0;
                ctx_RR[i]   <= '0;
                ctx_HHH[i]  <= '0;
                ctx_V[i]    <= '0;

                ctx_X3[i] <= '0;
                ctx_Y3[i] <= ONE_M;
                ctx_Z3[i] <= '0;
            end

            for (i = 0; i < MUL_LATENCY; i = i + 1) begin
                meta_valid[i] <= 1'b0;
                meta_ctx[i]   <= '0;
                meta_stage[i] <= STAGE_NONE;
            end
        end else begin
            // Metadata pipeline.
            for (i = MUL_LATENCY-1; i > 0; i = i - 1) begin
                meta_valid[i] <= meta_valid[i-1];
                meta_ctx[i]   <= meta_ctx[i-1];
                meta_stage[i] <= meta_stage[i-1];
            end

            meta_valid[0] <= issue_fire;
            meta_ctx[0]   <= sched_idx;
            meta_stage[0] <= sched_stage;

            // Release one completed context.
            if (output_fire) begin
                ctx_state[output_idx] <= C_FREE;
                output_rr             <= output_idx + 1'b1;
            end

            // Accept a new operation.
            if (input_fire) begin
                ctx_tag[alloc_idx]     <= in_tag;
                ctx_special[alloc_idx] <= 1'b0;

                ctx_X1[alloc_idx] <= in_X1;
                ctx_Y1[alloc_idx] <= in_Y1;
                ctx_Z1[alloc_idx] <= in_Z1;
                ctx_X2[alloc_idx] <= in_X2;
                ctx_Y2[alloc_idx] <= in_Y2;

                ctx_X3[alloc_idx] <= '0;
                ctx_Y3[alloc_idx] <= ONE_M;
                ctx_Z3[alloc_idx] <= '0;

                if (in_Z1 == '0) begin
                    ctx_X3[alloc_idx]    <= in_X2;
                    ctx_Y3[alloc_idx]    <= in_Y2;
                    ctx_Z3[alloc_idx]    <= ONE_M;
                    ctx_state[alloc_idx] <= C_DONE;
                end else begin
                    ctx_state[alloc_idx] <= C_R1_READY;
                end

                alloc_rr <= alloc_idx + 1'b1;
            end

            // Mark the issued context as waiting.
            if (issue_fire) begin
                case (sched_stage)
                    STAGE_R1: ctx_state[sched_idx] <= C_R1_WAIT;
                    STAGE_R2: ctx_state[sched_idx] <= C_R2_WAIT;
                    STAGE_R3: ctx_state[sched_idx] <= C_R3_WAIT;
                    STAGE_R4: ctx_state[sched_idx] <= C_R4_WAIT;
                    STAGE_R5: ctx_state[sched_idx] <= C_R5_WAIT;
                    STAGE_R6: ctx_state[sched_idx] <= C_R6_WAIT;
                    default: begin
                    end
                endcase

                sched_rr <= sched_idx + 1'b1;
            end

            // Start one queued doubling operation.
            if (double_start) begin
                double_active_ctx       <= double_idx;
                ctx_state[double_idx]   <= C_DOUBLE_WAIT;
                double_rr               <= double_idx + 1'b1;
            end

            // Capture the verified doubling result.
            if (double_done) begin
                ctx_X3[double_active_ctx]      <= double_X3;
                ctx_Y3[double_active_ctx]      <= double_Y3;
                ctx_Z3[double_active_ctx]      <= double_Z3;
                ctx_special[double_active_ctx] <= 1'b0;
                ctx_state[double_active_ctx]   <= C_DONE;
            end

            // Consume a completed multiplier bundle.
            if (meta_valid[MUL_LATENCY-1] && mul_out_valid[0]) begin
                case (meta_stage[MUL_LATENCY-1])
                    STAGE_R1: begin
                        ctx_Z1Z1[meta_ctx[MUL_LATENCY-1]] <= mul_result[0];
                        ctx_state[meta_ctx[MUL_LATENCY-1]] <= C_R2_READY;
                    end

                    STAGE_R2: begin
                        ctx_U2[meta_ctx[MUL_LATENCY-1]]  <= mul_result[0];
                        ctx_Z1C[meta_ctx[MUL_LATENCY-1]] <= mul_result[1];
                        ctx_state[meta_ctx[MUL_LATENCY-1]] <= C_R3_READY;
                    end

                    STAGE_R3: begin
                        ctx_S2[meta_ctx[MUL_LATENCY-1]] <= mul_result[0];

                        ctx_H[meta_ctx[MUL_LATENCY-1]] <= field_sub_mod(
                            ctx_U2[meta_ctx[MUL_LATENCY-1]],
                            ctx_X1[meta_ctx[MUL_LATENCY-1]]
                        );

                        ctx_Rr[meta_ctx[MUL_LATENCY-1]] <= field_sub_mod(
                            mul_result[0],
                            ctx_Y1[meta_ctx[MUL_LATENCY-1]]
                        );

                        if (field_sub_mod(
                                ctx_U2[meta_ctx[MUL_LATENCY-1]],
                                ctx_X1[meta_ctx[MUL_LATENCY-1]]
                            ) == '0) begin

                            if (field_sub_mod(
                                    mul_result[0],
                                    ctx_Y1[meta_ctx[MUL_LATENCY-1]]
                                ) == '0) begin
                                // Same point: route to verified doubling unit.
                                ctx_special[meta_ctx[MUL_LATENCY-1]] <= 1'b0;
                                ctx_state[meta_ctx[MUL_LATENCY-1]]
                                    <= C_DOUBLE_READY;
                            end else begin
                                // Opposite points: valid infinity result.
                                ctx_X3[meta_ctx[MUL_LATENCY-1]] <= '0;
                                ctx_Y3[meta_ctx[MUL_LATENCY-1]] <= ONE_M;
                                ctx_Z3[meta_ctx[MUL_LATENCY-1]] <= '0;
                                ctx_special[meta_ctx[MUL_LATENCY-1]] <= 1'b0;
                                ctx_state[meta_ctx[MUL_LATENCY-1]] <= C_DONE;
                            end
                        end else begin
                            ctx_state[meta_ctx[MUL_LATENCY-1]] <= C_R4_READY;
                        end
                    end

                    STAGE_R4: begin
                        ctx_HH[meta_ctx[MUL_LATENCY-1]] <= mul_result[0];
                        ctx_RR[meta_ctx[MUL_LATENCY-1]] <= mul_result[1];
                        ctx_Z3[meta_ctx[MUL_LATENCY-1]] <= mul_result[2];
                        ctx_state[meta_ctx[MUL_LATENCY-1]] <= C_R5_READY;
                    end

                    STAGE_R5: begin
                        ctx_HHH[meta_ctx[MUL_LATENCY-1]] <= mul_result[0];
                        ctx_V[meta_ctx[MUL_LATENCY-1]]   <= mul_result[1];

                        ctx_X3[meta_ctx[MUL_LATENCY-1]] <= field_sub_mod(
                            field_sub_mod(
                                ctx_RR[meta_ctx[MUL_LATENCY-1]],
                                mul_result[0]
                            ),
                            field_double_mod(mul_result[1])
                        );

                        ctx_state[meta_ctx[MUL_LATENCY-1]] <= C_R6_READY;
                    end

                    STAGE_R6: begin
                        ctx_Y3[meta_ctx[MUL_LATENCY-1]] <= field_sub_mod(
                            mul_result[0],
                            mul_result[1]
                        );

                        ctx_special[meta_ctx[MUL_LATENCY-1]] <= 1'b0;
                        ctx_state[meta_ctx[MUL_LATENCY-1]] <= C_DONE;
                    end

                    default: begin
                    end
                endcase
            end
        end
    end

    // ------------------------------------------------------------------------
    // Verified sequential Jacobian doubler for H=0, R=0 contexts
    // ------------------------------------------------------------------------
    jacobian_double_seq #(
        .WIDTH(WIDTH)
    ) u_double (
        .clk   (clk),
        .rst_n (rst_n),
        .start (double_start),

        .X1    (ctx_X1[double_idx]),
        .Y1    (ctx_Y1[double_idx]),
        .Z1    (ctx_Z1[double_idx]),

        .busy  (double_busy),
        .done  (double_done),

        .X3    (double_X3),
        .Y3    (double_Y3),
        .Z3    (double_Z3)
    );

    // ------------------------------------------------------------------------
    // Four fully pipelined Montgomery multipliers
    // ------------------------------------------------------------------------
    genvar g;
    generate
        for (g = 0; g < 4; g = g + 1) begin : GEN_MONT_MUL
            secp256k1_montgomery_mul #(
                .WIDTH(WIDTH)
            ) u_mont_mul (
                .clk       (clk),
                .rst_n     (rst_n),
                .in_valid  (mul_in_valid[g]),
                .op_a      (mul_op_a[g]),
                .op_b      (mul_op_b[g]),
                .out_valid (mul_out_valid[g]),
                .result    (mul_result[g]),
                .ready     (mul_ready[g])
            );
        end
    endgenerate

endmodule