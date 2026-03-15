```systemverilog
`timescale 1ns / 1ps

/**
 * ============================================================================
 * Module: main
 * Description: MSM Benchmark Controller (Montgomery-safe + fair counters)
 * 
 * This module translates the high-level Python benchmark logic into a 
 * synthesizable SystemVerilog FSM that orchestrates multiple MSM algorithms,
 * captures their operation counts, validates correctness, and calculates
 * weighted field-op costs.
 * ============================================================================
 */

module main #(
    parameter int W             = 8,
    parameter int HW_TILE_SIZE  = 4,
    parameter int HW_BATCH_SIZE = 1024,
    parameter int SCALAR_BITS   = 256,
    parameter int MAX_N         = 4096
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,

    // Input data interfaces (Scalars and Points)
    // In a full system, these would be fetched from memory/DMA
    input  logic [SCALAR_BITS-1:0] scalars [MAX_N],
    input  struct packed {
        logic [SCALAR_BITS-1:0] x;
        logic [SCALAR_BITS-1:0] y;
    } points [MAX_N],

    // Benchmark Control and Status
    output logic done,
    output logic all_match,
    
    // Performance Metrics (Weighted Costs for the current execution)
    output logic [63:0] cost_naive,
    output logic [63:0] cost_ref,
    output logic [63:0] cost_pip,
    output logic [63:0] cost_pip_hw,
    output logic [63:0] cost_ext
);

    // ------------------------------------------------------------------------
    // Parameters and Constants
    // ------------------------------------------------------------------------
    
    // N_LIST = [2, 16, 256, 1024, 4096]
    localparam int N_ENTRIES = 5;
    const int N_LIST[N_ENTRIES] = '{2, 16, 256, 1024, 4096};

    // Weighted cost model (ASIC-like)
    // WEIGHTS = {"mul": 1.0, "add": 0.1, "sub": 0.1, "inv": 80.0}
    // We scale by 10 to use integer arithmetic: 1.0 -> 10, 0.1 -> 1, 80.0 -> 800
    localparam int W_MUL = 10;
    localparam int W_ADD = 1;
    localparam int W_SUB = 1;
    localparam int W_INV = 800;

    // ------------------------------------------------------------------------
    // Type Definitions
    // ------------------------------------------------------------------------
    
    typedef struct packed {
        logic [SCALAR_BITS-1:0] x;
        logic [SCALAR_BITS-1:0] y;
    } point_affine_t;

    // Operation counters structure mapping Python op_counter fields
    typedef struct packed {
        logic [31:0] mul;
        logic [31:0] add;
        logic [31:0] sub;
        logic [31:0] inv;
        logic [31:0] aff;
        logic [31:0] jac;
        logic [31:0] mix;
        logic [31:0] dbl;
        logic [31:0] ext_add;
        logic [31:0] ext_mix;
        logic [31:0] ext_dbl;
    } op_counts_t;

    // ------------------------------------------------------------------------
    // State Machine
    // ------------------------------------------------------------------------
    
    typedef enum logic [3:0] {
        ST_IDLE,
        ST_INIT_N,
        ST_RUN_NAIVE,
        ST_RUN_REFERENCE,
        ST_RUN_PIPPENGER,
        ST_RUN_PIPPENGER_HW,
        ST_RUN_EXTENDED,
        ST_VALIDATE,
        ST_NEXT_N,
        ST_DONE
    } state_e;

    state_e state;
    logic [2:0] n_ptr;
    logic [31:0] current_n;

    // ------------------------------------------------------------------------
    // Internal Registers and Handshaking
    // ------------------------------------------------------------------------
    
    point_affine_t results [5]; // Storage for results from 5 models
    op_counts_t    counts  [5]; // Storage for operation counts

    // Control signals for sub-modules
    logic [4:0] msm_start_bus;
    logic [4:0] msm_done_bus;
    point_affine_t msm_res_bus [5];
    op_counts_t    msm_cnt_bus [5];

    // ------------------------------------------------------------------------
    // Datapath Logic: Weighted Cost Calculation
    // ------------------------------------------------------------------------
    
    // Purely combinational function to compute weighted cost
    function automatic logic [63:0] get_cost(input op_counts_t c);
        // cost = (mul*1.0 + add*0.1 + sub*0.1 + inv*80.0)
        // Scaled by 10: (mul*10 + add*1 + sub*1 + inv*800)
        return (64'(c.mul) * W_MUL + 64'(c.add) * W_ADD + 64'(c.sub) * W_SUB + 64'(c.inv) * W_INV);
    endfunction

    // Continuous assignment for outputs based on captured counts
    always_comb begin
        cost_naive  = get_cost(counts[0]) / 10;
        cost_ref    = get_cost(counts[1]) / 10;
        cost_pip    = get_cost(counts[2]) / 10;
        cost_pip_hw = get_cost(counts[3]) / 10;
        cost_ext    = get_cost(counts[4]) / 10;
    end

    // ------------------------------------------------------------------------
    // Sequential Control Logic (FSM)
    // ------------------------------------------------------------------------
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= ST_IDLE;
            n_ptr         <= 0;
            current_n     <= 0;
            done          <= 0;
            all_match     <= 1;
            msm_start_bus <= 5'b0;
            
            for (int i=0; i<5; i++) begin
                results[i] <= '0;
                counts[i]  <= '0;
            end
        end else begin
            case (state)
                ST_IDLE: begin
                    done <= 0;
                    if (start) begin
                        n_ptr     <= 0;
                        all_match <= 1;
                        state     <= ST_INIT_N;
                    end
                end

                ST_INIT_N: begin
                    current_n <= N_LIST[n_ptr];
                    state     <= ST_RUN_NAIVE;
                end

                // Sequential execution of MSM algorithms as per Python main() logic
                ST_RUN_NAIVE: begin
                    msm_start_bus[0] <= 1;
                    if (msm_done_bus[0]) begin
                        results[0]       <= msm_res_bus[0];
                        counts[0]        <= msm_cnt_bus[0];
                        msm_start_bus[0] <= 0;
                        state            <= ST_RUN_REFERENCE;
                    end
                end

                ST_RUN_REFERENCE: begin
                    msm_start_bus[1] <= 1;
                    if (msm_done_bus[1]) begin
                        results[1]       <= msm_res_bus[1];
                        counts[1]        <= msm_cnt_bus[1];
                        msm_start_bus[1] <= 0;
                        state            <= ST_RUN_PIPPENGER;
                    end
                end

                ST_RUN_PIPPENGER: begin
                    msm_start_bus[2] <= 1;
                    if (msm_done_bus[2]) begin
                        results[2]       <= msm_res_bus[2];
                        counts[2]        <= msm_cnt_bus[2];
                        msm_start_bus[2] <= 0;
                        state            <= ST_RUN_PIPPENGER_HW;
                    end
                end

                ST_RUN_PIPPENGER_HW: begin
                    msm_start_bus[3] <= 1;
                    if (msm_done_bus[3]) begin
                        results[3]       <= msm_res_bus[3];
                        counts[3]        <= msm_cnt_bus[3];
                        msm_start_bus[3] <= 0;
                        state            <= ST_RUN_EXTENDED;
                    end
                end

                ST_RUN_EXTENDED: begin
                    msm_start_bus[4] <= 1;
                    if (msm_done_bus[4]) begin
                        results[4]       <= msm_res_bus[4];
                        counts[4]        <= msm_cnt_bus[4];
                        msm_start_bus[4] <= 0;
                        state            <= ST_VALIDATE;
                    end
                end

                ST_VALIDATE: begin
                    // assert (Naive == Reference == Pippenger == Pippenger_HW == Extended)
                    if ((results[0] != results[1]) || (results[0] != results[2]) || 
                        (results[0] != results[3]) || (results[0] != results[4])) begin
                        all_match <= 0;
                    end
                    state <= ST_NEXT_N;
                end

                ST_NEXT_N: begin
                    if (n_ptr == N_ENTRIES - 1)
                        state <= ST_DONE;
                    else begin
                        n_ptr <= n_ptr + 1;
                        state <= ST_INIT_N;
                    end
                end

                ST_DONE: begin
                    done <= 1;
                    if (!start) state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    // ------------------------------------------------------------------------
    // MSM Algorithm Core Instantiations
    // ------------------------------------------------------------------------
    
    // Model 0: Naive (Golden Affine)
    msm_naive i_msm_naive (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (msm_start_bus[0]),
        .n       (current_n),
        .scalars (scalars),
        .points  (points),
        .done    (msm_done_bus[0]),
        .result  (msm_res_bus[0]),
        .counts  (msm_cnt_bus[0])
    );

    // Model 1: Reference (Jacobian Montgomery)
    msm_reference #( .W(W) ) i_msm_ref (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (msm_start_bus[1]),
        .n       (current_n),
        .scalars (scalars),
        .points  (points),
        .done    (msm_done_bus[1]),
        .result  (msm_res_bus[1]),
        .counts  (msm_cnt_bus[1])
    );

    // Model 2: Pippenger (Jacobian Montgomery)
    msm_pippenger #( .W(W) ) i_msm_pip (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (msm_start_bus[2]),
        .n       (current_n),
        .scalars (scalars),
        .points  (points),
        .done    (msm_done_bus[2]),
        .result  (msm_res_bus[2]),
        .counts  (msm_cnt_bus[2])
    );

    // Model 3: Pippenger HW-like (Tiled + Ping-Pong)
    msm_pippenger_hw #(
        .W(W),
        .TILE_SIZE(HW_TILE_SIZE),
        .BATCH_SIZE(HW_BATCH_SIZE)
    ) i_msm_pip_hw (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (msm_start_bus[3]),
        .n       (current_n),
        .scalars (scalars),
        .points  (points),
        .done    (msm_done_bus[3]),
        .result  (msm_res_bus[3]),
        .counts  (msm_cnt_bus[3])
    );

    // Model 4: Extended (Extended Montgomery)
    msm_extended #( .W(W) ) i_msm_ext (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (msm_start_bus[4]),
        .n       (current_n),
        .scalars (scalars),
        .points  (points),
        .done    (msm_done_bus[4]),
        .result  (msm_res_bus[4]),
        .counts  (msm_cnt_bus[4])
    );

endmodule
```
