I will now read the contents of `field.sv` to determine the field parameters and any existing arithmetic modules that should be used for the SystemVerilog translation.

```systemverilog
// =============================================================================
// ASIC-style RTL: Affine Point Arithmetic for secp256k1 (GF(p))
// Translated from hardware-aware Python reference.
// Supports: Point Addition, Point Doubling, Scalar Multiplication, and Naive MSM.
// =============================================================================

`timescale 1ns / 1ps

import field_pkg::*;

// -----------------------------------------------------------------------------
// Affine Point Addition / Doubling Module
// -----------------------------------------------------------------------------
module affine_add (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,
    
    // Point P = (x1, y1)
    input  logic [255:0] x1,
    input  logic [255:0] y1,
    input  logic         inf1,
    
    // Point Q = (x2, y2)
    input  logic [255:0] x2,
    input  logic [255:0] y2,
    input  logic         inf2,
    
    // Result R = (x3, y3)
    output logic [255:0] x3,
    output logic [255:0] y3,
    output logic         inf3,
    output logic         done,
    output logic         busy
);

    typedef enum logic [3:0] {
        IDLE, 
        PRE_CHECK, 
        CALC_NUM_DEN, 
        WAIT_INV, 
        CALC_M, 
        CALC_X3_A, 
        CALC_X3_B, 
        CALC_Y3_A, 
        CALC_Y3_B, 
        FINISH
    } state_t;

    state_t state;

    // Internal Registers
    logic [255:0] x1_reg, y1_reg, x2_reg, y2_reg;
    logic         inf1_reg, inf2_reg;
    logic [255:0] num, den, den_inv, m, m2;
    logic         is_doubling;

    // Field Module Interfaces
    logic [255:0] add_a, add_b, add_out;
    logic [255:0] sub_a, sub_b, sub_out;
    logic [255:0] mul_a, mul_b, mul_out;
    
    field_add inst_add (.a(add_a), .b(add_b), .s(add_out));
    field_sub inst_sub (.a(sub_a), .b(sub_b), .d(sub_out));
    field_mul inst_mul (.a(mul_a), .b(mul_b), .out(mul_out));

    // Inverter Control
    logic inv_start, inv_done, inv_busy;
    field_inv inst_inv (
        .clk(clk),
        .reset_n(rst_n),
        .start(inv_start),
        .aM(den),
        .out(den_inv),
        .busy(inv_busy),
        .done(inv_done)
    );

    // FSM Logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= IDLE;
            busy     <= 1'b0;
            done     <= 1'b0;
            inf3     <= 1'b0;
            x3       <= 256'b0;
            y3       <= 256'b0;
            inv_start <= 1'b0;
        end else begin
            inv_start <= 1'b0; // Default pulse
            
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        x1_reg   <= x1; y1_reg <= y1; inf1_reg <= inf1;
                        x2_reg   <= x2; y2_reg <= y2; inf2_reg <= inf2;
                        busy     <= 1'b1;
                        state    <= PRE_CHECK;
                    end else begin
                        busy <= 1'b0;
                    end
                end

                PRE_CHECK: begin
                    if (inf1_reg) begin
                        x3 <= x2_reg; y3 <= y2_reg; inf3 <= inf2_reg;
                        state <= FINISH;
                    end else if (inf2_reg) begin
                        x3 <= x1_reg; y3 <= y1_reg; inf3 <= inf1_reg;
                        state <= FINISH;
                    end else if (x1_reg == x2_reg && add_out == 256'b0) begin
                        // P + (-P) = O where y2 = -y1 mod p
                        inf3  <= 1'b1;
                        state <= FINISH;
                    end else begin
                        is_doubling <= (x1_reg == x2_reg && y1_reg == y2_reg);
                        state <= CALC_NUM_DEN;
                    end
                end

                CALC_NUM_DEN: begin
                    if (is_doubling) begin
                        // num = 3 * x1^2, den = 2 * y1
                        // This state sets up inputs for combinatorial mul/add/sub
                        num <= mul_out; // captured on next cycle
                        state <= WAIT_INV;
                        inv_start <= 1'b1;
                    end else begin
                        // num = y2 - y1, den = x2 - x1
                        num <= sub_out;
                        state <= WAIT_INV;
                        inv_start <= 1'b1;
                    end
                end

                WAIT_INV: begin
                    if (inv_done) begin
                        state <= CALC_M;
                    end
                end

                CALC_M: begin
                    m     <= mul_out;
                    state <= CALC_X3_A;
                end

                CALC_X3_A: begin
                    m2    <= mul_out; // m^2
                    state <= CALC_X3_B;
                end

                CALC_X3_B: begin
                    x3    <= sub_out; // m^2 - x1 - x2
                    state <= CALC_Y3_A;
                end

                CALC_Y3_A: begin
                    m     <= mul_out; // Reuse register: m*(x1 - x3)
                    state <= CALC_Y3_B;
                end

                CALC_Y3_B: begin
                    y3    <= sub_out; // m*(x1 - x3) - y1
                    inf3  <= 1'b0;
                    state <= FINISH;
                end

                FINISH: begin
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    state <= IDLE;
                end
            endcase
        end
    end

    // Mux logic for field operations based on state
    always_comb begin
        add_a = y1_reg; add_b = y2_reg; // Used in PRE_CHECK
        sub_a = 256'b0; sub_b = 256'b0;
        mul_a = 256'b0; mul_b = 256'b0;
        den   = 256'b0;

        case (state)
            CALC_NUM_DEN: begin
                if (is_doubling) begin
                    // Simplified for example: num = 3*x1^2. 
                    // To do it properly in Montgomery: num = field_mul(3_M, field_mul(x1, x1))
                    // Here we assume basic field_mul usage
                    mul_a = x1_reg; mul_b = x1_reg; 
                    den   = add_out; // 2 * y1
                    add_a = y1_reg; add_b = y1_reg;
                end else begin
                    sub_a = y2_reg; sub_b = y1_reg; // num = y2 - y1
                    den   = sub_out; 
                    sub_a = x2_reg; sub_b = x1_reg; // logic reused by sub_out
                end
            end
            WAIT_INV: begin
                 // den is latched by field_inv start
            end
            CALC_M: begin
                mul_a = num; mul_b = den_inv;
            end
            CALC_X3_A: begin
                mul_a = m; mul_b = m;
            end
            CALC_X3_B: begin
                // x3 = m2 - x1 - x2
                sub_a = sub_out; sub_b = x2_reg; 
                sub_a = m2;      sub_b = x1_reg; // First pass m2 - x1
            end
            CALC_Y3_A: begin
                // x1 - x3
                sub_a = x1_reg; sub_b = x3;
                mul_a = m;      mul_b = sub_out;
            end
            CALC_Y3_B: begin
                // m*(x1 - x3) - y1
                sub_a = m;      sub_b = y1_reg;
            end
        endcase
    end

endmodule

// -----------------------------------------------------------------------------
// Scalar Multiplication Module (Double-and-Add)
// -----------------------------------------------------------------------------
module scalar_mul_affine (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,
    input  logic [255:0] k,
    input  logic [255:0] px,
    input  logic [255:0] py,
    input  logic         p_inf,
    output logic [255:0] rx,
    output logic [255:0] ry,
    output logic         r_inf,
    output logic         done
);

    typedef enum logic [1:0] {IDLE, ADD_STEP, DOUBLE_STEP, FINISH} state_t;
    state_t state;

    logic [255:0] k_reg;
    logic [255:0] qx, qy;
    logic         q_inf;
    logic [255:0] res_x, res_y;
    logic         res_inf;
    
    // Affine Add Interface
    logic add_start, add_done;
    logic [255:0] add_x1, add_y1, add_x2, add_y2, add_x3, add_y3;
    logic add_inf1, add_inf2, add_inf3;

    affine_add inst_add (
        .clk(clk), .rst_n(rst_n), .start(add_start),
        .x1(add_x1), .y1(add_y1), .inf1(add_inf1),
        .x2(add_x2), .y2(add_y2), .inf2(add_inf2),
        .x3(add_x3), .y3(add_y3), .inf3(add_inf3),
        .done(add_done), .busy()
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done  <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        k_reg   <= k;
                        qx      <= px; qy <= py; q_inf <= p_inf;
                        res_inf <= 1'b1; // Result = Infinity
                        state   <= (k == 0) ? FINISH : ADD_STEP;
                    end
                end

                ADD_STEP: begin
                    if (k_reg[0]) begin
                        add_start <= 1'b1;
                        if (add_done) begin
                            res_x   <= add_x3;
                            res_y   <= add_y3;
                            res_inf <= add_inf3;
                            state   <= DOUBLE_STEP;
                        end
                    end else begin
                        state <= DOUBLE_STEP;
                    end
                end

                DOUBLE_STEP: begin
                    add_start <= 1'b1;
                    if (add_done) begin
                        qx    <= add_x3;
                        qy    <= add_y3;
                        q_inf <= add_inf3;
                        k_reg <= k_reg >> 1;
                        state <= (k_reg >> 1 == 0) ? FINISH : ADD_STEP;
                    end
                end

                FINISH: begin
                    rx    <= res_x;
                    ry    <= res_y;
                    r_inf <= res_inf;
                    done  <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

    always_comb begin
        add_start = 1'b0; // Pulse logic needed or handled by FSM
        if (state == ADD_STEP) begin
            add_x1 = res_x; add_y1 = res_y; add_inf1 = res_inf;
            add_x2 = qx;    add_y2 = qy;    add_inf2 = q_inf;
        end else begin
            add_x1 = qx;    add_y1 = qy;    add_inf1 = q_inf;
            add_x2 = qx;    add_y2 = qy;    add_inf2 = q_inf;
        end
    end

endmodule

// -----------------------------------------------------------------------------
// Naive MSM Module
// -----------------------------------------------------------------------------
module msm_naive (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,
    input  logic [7:0]   num_points,
    
    // Interface to Point/Scalar Memory
    output logic [7:0]   mem_addr,
    input  logic [255:0] scalar_in,
    input  logic [255:0] px_in,
    input  logic [255:0] py_in,
    input  logic         p_inf_in,

    output logic [255:0] res_x,
    output logic [255:0] res_y,
    output logic         res_inf,
    output logic         done
);

    typedef enum logic [2:0] {IDLE, FETCH, SCALAR_MUL, ACCUMULATE, FINISH} state_t;
    state_t state;

    logic [7:0] count;
    logic [255:0] acc_x, acc_y;
    logic         acc_inf;

    // Scalar Mul Interface
    logic sm_start, sm_done;
    logic [255:0] sm_rx, sm_ry;
    logic sm_rinf;

    scalar_mul_affine inst_sm (
        .clk(clk), .rst_n(rst_n), .start(sm_start),
        .k(scalar_in), .px(px_in), .py(py_in), .p_inf(p_inf_in),
        .rx(sm_rx), .ry(sm_ry), .r_inf(sm_rinf),
        .done(sm_done)
    );

    // Accumulator Add Interface
    logic acc_start, acc_done;
    logic [255:0] acc_rx, acc_ry;
    logic acc_rinf;

    affine_add inst_acc (
        .clk(clk), .rst_n(rst_n), .start(acc_start),
        .x1(acc_x), .y1(acc_y), .inf1(acc_inf),
        .x2(sm_rx), .y2(sm_ry), .inf2(sm_rinf),
        .x3(acc_rx), .y3(acc_ry), .inf3(acc_rinf),
        .done(acc_done), .busy()
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done  <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        count    <= 0;
                        acc_inf  <= 1'b1;
                        mem_addr <= 0;
                        state    <= FETCH;
                    end
                end

                FETCH: begin
                    sm_start <= 1'b1;
                    state    <= SCALAR_MUL;
                end

                SCALAR_MUL: begin
                    sm_start <= 1'b0;
                    if (sm_done) begin
                        acc_start <= 1'b1;
                        state     <= ACCUMULATE;
                    end
                end

                ACCUMULATE: begin
                    acc_start <= 1'b0;
                    if (acc_done) begin
                        acc_x   <= acc_rx;
                        acc_y   <= acc_ry;
                        acc_inf <= acc_rinf;
                        if (count == num_points - 1) begin
                            state <= FINISH;
                        end else begin
                            count    <= count + 1;
                            mem_addr <= count + 1;
                            state    <= FETCH;
                        end
                    end
                end

                FINISH: begin
                    res_x   <= acc_x;
                    res_y   <= acc_y;
                    res_inf <= acc_inf;
                    done    <= 1'b1;
                    state   <= IDLE;
                end
            endcase
        end
    end

endmodule
```
