// jacobian.sv
// Expert RTL Design: Jacobian Point Arithmetic Module
// Translated from hardware-aware Python code.
// Logic: Separated into always_ff (state, data) and always_comb (next_state, ALU steering).

`timescale 1ns / 1ps

module jacobian_unit #(
    parameter int WIDTH = 384
)(
    input  logic              clk,
    input  logic              rst_n,

    // Command Interface
    input  logic              start,
    input  logic [1:0]        op, // 0: Double, 1: Mixed Add (Z2=1), 2: Full Add

    // Inputs (P and Q)
    input  logic [WIDTH-1:0]  x1, y1, z1,
    input  logic [WIDTH-1:0]  x2, y2, z2,

    // Outputs
    output logic              done,
    output logic [WIDTH-1:0]  x_out, y_out, z_out,
    output logic              is_inf,

    // Field Arithmetic Interface (External)
    output logic              fmul_start,
    output logic [WIDTH-1:0]  fmul_a,
    output logic [WIDTH-1:0]  fmul_b,
    input  logic              fmul_ready,
    input  logic [WIDTH-1:0]  fmul_res,

    output logic [WIDTH-1:0]  fadd_a, fadd_b,
    input  logic [WIDTH-1:0]  fadd_res,
    output logic [WIDTH-1:0]  fsub_a, fsub_b,
    input  logic [WIDTH-1:0]  fsub_res
);

    // --- State Machine Definition ---
    typedef enum logic [5:0] {
        IDLE,
        // Doubling sequence
        DBL_Y1_SQ, DBL_S_PRE, DBL_X1_SQ, DBL_M_SQ, DBL_X3_PRE, DBL_Y1_SQ_SQ, DBL_Y3_PRE, DBL_Z3,
        // Addition sequence
        ADD_Z2_SQ, ADD_U1, ADD_Z1_SQ, ADD_U2, ADD_Z2_CU, ADD_S1, ADD_Z1_CU, ADD_S2,
        ADD_H_SQ, ADD_H_CU, ADD_U1H2, ADD_X3_PRE, ADD_Y3_PRE, ADD_Z3_PRE, ADD_Z3_FINAL,
        FINISH
    } state_t;

    state_t state, next_state;

    // --- Data Registers ---
    logic [WIDTH-1:0] r_x1, r_y1, r_z1;
    logic [WIDTH-1:0] r_x2, r_y2, r_z2;
    
    // Intermediate Results (mapping to Python variables)
    logic [WIDTH-1:0] r_y1_sq, r_s, r_x1_sq, r_m, r_x3, r_y3, r_z3;
    logic [WIDTH-1:0] r_z1_sq, r_z2_sq, r_u1, r_u2, r_s1, r_s2;
    logic [WIDTH-1:0] r_h, r_rr, r_h_sq, r_h_cu, r_u1h2;

    // --- Sequential Logic: State and Data ---
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            is_inf <= 1'b0;
            {r_x3, r_y3, r_z3} <= '0;
        end else begin
            state <= next_state;

            if (state == IDLE && start) begin
                r_x1 <= x1; r_y1 <= y1; r_z1 <= z1;
                r_x2 <= x2; r_y2 <= y2; r_z2 <= z2;
                is_inf <= 1'b0;
            end

            if (fmul_ready) begin
                case (state)
                    // Doubling Path
                    DBL_Y1_SQ:    r_y1_sq <= fmul_res;
                    DBL_S_PRE:    r_s     <= {fmul_res[WIDTH-3:0], 2'b0}; // S = 4 * X1 * Y1_sq
                    DBL_X1_SQ:    r_x1_sq <= fmul_res;
                    DBL_M_SQ:     r_x3    <= fsub_res; // X3 = M^2 - 2*S
                    DBL_Y1_SQ_SQ: r_y3    <= fsub_res; // Y3 = M*(S-X3) - 8*Y1_sq_sq
                    DBL_Z3:       r_z3    <= fmul_res;

                    // Addition Path
                    ADD_Z2_SQ:    r_z2_sq <= fmul_res;
                    ADD_U1:       r_u1    <= fmul_res;
                    ADD_Z1_SQ:    r_z1_sq <= fmul_res;
                    ADD_U2:       r_u2    <= fmul_res;
                    ADD_Z2_CU:    r_s1    <= fmul_res;
                    ADD_Z1_CU:    r_s2    <= fmul_res;
                    ADD_H_SQ:     r_h_sq  <= fmul_res;
                    ADD_H_CU:     r_h_cu  <= fmul_res;
                    ADD_U1H2:     r_u1h2  <= fmul_res;
                    ADD_X3_PRE:   r_x3    <= fsub_res; // X3 = Rr^2 - H_cu - 2*U1H2
                    ADD_Y3_PRE:   r_y3    <= fsub_res; // Y3 = Rr*(U1H2-X3) - S1*H_cu
                    ADD_Z3_FINAL: r_z3    <= fmul_res;
                    default: ;
                endcase
            end

            // Capture Subtractors/Adders that are used between muls
            case (state)
                DBL_X1_SQ: begin
                    // M = 3 * X1_sq = X1_sq + (X1_sq + X1_sq)
                    r_m <= fadd_res; 
                end
                ADD_U2: begin
                    r_h  <= fsub_res; // H = U2 - U1
                end
                ADD_S2: begin
                    r_rr <= fsub_res; // Rr = S2 - S1
                    if (r_u1 == r_u2 && r_s1 != r_s2) is_inf <= 1'b1;
                end
                default: ;
            endcase
        end
    end

    // --- Combinational Logic: Next State and ALU Control ---
    always_comb begin
        next_state = state;
        fmul_start = 1'b0;
        fmul_a = '0; fmul_b = '0;
        fadd_a = '0; fadd_b = '0;
        fsub_a = '0; fsub_b = '0;
        done = 1'b0;

        case (state)
            IDLE: if (start) begin
                if (z1 == '0 || (op == 2'b00 && y1 == '0)) next_state = FINISH; // P is INF or Double(P) where P.y=0
                else if (op == 2'b10 && z2 == '0) next_state = FINISH;           // Add and Q is INF
                else if (op == 2'b00) next_state = DBL_Y1_SQ;
                else if (op == 2'b01) next_state = ADD_Z1_SQ; // Mixed Add (Z2=1 assumed in inputs)
                else next_state = ADD_Z2_SQ;                  // Full Add
            end

            // --- Jacobian Double ---
            DBL_Y1_SQ: begin
                fmul_start = 1'b1; fmul_a = r_y1; fmul_b = r_y1;
                if (fmul_ready) next_state = DBL_S_PRE;
            end
            DBL_S_PRE: begin
                fmul_start = 1'b1; fmul_a = r_x1; fmul_b = r_y1_sq;
                if (fmul_ready) next_state = DBL_X1_SQ;
            end
            DBL_X1_SQ: begin
                fmul_start = 1'b1; fmul_a = r_x1; fmul_b = r_x1;
                fadd_a = r_x1_sq; fadd_b = {r_x1_sq[WIDTH-2:0], 1'b0}; // mul3
                if (fmul_ready) next_state = DBL_M_SQ;
            end
            DBL_M_SQ: begin
                fmul_start = 1'b1; fmul_a = r_m; fmul_b = r_m;
                fsub_a = fmul_res; fsub_b = {r_s[WIDTH-2:0], 1'b0}; // M^2 - 2*S
                if (fmul_ready) next_state = DBL_Y1_SQ_SQ;
            end
            DBL_Y1_SQ_SQ: begin
                fmul_start = 1'b1; fmul_a = r_y1_sq; fmul_b = r_y1_sq;
                fsub_a = r_s; fsub_b = r_x3; // S - X3
                if (fmul_ready) next_state = DBL_Y3_PRE;
            end
            DBL_Y3_PRE: begin
                fmul_start = 1'b1; fmul_a = r_m; fmul_b = fsub_res; // M * (S-X3)
                fsub_a = fmul_res; fsub_b = {r_y1_sq_sq[WIDTH-4:0], 3'b0}; // 8*Y1_sq_sq
                if (fmul_ready) next_state = DBL_Z3;
            end
            DBL_Z3: begin
                fmul_start = 1'b1; fmul_a = {r_y1[WIDTH-2:0], 1'b0}; fmul_b = r_z1; // 2*Y1*Z1
                if (fmul_ready) next_state = FINISH;
            end

            // --- Jacobian Add ---
            ADD_Z2_SQ: begin
                fmul_start = 1'b1; fmul_a = r_z2; fmul_b = r_z2;
                if (fmul_ready) next_state = ADD_U1;
            end
            ADD_U1: begin
                fmul_start = 1'b1; fmul_a = r_x1; fmul_b = r_z2_sq;
                if (fmul_ready) next_state = ADD_Z1_SQ;
            end
            ADD_Z1_SQ: begin
                fmul_start = 1'b1; fmul_a = r_z1; fmul_b = r_z1;
                if (fmul_ready) next_state = ADD_U2;
            end
            ADD_U2: begin
                fmul_start = 1'b1; fmul_a = r_x2; fmul_b = r_z1_sq;
                fsub_a = fmul_res; fsub_b = r_u1; // H = U2 - U1
                if (fmul_ready) next_state = ADD_Z2_CU;
            end
            ADD_Z2_CU: begin
                fmul_start = 1'b1; fmul_a = r_z2_sq; fmul_b = r_z2;
                if (fmul_ready) next_state = ADD_S1;
            end
            ADD_S1: begin
                fmul_start = 1'b1; fmul_a = r_y1; fmul_b = r_s1; // Y1 * Z2_cu
                if (fmul_ready) next_state = ADD_Z1_CU;
            end
            ADD_Z1_CU: begin
                fmul_start = 1'b1; fmul_a = r_z1_sq; fmul_b = r_z1;
                if (fmul_ready) next_state = ADD_S2;
            end
            ADD_S2: begin
                fmul_start = 1'b1; fmul_a = r_y2; fmul_b = r_s2; // Y2 * Z1_cu
                fsub_a = fmul_res; fsub_b = r_s1; // Rr = S2 - S1
                if (fmul_ready) next_state = ADD_H_SQ;
            end
            ADD_H_SQ: begin
                fmul_start = 1'b1; fmul_a = r_h; fmul_b = r_h;
                if (fmul_ready) next_state = ADD_H_CU;
            end
            ADD_H_CU: begin
                fmul_start = 1'b1; fmul_a = r_h_sq; fmul_b = r_h;
                if (fmul_ready) next_state = ADD_U1H2;
            end
            ADD_U1H2: begin
                fmul_start = 1'b1; fmul_a = r_u1; fmul_b = r_h_sq;
                if (fmul_ready) next_state = ADD_X3_PRE;
            end
            ADD_X3_PRE: begin
                fmul_start = 1'b1; fmul_a = r_rr; fmul_b = r_rr; // Rr^2
                fsub_a = fmul_res; fsub_b = r_h_cu; // Rr^2 - H_cu
                // Final X3 = (Rr^2 - H_cu) - 2*U1H2
                if (fmul_ready) next_state = ADD_Y3_PRE;
            end
            ADD_Y3_PRE: begin
                fmul_start = 1'b1; fmul_a = r_rr; fmul_b = fsub_res; // Rr * (U1H2 - X3)
                fsub_a = r_u1h2; fsub_b = r_x3; // U1H2 - X3
                if (fmul_ready) next_state = ADD_Z3_PRE;
            end
            ADD_Z3_PRE: begin
                fmul_start = 1'b1; fmul_a = r_z1; fmul_b = r_z2;
                if (fmul_ready) next_state = ADD_Z3_FINAL;
            end
            ADD_Z3_FINAL: begin
                fmul_start = 1'b1; fmul_a = fmul_res; fmul_b = r_h;
                if (fmul_ready) next_state = FINISH;
            end

            FINISH: begin
                done = 1'b1;
                if (!start) next_state = IDLE;
            end
            default: next_state = IDLE;
        endcase
    end

    // Output Mapping
    always_comb begin
        if (state == FINISH) begin
            if (z1 == '0 || (op == 2'b00 && r_y1 == '0)) begin
                // Case: P is INF or Double(INF) or Double(Y=0)
                x_out = r_x2; y_out = r_y2; z_out = r_z2; // Return Q for Add, or INF
                if (op == 2'b00) z_out = '0; // Double(Y=0) -> INF
            end else if (op == 2'b10 && r_z2 == '0) begin
                // Case: Q is INF
                x_out = r_x1; y_out = r_y1; z_out = r_z1;
            end else begin
                x_out = r_x3; y_out = r_y3; z_out = r_z3;
            end
        end else begin
            x_out = '0; y_out = '0; z_out = '0;
        end
    end
    
    assign is_inf = (state == FINISH) && (z_out == '0);

endmodule
