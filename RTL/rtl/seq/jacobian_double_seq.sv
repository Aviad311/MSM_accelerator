// =================================================================
// File: RTL/seq/jacobian_double_seq.sv
// =================================================================
// Sequential Jacobian point doubling for secp256k1, Montgomery domain.
//
// Input:
//   P = (X1, Y1, Z1)
//
// Output:
//   2P = (X3, Y3, Z3)
//
// Formula for a = 0 short Weierstrass curve:
//
//   XX    = X1^2
//   YY    = Y1^2
//   YYYY  = YY^2
//   S     = 4*X1*YY
//   M     = 3*XX
//   X3    = M^2 - 2*S
//   Y3    = M*(S - X3) - 8*YYYY
//   Z3    = 2*Y1*Z1
//
// Multiplications use field_mul_seq.
// Add/sub/double are combinational modular field operations.
// =================================================================

`timescale 1ns/1ps

module jacobian_double_seq #(
    parameter int WIDTH = 256
) (
    input  logic             clk,
    input  logic             rst_n,

    input  logic             start,
    input  logic [WIDTH-1:0] X1,
    input  logic [WIDTH-1:0] Y1,
    input  logic [WIDTH-1:0] Z1,

    output logic             busy,
    output logic             done,
    output logic [WIDTH-1:0] X3,
    output logic [WIDTH-1:0] Y3,
    output logic [WIDTH-1:0] Z3
);

    localparam logic [255:0] P =
        256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    // -------------------------------------------------------------
    // Combinational modular field operations
    // -------------------------------------------------------------
    function automatic logic [255:0] field_add_mod(
        input logic [255:0] a,
        input logic [255:0] b
    );
        logic [256:0] sum;
        logic [256:0] reduced;
        begin
            sum = {1'b0, a} + {1'b0, b};

            if (sum >= {1'b0, P}) begin
                reduced = sum - {1'b0, P};
                field_add_mod = reduced[255:0];
            end else begin
                field_add_mod = sum[255:0];
            end
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

    function automatic logic [255:0] field_triple_mod(
        input logic [255:0] a
    );
        logic [255:0] two_a;
        begin
            two_a = field_add_mod(a, a);
            field_triple_mod = field_add_mod(two_a, a);
        end
    endfunction

    function automatic logic [255:0] field_times4_mod(
        input logic [255:0] a
    );
        logic [255:0] two_a;
        begin
            two_a = field_double_mod(a);
            field_times4_mod = field_double_mod(two_a);
        end
    endfunction

    function automatic logic [255:0] field_times8_mod(
        input logic [255:0] a
    );
        logic [255:0] four_a;
        begin
            four_a = field_times4_mod(a);
            field_times8_mod = field_double_mod(four_a);
        end
    endfunction

    // -------------------------------------------------------------
    // FSM states
    // -------------------------------------------------------------
    typedef enum logic [4:0] {
        S_IDLE,

        S_MUL_XX_START,
        S_MUL_XX_WAIT,

        S_MUL_YY_START,
        S_MUL_YY_WAIT,

        S_MUL_YYYY_START,
        S_MUL_YYYY_WAIT,

        S_MUL_S_START,
        S_MUL_S_WAIT,

        S_MUL_T_START,
        S_MUL_T_WAIT,

        S_MUL_Y_START,
        S_MUL_Y_WAIT,

        S_MUL_Z_START,
        S_MUL_Z_WAIT,

        S_DONE
    } state_t;

    state_t state, next_state;

    // Input/output registers
    logic [255:0] X1_reg, Y1_reg, Z1_reg;
    logic [255:0] X3_reg, Y3_reg, Z3_reg;

    // Temporaries
    logic [255:0] XX;
    logic [255:0] YY;
    logic [255:0] YYYY;
    logic [255:0] S;
    logic [255:0] M;

    // Multiplier interface
    logic         mul_start;
    logic [255:0] mul_a;
    logic [255:0] mul_b;
    logic         mul_busy;
    logic         mul_done;
    logic [255:0] mul_result;

    assign X3 = X3_reg;
    assign Y3 = Y3_reg;
    assign Z3 = Z3_reg;

    // -------------------------------------------------------------
    // State and datapath registers
    // -------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state  <= S_IDLE;

            X1_reg <= '0;
            Y1_reg <= '0;
            Z1_reg <= '0;

            X3_reg <= '0;
            Y3_reg <= ONE_M;
            Z3_reg <= '0;

            XX     <= '0;
            YY     <= '0;
            YYYY   <= '0;
            S      <= '0;
            M      <= '0;
        end else begin
            state <= next_state;

            if (state == S_IDLE && start) begin
                X1_reg <= X1;
                Y1_reg <= Y1;
                Z1_reg <= Z1;

                // Infinity / zero-Y case:
                // double(INF) = INF
                if (Z1 == '0 || Y1 == '0) begin
                    X3_reg <= '0;
                    Y3_reg <= ONE_M;
                    Z3_reg <= '0;
                end
            end

            // XX = X1^2
            // M  = 3*XX
            if (state == S_MUL_XX_WAIT && mul_done) begin
                XX <= mul_result;
                M  <= field_triple_mod(mul_result);
            end

            // YY = Y1^2
            if (state == S_MUL_YY_WAIT && mul_done) begin
                YY <= mul_result;
            end

            // YYYY = YY^2
            if (state == S_MUL_YYYY_WAIT && mul_done) begin
                YYYY <= mul_result;
            end

            // S = 4*X1*YY
            if (state == S_MUL_S_WAIT && mul_done) begin
                S <= field_times4_mod(mul_result);
            end

            // X3 = M^2 - 2*S
            if (state == S_MUL_T_WAIT && mul_done) begin
                X3_reg <= field_sub_mod(
                              field_sub_mod(mul_result, S),
                              S
                          );
            end

            // Y3 = M*(S - X3) - 8*YYYY
            if (state == S_MUL_Y_WAIT && mul_done) begin
                Y3_reg <= field_sub_mod(
                              mul_result,
                              field_times8_mod(YYYY)
                          );
            end

            // Z3 = 2*Y1*Z1
            if (state == S_MUL_Z_WAIT && mul_done) begin
                Z3_reg <= field_double_mod(mul_result);
            end
        end
    end

    // -------------------------------------------------------------
    // FSM combinational logic
    // -------------------------------------------------------------
    always_comb begin
        next_state = state;

        busy = 1'b0;
        done = 1'b0;

        mul_start = 1'b0;
        mul_a     = '0;
        mul_b     = '0;

        case (state)
            S_IDLE: begin
                busy = 1'b0;

                if (start) begin
                    if (Z1 == '0 || Y1 == '0) begin
                        next_state = S_DONE;
                    end else begin
                        next_state = S_MUL_XX_START;
                    end
                end
            end

            S_MUL_XX_START: begin
                busy       = 1'b1;
                mul_start  = 1'b1;
                mul_a      = X1_reg;
                mul_b      = X1_reg;
                next_state = S_MUL_XX_WAIT;
            end

            S_MUL_XX_WAIT: begin
                busy = 1'b1;
                if (mul_done) begin
                    next_state = S_MUL_YY_START;
                end
            end

            S_MUL_YY_START: begin
                busy       = 1'b1;
                mul_start  = 1'b1;
                mul_a      = Y1_reg;
                mul_b      = Y1_reg;
                next_state = S_MUL_YY_WAIT;
            end

            S_MUL_YY_WAIT: begin
                busy = 1'b1;
                if (mul_done) begin
                    next_state = S_MUL_YYYY_START;
                end
            end

            S_MUL_YYYY_START: begin
                busy       = 1'b1;
                mul_start  = 1'b1;
                mul_a      = YY;
                mul_b      = YY;
                next_state = S_MUL_YYYY_WAIT;
            end

            S_MUL_YYYY_WAIT: begin
                busy = 1'b1;
                if (mul_done) begin
                    next_state = S_MUL_S_START;
                end
            end

            S_MUL_S_START: begin
                busy       = 1'b1;
                mul_start  = 1'b1;
                mul_a      = X1_reg;
                mul_b      = YY;
                next_state = S_MUL_S_WAIT;
            end

            S_MUL_S_WAIT: begin
                busy = 1'b1;
                if (mul_done) begin
                    next_state = S_MUL_T_START;
                end
            end

            S_MUL_T_START: begin
                busy       = 1'b1;
                mul_start  = 1'b1;
                mul_a      = M;
                mul_b      = M;
                next_state = S_MUL_T_WAIT;
            end

            S_MUL_T_WAIT: begin
                busy = 1'b1;
                if (mul_done) begin
                    next_state = S_MUL_Y_START;
                end
            end

            S_MUL_Y_START: begin
                busy       = 1'b1;
                mul_start  = 1'b1;
                mul_a      = M;
                mul_b      = field_sub_mod(S, X3_reg);
                next_state = S_MUL_Y_WAIT;
            end

            S_MUL_Y_WAIT: begin
                busy = 1'b1;
                if (mul_done) begin
                    next_state = S_MUL_Z_START;
                end
            end

            S_MUL_Z_START: begin
                busy       = 1'b1;
                mul_start  = 1'b1;
                mul_a      = Y1_reg;
                mul_b      = Z1_reg;
                next_state = S_MUL_Z_WAIT;
            end

            S_MUL_Z_WAIT: begin
                busy = 1'b1;
                if (mul_done) begin
                    next_state = S_DONE;
                end
            end

            S_DONE: begin
                busy       = 1'b0;
                done       = 1'b1;
                next_state = S_IDLE;
            end

            default: begin
                next_state = S_IDLE;
            end
        endcase
    end

    // -------------------------------------------------------------
    // One shared field multiplier
    // -------------------------------------------------------------
    field_mul_seq #(
        .WIDTH(WIDTH)
    ) u_field_mul (
        .clk    (clk),
        .rst_n  (rst_n),
        .start  (mul_start),
        .a      (mul_a),
        .b      (mul_b),
        .busy   (mul_busy),
        .done   (mul_done),
        .result (mul_result)
    );

endmodule