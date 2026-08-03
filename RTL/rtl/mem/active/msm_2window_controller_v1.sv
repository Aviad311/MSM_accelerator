`timescale 1ns/1ps

module msm_2window_controller_v1 #(
    parameter int ADDR_W          = 16,
    parameter int DATA_W          = 256,
    parameter int DEPTH           = (1 << ADDR_W),
    parameter int SRAM_RD_LATENCY = 1,
    parameter int GEN_W           = 16,
    parameter int FIFO_DEPTH      = 16,
    parameter int SLOT_COUNT      = 16,
    parameter int MIX_CTX_COUNT   = 40,
    parameter int MUL_LATENCY     = 16,
    parameter int WINDOW_BITS     = 16
)(
    input  logic                  clk,
    input  logic                  rst_n,

    // Starts one complete two-window operation.
    input  logic                  start,

    // Point stream for the currently requested window.
    // window_index=1 means high window, window_index=0 means low window.
    input  logic                  in_valid,
    output logic                  in_ready,
    input  logic [ADDR_W-1:0]     in_bucket_id,
    input  logic [DATA_W-1:0]     in_point_x,
    input  logic [DATA_W-1:0]     in_point_y,
    input  logic                  last_point,

    output logic                  window_index,
    output logic                  busy,
    output logic                  done,

    output logic [DATA_W-1:0]     result_x,
    output logic [DATA_W-1:0]     result_y,
    output logic [DATA_W-1:0]     result_z
);

    localparam logic [DATA_W-1:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    typedef enum logic [4:0] {
        S_IDLE,

        S_START_HIGH,
        S_WAIT_HIGH,

        S_WAIT_WINDOW_IDLE_1,
        S_START_LOW,
        S_WAIT_LOW,

        S_PREP_DOUBLE,
        S_DOUBLE_START,
        S_DOUBLE_WAIT,

        S_ADD_START,
        S_ADD_WAIT,

        S_DONE
    } state_t;

    state_t state;

    logic window_start;
    logic window_busy;
    logic window_done;
    logic window_in_ready;

    logic [DATA_W-1:0] window_result_x;
    logic [DATA_W-1:0] window_result_y;
    logic [DATA_W-1:0] window_result_z;

    logic [DATA_W-1:0] high_x;
    logic [DATA_W-1:0] high_y;
    logic [DATA_W-1:0] high_z;

    logic [DATA_W-1:0] low_x;
    logic [DATA_W-1:0] low_y;
    logic [DATA_W-1:0] low_z;

    logic [DATA_W-1:0] acc_x;
    logic [DATA_W-1:0] acc_y;
    logic [DATA_W-1:0] acc_z;

    logic [$clog2(WINDOW_BITS+1)-1:0] double_count;

    logic double_start;
    logic double_busy;
    logic double_done;
    logic [DATA_W-1:0] double_x;
    logic [DATA_W-1:0] double_y;
    logic [DATA_W-1:0] double_z;

    logic add_start;
    logic add_busy;
    logic add_done;
    logic [DATA_W-1:0] add_x;
    logic [DATA_W-1:0] add_y;
    logic [DATA_W-1:0] add_z;

    logic stream_enable;

    assign busy = (state != S_IDLE);
    assign done = (state == S_DONE);

    assign window_index =
        (state == S_START_HIGH ||
         state == S_WAIT_HIGH) ? 1'b1 : 1'b0;

    assign stream_enable =
        (state == S_WAIT_HIGH) ||
        (state == S_WAIT_LOW);

    assign in_ready =
        stream_enable ? window_in_ready : 1'b0;

    assign window_start =
        (state == S_START_HIGH) ||
        (state == S_START_LOW);

    assign double_start = (state == S_DOUBLE_START);
    assign add_start    = (state == S_ADD_START);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;

            high_x       <= '0;
            high_y       <= ONE_M;
            high_z       <= '0;

            low_x        <= '0;
            low_y        <= ONE_M;
            low_z        <= '0;

            acc_x        <= '0;
            acc_y        <= ONE_M;
            acc_z        <= '0;

            double_count <= '0;

            result_x     <= '0;
            result_y     <= ONE_M;
            result_z     <= '0;
        end else begin
            unique case (state)

                S_IDLE: begin
                    if (start) begin
                        high_x       <= '0;
                        high_y       <= ONE_M;
                        high_z       <= '0;

                        low_x        <= '0;
                        low_y        <= ONE_M;
                        low_z        <= '0;

                        acc_x        <= '0;
                        acc_y        <= ONE_M;
                        acc_z        <= '0;

                        double_count <= '0;

                        result_x     <= '0;
                        result_y     <= ONE_M;
                        result_z     <= '0;

                        state        <= S_START_HIGH;
                    end
                end

                S_START_HIGH: begin
                    state <= S_WAIT_HIGH;
                end

                S_WAIT_HIGH: begin
                    if (window_done) begin
                        high_x <= window_result_x;
                        high_y <= window_result_y;
                        high_z <= window_result_z;
                        state  <= S_WAIT_WINDOW_IDLE_1;
                    end
                end

                // The one-window top asserts done in S_DONE and returns
                // to S_IDLE on the following cycle. Wait until busy drops
                // before launching the next generation.
                S_WAIT_WINDOW_IDLE_1: begin
                    if (!window_busy)
                        state <= S_START_LOW;
                end

                S_START_LOW: begin
                    state <= S_WAIT_LOW;
                end

                S_WAIT_LOW: begin
                    if (window_done) begin
                        low_x <= window_result_x;
                        low_y <= window_result_y;
                        low_z <= window_result_z;

                        acc_x <= high_x;
                        acc_y <= high_y;
                        acc_z <= high_z;

                        double_count <= '0;
                        state        <= S_PREP_DOUBLE;
                    end
                end

                S_PREP_DOUBLE: begin
                    if (WINDOW_BITS == 0)
                        state <= S_ADD_START;
                    else
                        state <= S_DOUBLE_START;
                end

                S_DOUBLE_START: begin
                    state <= S_DOUBLE_WAIT;
                end

                S_DOUBLE_WAIT: begin
                    if (double_done) begin
                        acc_x <= double_x;
                        acc_y <= double_y;
                        acc_z <= double_z;

                        if (double_count == WINDOW_BITS-1) begin
                            state <= S_ADD_START;
                        end else begin
                            double_count <= double_count + 1'b1;
                            state        <= S_DOUBLE_START;
                        end
                    end
                end

                S_ADD_START: begin
                    state <= S_ADD_WAIT;
                end

                S_ADD_WAIT: begin
                    if (add_done) begin
                        result_x <= add_x;
                        result_y <= add_y;
                        result_z <= add_z;
                        state    <= S_DONE;
                    end
                end

                S_DONE: begin
                    state <= S_IDLE;
                end

                default: begin
                    state <= S_IDLE;
                end

            endcase
        end
    end

    pippenger_window_mem_stream_top_8lane_pipeline_reduce4mul_overlap_sram_macro_v2 #(
        .ADDR_W          (ADDR_W),
        .DATA_W          (DATA_W),
        .DEPTH           (DEPTH),
        .SRAM_RD_LATENCY (SRAM_RD_LATENCY),
        .GEN_W           (GEN_W),
        .FIFO_DEPTH      (FIFO_DEPTH),
        .SLOT_COUNT      (SLOT_COUNT),
        .MIX_CTX_COUNT   (MIX_CTX_COUNT),
        .MUL_LATENCY     (MUL_LATENCY)
    ) u_window (
        .clk          (clk),
        .rst_n        (rst_n),

        .start        (window_start),

        .in_valid     (stream_enable ? in_valid : 1'b0),
        .in_ready     (window_in_ready),
        .in_bucket_id (in_bucket_id),
        .in_point_x   (in_point_x),
        .in_point_y   (in_point_y),
        .last_point   (stream_enable ? last_point : 1'b0),

        .busy         (window_busy),
        .done         (window_done),

        .result_x     (window_result_x),
        .result_y     (window_result_y),
        .result_z     (window_result_z)
    );

    jacobian_double_seq #(
        .WIDTH(DATA_W)
    ) u_double (
        .clk   (clk),
        .rst_n (rst_n),

        .start (double_start),

        .X1    (acc_x),
        .Y1    (acc_y),
        .Z1    (acc_z),

        .busy  (double_busy),
        .done  (double_done),

        .X3    (double_x),
        .Y3    (double_y),
        .Z3    (double_z)
    );

    jacobian_add_4mul_seq #(
        .WIDTH(DATA_W)
    ) u_add (
        .clk   (clk),
        .rst_n (rst_n),

        .start (add_start),

        .X1    (acc_x),
        .Y1    (acc_y),
        .Z1    (acc_z),

        .X2    (low_x),
        .Y2    (low_y),
        .Z2    (low_z),

        .busy  (add_busy),
        .done  (add_done),

        .X3    (add_x),
        .Y3    (add_y),
        .Z3    (add_z)
    );

    initial begin
        if (WINDOW_BITS != 16) begin
            $fatal(
                1,
                "msm_2window_controller_v1 currently expects WINDOW_BITS=16, got %0d",
                WINDOW_BITS
            );
        end
    end

endmodule
