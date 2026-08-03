`timescale 1ns/1ps

module msm_multiwindow_controller_v1 #(
    parameter int ADDR_W          = 16,
    parameter int DATA_W          = 256,
    parameter int DEPTH           = (1 << ADDR_W),
    parameter int SRAM_RD_LATENCY = 1,
    parameter int GEN_W           = 16,
    parameter int FIFO_DEPTH      = 16,
    parameter int SLOT_COUNT      = 16,
    parameter int MIX_CTX_COUNT   = 40,
    parameter int MUL_LATENCY     = 16,
    parameter int WINDOW_BITS     = 16,
    parameter int NUM_WINDOWS     = 4
)(
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  start,

    input  logic                  in_valid,
    output logic                  in_ready,
    input  logic [ADDR_W-1:0]     in_bucket_id,
    input  logic [DATA_W-1:0]     in_point_x,
    input  logic [DATA_W-1:0]     in_point_y,
    input  logic                  last_point,

    output logic [$clog2(NUM_WINDOWS)-1:0] window_index,
    output logic                  busy,
    output logic                  done,

    output logic [DATA_W-1:0]     result_x,
    output logic [DATA_W-1:0]     result_y,
    output logic [DATA_W-1:0]     result_z
);

    localparam logic [DATA_W-1:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    localparam int WIN_IDX_W =
        (NUM_WINDOWS <= 1) ? 1 : $clog2(NUM_WINDOWS);

    typedef enum logic [4:0] {
        S_IDLE,
        S_WAIT_INIT,
        S_START_WINDOW,
        S_WAIT_WINDOW,
        S_WAIT_WINDOW_IDLE,
        S_CAPTURE_FIRST,
        S_PREP_DOUBLE,
        S_DOUBLE_START,
        S_DOUBLE_WAIT,
        S_ADD_START,
        S_ADD_WAIT,
        S_DONE
    } state_t;

    state_t state;

    logic [WIN_IDX_W-1:0] current_window;

    logic window_start;
    logic window_busy;
    logic window_done;
    logic window_in_ready;

    logic [DATA_W-1:0] window_result_x;
    logic [DATA_W-1:0] window_result_y;
    logic [DATA_W-1:0] window_result_z;

    logic [DATA_W-1:0] pending_x;
    logic [DATA_W-1:0] pending_y;
    logic [DATA_W-1:0] pending_z;

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

    assign window_index = current_window;

    assign stream_enable = (state == S_WAIT_WINDOW);
    assign in_ready = stream_enable ? window_in_ready : 1'b0;

    assign window_start = (state == S_START_WINDOW);
    assign double_start = (state == S_DOUBLE_START);
    assign add_start    = (state == S_ADD_START);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            current_window <= '0;

            pending_x      <= '0;
            pending_y      <= ONE_M;
            pending_z      <= '0;

            acc_x          <= '0;
            acc_y          <= ONE_M;
            acc_z          <= '0;

            double_count   <= '0;

            result_x       <= '0;
            result_y       <= ONE_M;
            result_z       <= '0;
        end else begin
            unique case (state)

                S_IDLE: begin
                    if (start) begin
                        current_window <= NUM_WINDOWS-1;

                        pending_x      <= '0;
                        pending_y      <= ONE_M;
                        pending_z      <= '0;

                        acc_x          <= '0;
                        acc_y          <= ONE_M;
                        acc_z          <= '0;

                        double_count   <= '0;

                        result_x       <= '0;
                        result_y       <= ONE_M;
                        result_z       <= '0;

                        state          <= S_WAIT_INIT;
                    end
                end

                S_WAIT_INIT: begin
                    if (!window_busy)
                        state <= S_START_WINDOW;
                end

                S_START_WINDOW: begin
                    state <= S_WAIT_WINDOW;
                end

                S_WAIT_WINDOW: begin
                    if (window_done) begin
                        pending_x <= window_result_x;
                        pending_y <= window_result_y;
                        pending_z <= window_result_z;
                        state     <= S_WAIT_WINDOW_IDLE;
                    end
                end

                S_WAIT_WINDOW_IDLE: begin
                    if (!window_busy) begin
                        if (current_window == NUM_WINDOWS-1)
                            state <= S_CAPTURE_FIRST;
                        else begin
                            double_count <= '0;
                            state        <= S_PREP_DOUBLE;
                        end
                    end
                end

                S_CAPTURE_FIRST: begin
                    acc_x <= pending_x;
                    acc_y <= pending_y;
                    acc_z <= pending_z;

                    if (current_window == 0) begin
                        result_x <= pending_x;
                        result_y <= pending_y;
                        result_z <= pending_z;
                        state    <= S_DONE;
                    end else begin
                        current_window <= current_window - 1'b1;
                        state          <= S_START_WINDOW;
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
                        acc_x <= add_x;
                        acc_y <= add_y;
                        acc_z <= add_z;

                        if (current_window == 0) begin
                            result_x <= add_x;
                            result_y <= add_y;
                            result_z <= add_z;
                            state    <= S_DONE;
                        end else begin
                            current_window <= current_window - 1'b1;
                            state          <= S_START_WINDOW;
                        end
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
        .X2    (pending_x),
        .Y2    (pending_y),
        .Z2    (pending_z),
        .busy  (add_busy),
        .done  (add_done),
        .X3    (add_x),
        .Y3    (add_y),
        .Z3    (add_z)
    );

    initial begin
        if (NUM_WINDOWS < 1)
            $fatal(1, "NUM_WINDOWS must be >= 1");

        if (WINDOW_BITS != 16)
            $fatal(
                1,
                "msm_multiwindow_controller_v1 currently expects WINDOW_BITS=16, got %0d",
                WINDOW_BITS
            );
    end

endmodule
