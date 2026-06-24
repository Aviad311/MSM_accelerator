`timescale 1ns/1ps

module reduce_buckets_mem_4mul_overlap #(
    parameter int ADDR_W = 4,
    parameter int DATA_W = 256,
    parameter int GEN_W  = 16
)(
    input  logic                clk,
    input  logic                rst_n,

    input  logic                start,
    input  logic [GEN_W-1:0]    current_gen,

    output logic                busy,
    output logic                done,

    output logic [DATA_W-1:0]   result_x,
    output logic [DATA_W-1:0]   result_y,
    output logic [DATA_W-1:0]   result_z,

    output logic                mem_valid,
    output logic                mem_write_en,
    output logic [ADDR_W-1:0]   mem_addr,

    output logic [DATA_W-1:0]   mem_wdata_x,
    output logic [DATA_W-1:0]   mem_wdata_y,
    output logic [DATA_W-1:0]   mem_wdata_z,

    output logic                mem_tag_write_en,
    output logic [GEN_W-1:0]    mem_tag_wdata,

    input  logic                mem_ready,
    input  logic                mem_rvalid,

    input  logic [DATA_W-1:0]   mem_rdata_x,
    input  logic [DATA_W-1:0]   mem_rdata_y,
    input  logic [DATA_W-1:0]   mem_rdata_z,
    input  logic [GEN_W-1:0]    mem_tag_rdata
);

    localparam logic [255:0] ZERO =
        256'h0000000000000000000000000000000000000000000000000000000000000000;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    // ----------------------------------------------------------------
    // Reduction recurrence, descending from MAX bucket to bucket 1:
    //
    //   running_i = running_(i+1) + bucket_i
    //   accum_i   = accum_(i+1)   + running_i
    //
    // The running chain is truly dependent and cannot be fully
    // pipelined. However, after running_i is known, these two operations
    // are independent and can overlap:
    //
    //   accum_i       = accum_(i+1) + running_i
    //   running_(i-1) = running_i   + bucket_(i-1)
    //
    // This module uses two jacobian_add_4mul_seq instances concurrently.
    // ----------------------------------------------------------------

    typedef enum logic [4:0] {
        S_IDLE,
        S_INIT,

        // Bootstrap the highest bucket.
        S_READ_FIRST,
        S_WAIT_FIRST,
        S_FIRST_RUNNING_START,
        S_FIRST_RUNNING_WAIT,

        // Steady-state overlapped loop.
        S_ACCUM_START,
        S_READ_NEXT,
        S_WAIT_NEXT,
        S_NEXT_RUNNING_START,
        S_WAIT_BOTH,
        S_COMMIT_BOTH,

        // Final bucket only needs the last accum update.
        S_FINAL_ACCUM_START,
        S_FINAL_ACCUM_WAIT,

        S_DONE
    } state_t;

    state_t state;

    // bucket_idx always names the bucket whose running value is already
    // available in running_x/y/z and whose accum update is next.
    logic [ADDR_W-1:0] bucket_idx;

    logic [DATA_W-1:0] running_x;
    logic [DATA_W-1:0] running_y;
    logic [DATA_W-1:0] running_z;

    logic [DATA_W-1:0] accum_x;
    logic [DATA_W-1:0] accum_y;
    logic [DATA_W-1:0] accum_z;

    logic [DATA_W-1:0] bucket_x_r;
    logic [DATA_W-1:0] bucket_y_r;
    logic [DATA_W-1:0] bucket_z_r;

    logic add_running_start;
    logic add_running_busy;
    logic add_running_done;

    logic [DATA_W-1:0] add_running_x3;
    logic [DATA_W-1:0] add_running_y3;
    logic [DATA_W-1:0] add_running_z3;

    logic add_result_start;
    logic add_result_busy;
    logic add_result_done;

    logic [DATA_W-1:0] add_result_x3;
    logic [DATA_W-1:0] add_result_y3;
    logic [DATA_W-1:0] add_result_z3;

    // Completion holding registers are required because the two adders
    // start on different cycles and may finish on different cycles.
    logic accum_done_seen;
    logic running_done_seen;

    logic [DATA_W-1:0] accum_next_x;
    logic [DATA_W-1:0] accum_next_y;
    logic [DATA_W-1:0] accum_next_z;

    logic [DATA_W-1:0] running_next_x;
    logic [DATA_W-1:0] running_next_y;
    logic [DATA_W-1:0] running_next_z;

    jacobian_add_4mul_seq #(
        .WIDTH(DATA_W)
    ) u_add_bucket_to_running (
        .clk   (clk),
        .rst_n (rst_n),
        .start (add_running_start),

        .X1    (running_x),
        .Y1    (running_y),
        .Z1    (running_z),

        .X2    (bucket_x_r),
        .Y2    (bucket_y_r),
        .Z2    (bucket_z_r),

        .busy  (add_running_busy),
        .done  (add_running_done),

        .X3    (add_running_x3),
        .Y3    (add_running_y3),
        .Z3    (add_running_z3)
    );

    jacobian_add_4mul_seq #(
        .WIDTH(DATA_W)
    ) u_add_running_to_accum (
        .clk   (clk),
        .rst_n (rst_n),
        .start (add_result_start),

        .X1    (accum_x),
        .Y1    (accum_y),
        .Z1    (accum_z),

        .X2    (running_x),
        .Y2    (running_y),
        .Z2    (running_z),

        .busy  (add_result_busy),
        .done  (add_result_done),

        .X3    (add_result_x3),
        .Y3    (add_result_y3),
        .Z3    (add_result_z3)
    );

    assign busy = (state != S_IDLE);
    assign done = (state == S_DONE);

    always_comb begin
        mem_valid        = 1'b0;
        mem_write_en     = 1'b0;
        mem_addr         = '0;

        mem_wdata_x      = '0;
        mem_wdata_y      = '0;
        mem_wdata_z      = '0;

        mem_tag_write_en = 1'b0;
        mem_tag_wdata    = '0;

        add_running_start = 1'b0;
        add_result_start  = 1'b0;

        unique case (state)

            S_READ_FIRST: begin
                mem_valid    = 1'b1;
                mem_write_en = 1'b0;
                mem_addr     = bucket_idx;
            end

            S_FIRST_RUNNING_START: begin
                add_running_start = 1'b1;
            end

            S_ACCUM_START: begin
                add_result_start = 1'b1;
            end

            S_READ_NEXT: begin
                mem_valid    = 1'b1;
                mem_write_en = 1'b0;
                mem_addr     = bucket_idx - 1'b1;
            end

            S_NEXT_RUNNING_START: begin
                add_running_start = 1'b1;
            end

            S_FINAL_ACCUM_START: begin
                add_result_start = 1'b1;
            end

            default: begin
            end

        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            bucket_idx <= '0;

            running_x <= ZERO;
            running_y <= ONE_M;
            running_z <= ZERO;

            accum_x <= ZERO;
            accum_y <= ONE_M;
            accum_z <= ZERO;

            bucket_x_r <= ZERO;
            bucket_y_r <= ONE_M;
            bucket_z_r <= ZERO;

            accum_done_seen   <= 1'b0;
            running_done_seen <= 1'b0;

            accum_next_x <= ZERO;
            accum_next_y <= ONE_M;
            accum_next_z <= ZERO;

            running_next_x <= ZERO;
            running_next_y <= ONE_M;
            running_next_z <= ZERO;

            result_x <= ZERO;
            result_y <= ONE_M;
            result_z <= ZERO;
        end else begin

            // Capture completions independently while an overlapped pair
            // is in flight.
            if ((state == S_READ_NEXT       ||
                 state == S_WAIT_NEXT       ||
                 state == S_NEXT_RUNNING_START ||
                 state == S_WAIT_BOTH) &&
                add_result_done) begin

                accum_done_seen <= 1'b1;
                accum_next_x    <= add_result_x3;
                accum_next_y    <= add_result_y3;
                accum_next_z    <= add_result_z3;
            end

            if ((state == S_WAIT_BOTH) && add_running_done) begin
                running_done_seen <= 1'b1;
                running_next_x    <= add_running_x3;
                running_next_y    <= add_running_y3;
                running_next_z    <= add_running_z3;
            end

            unique case (state)

                S_IDLE: begin
                    if (start)
                        state <= S_INIT;
                end

                S_INIT: begin
                    bucket_idx <= {ADDR_W{1'b1}};

                    running_x <= ZERO;
                    running_y <= ONE_M;
                    running_z <= ZERO;

                    accum_x <= ZERO;
                    accum_y <= ONE_M;
                    accum_z <= ZERO;

                    result_x <= ZERO;
                    result_y <= ONE_M;
                    result_z <= ZERO;

                    accum_done_seen   <= 1'b0;
                    running_done_seen <= 1'b0;

                    state <= S_READ_FIRST;
                end

                // ----------------------------------------------------
                // Bootstrap:
                // Obtain running_MAX before the overlap loop begins.
                // ----------------------------------------------------
                S_READ_FIRST: begin
                    if (mem_ready)
                        state <= S_WAIT_FIRST;
                end

                S_WAIT_FIRST: begin
                    if (mem_rvalid) begin
                        if (mem_tag_rdata === current_gen) begin
                            bucket_x_r <= mem_rdata_x;
                            bucket_y_r <= mem_rdata_y;
                            bucket_z_r <= mem_rdata_z;
                            state      <= S_FIRST_RUNNING_START;
                        end else begin
                            // Empty highest bucket:
                            // running remains infinity.
                            state <= S_ACCUM_START;
                        end
                    end
                end

                S_FIRST_RUNNING_START: begin
                    state <= S_FIRST_RUNNING_WAIT;
                end

                S_FIRST_RUNNING_WAIT: begin
                    if (add_running_done) begin
                        running_x <= add_running_x3;
                        running_y <= add_running_y3;
                        running_z <= add_running_z3;
                        state     <= S_ACCUM_START;
                    end
                end

                // ----------------------------------------------------
                // Start accum_i. If this is bucket 1, no next running
                // value is needed; otherwise prefetch bucket i-1.
                // ----------------------------------------------------
                S_ACCUM_START: begin
                    accum_done_seen   <= 1'b0;
                    running_done_seen <= 1'b0;

                    if (bucket_idx ==
                        {{(ADDR_W-1){1'b0}}, 1'b1}) begin

                        state <= S_FINAL_ACCUM_WAIT;
                    end else begin
                        state <= S_READ_NEXT;
                    end
                end

                S_READ_NEXT: begin
                    if (mem_ready)
                        state <= S_WAIT_NEXT;
                end

                S_WAIT_NEXT: begin
                    if (mem_rvalid) begin
                        if (mem_tag_rdata === current_gen) begin
                            bucket_x_r <= mem_rdata_x;
                            bucket_y_r <= mem_rdata_y;
                            bucket_z_r <= mem_rdata_z;
                            state      <= S_NEXT_RUNNING_START;
                        end else begin
                            // Empty bucket i-1:
                            // running_(i-1) = running_i.
                            running_next_x    <= running_x;
                            running_next_y    <= running_y;
                            running_next_z    <= running_z;
                            running_done_seen <= 1'b1;
                            state             <= S_WAIT_BOTH;
                        end
                    end
                end

                S_NEXT_RUNNING_START: begin
                    state <= S_WAIT_BOTH;
                end

                S_WAIT_BOTH: begin
                    if (accum_done_seen && running_done_seen)
                        state <= S_COMMIT_BOTH;
                end

                S_COMMIT_BOTH: begin
                    accum_x <= accum_next_x;
                    accum_y <= accum_next_y;
                    accum_z <= accum_next_z;

                    running_x <= running_next_x;
                    running_y <= running_next_y;
                    running_z <= running_next_z;

                    bucket_idx <= bucket_idx - 1'b1;

                    state <= S_ACCUM_START;
                end

                // This state is entered after the start pulse for the
                // final accum_1 update has already been issued.
                S_FINAL_ACCUM_WAIT: begin
                    if (add_result_done) begin
                        accum_x <= add_result_x3;
                        accum_y <= add_result_y3;
                        accum_z <= add_result_z3;

                        result_x <= add_result_x3;
                        result_y <= add_result_y3;
                        result_z <= add_result_z3;

                        state <= S_DONE;
                    end
                end

                S_FINAL_ACCUM_START: begin
                    // Kept in the enum for waveform readability.
                    // The current schedule starts the final accumulation
                    // directly from S_ACCUM_START.
                    state <= S_FINAL_ACCUM_WAIT;
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

endmodule