`timescale 1ns/1ps

module pippenger_window_mem_seq_4 #(
    parameter int ADDR_W = 4,
    parameter int DATA_W = 256,
    parameter int DEPTH  = (1 << ADDR_W)
)(
    input  logic                clk,
    input  logic                rst_n,

    input  logic                start,

    input  logic [DATA_W-1:0]   p0_x,
    input  logic [DATA_W-1:0]   p0_y,
    input  logic [ADDR_W-1:0]   b0,

    input  logic [DATA_W-1:0]   p1_x,
    input  logic [DATA_W-1:0]   p1_y,
    input  logic [ADDR_W-1:0]   b1,

    input  logic [DATA_W-1:0]   p2_x,
    input  logic [DATA_W-1:0]   p2_y,
    input  logic [ADDR_W-1:0]   b2,

    input  logic [DATA_W-1:0]   p3_x,
    input  logic [DATA_W-1:0]   p3_y,
    input  logic [ADDR_W-1:0]   b3,

    output logic                busy,
    output logic                done,

    output logic [DATA_W-1:0]   result_x,
    output logic [DATA_W-1:0]   result_y,
    output logic [DATA_W-1:0]   result_z
);

    typedef enum logic [2:0] {
        S_IDLE,
        S_BUILD_START,
        S_BUILD_WAIT,
        S_REDUCE_START,
        S_REDUCE_WAIT,
        S_DONE
    } state_t;

    state_t state;

    logic build_start;
    logic build_busy;
    logic build_done;

    logic [2:0] build_processed_count;
    logic [2:0] build_skipped_count;

    logic [DATA_W-1:0] build_last_x;
    logic [DATA_W-1:0] build_last_y;
    logic [DATA_W-1:0] build_last_z;

    logic build_mem_valid;
    logic build_mem_write_en;
    logic [ADDR_W-1:0] build_mem_addr;

    logic [DATA_W-1:0] build_mem_wdata_x;
    logic [DATA_W-1:0] build_mem_wdata_y;
    logic [DATA_W-1:0] build_mem_wdata_z;

    logic reduce_start;
    logic reduce_busy;
    logic reduce_done;

    logic [DATA_W-1:0] reduce_result_x;
    logic [DATA_W-1:0] reduce_result_y;
    logic [DATA_W-1:0] reduce_result_z;

    logic reduce_mem_valid;
    logic reduce_mem_write_en;
    logic [ADDR_W-1:0] reduce_mem_addr;

    logic [DATA_W-1:0] reduce_mem_wdata_x;
    logic [DATA_W-1:0] reduce_mem_wdata_y;
    logic [DATA_W-1:0] reduce_mem_wdata_z;

    logic mem_valid;
    logic mem_write_en;
    logic [ADDR_W-1:0] mem_addr;

    logic [DATA_W-1:0] mem_wdata_x;
    logic [DATA_W-1:0] mem_wdata_y;
    logic [DATA_W-1:0] mem_wdata_z;

    logic mem_ready;
    logic mem_rvalid;

    logic [DATA_W-1:0] mem_rdata_x;
    logic [DATA_W-1:0] mem_rdata_y;
    logic [DATA_W-1:0] mem_rdata_z;

    assign busy = (state != S_IDLE);
    assign done = (state == S_DONE);

    // Shared bucket memory.
    bucket_mem_3coord #(
        .ADDR_W(ADDR_W),
        .DATA_W(DATA_W),
        .DEPTH (DEPTH)
    ) u_bucket_mem (
        .clk      (clk),
        .rst_n    (rst_n),

        .valid    (mem_valid),
        .write_en (mem_write_en),
        .addr     (mem_addr),

        .wdata_x  (mem_wdata_x),
        .wdata_y  (mem_wdata_y),
        .wdata_z  (mem_wdata_z),

        .ready    (mem_ready),
        .rvalid   (mem_rvalid),

        .rdata_x  (mem_rdata_x),
        .rdata_y  (mem_rdata_y),
        .rdata_z  (mem_rdata_z)
    );

    bucket_build_mem_seq_4_ext #(
        .ADDR_W(ADDR_W),
        .DATA_W(DATA_W),
        .DEPTH (DEPTH)
    ) u_build (
        .clk   (clk),
        .rst_n (rst_n),

        .start (build_start),

        .p0_x  (p0_x),
        .p0_y  (p0_y),
        .b0    (b0),

        .p1_x  (p1_x),
        .p1_y  (p1_y),
        .b1    (b1),

        .p2_x  (p2_x),
        .p2_y  (p2_y),
        .b2    (b2),

        .p3_x  (p3_x),
        .p3_y  (p3_y),
        .b3    (b3),

        .busy  (build_busy),
        .done  (build_done),

        .processed_count(build_processed_count),
        .skipped_count  (build_skipped_count),

        .last_x(build_last_x),
        .last_y(build_last_y),
        .last_z(build_last_z),

        .mem_valid    (build_mem_valid),
        .mem_write_en (build_mem_write_en),
        .mem_addr     (build_mem_addr),

        .mem_wdata_x  (build_mem_wdata_x),
        .mem_wdata_y  (build_mem_wdata_y),
        .mem_wdata_z  (build_mem_wdata_z),

        .mem_ready    (mem_ready),
        .mem_rvalid   (mem_rvalid),

        .mem_rdata_x  (mem_rdata_x),
        .mem_rdata_y  (mem_rdata_y),
        .mem_rdata_z  (mem_rdata_z)
    );

    reduce_buckets_mem_seq_4 #(
        .ADDR_W(ADDR_W),
        .DATA_W(DATA_W)
    ) u_reduce (
        .clk   (clk),
        .rst_n (rst_n),

        .start (reduce_start),

        .busy  (reduce_busy),
        .done  (reduce_done),

        .result_x(reduce_result_x),
        .result_y(reduce_result_y),
        .result_z(reduce_result_z),

        .mem_valid    (reduce_mem_valid),
        .mem_write_en (reduce_mem_write_en),
        .mem_addr     (reduce_mem_addr),

        .mem_wdata_x  (reduce_mem_wdata_x),
        .mem_wdata_y  (reduce_mem_wdata_y),
        .mem_wdata_z  (reduce_mem_wdata_z),

        .mem_ready    (mem_ready),
        .mem_rvalid   (mem_rvalid),

        .mem_rdata_x  (mem_rdata_x),
        .mem_rdata_y  (mem_rdata_y),
        .mem_rdata_z  (mem_rdata_z)
    );

    always_comb begin
        build_start  = 1'b0;
        reduce_start = 1'b0;

        mem_valid    = 1'b0;
        mem_write_en = 1'b0;
        mem_addr     = '0;

        mem_wdata_x  = '0;
        mem_wdata_y  = '0;
        mem_wdata_z  = '0;

        unique case (state)

            S_BUILD_START: begin
                build_start = 1'b1;

                mem_valid    = build_mem_valid;
                mem_write_en = build_mem_write_en;
                mem_addr     = build_mem_addr;

                mem_wdata_x  = build_mem_wdata_x;
                mem_wdata_y  = build_mem_wdata_y;
                mem_wdata_z  = build_mem_wdata_z;
            end

            S_BUILD_WAIT: begin
                mem_valid    = build_mem_valid;
                mem_write_en = build_mem_write_en;
                mem_addr     = build_mem_addr;

                mem_wdata_x  = build_mem_wdata_x;
                mem_wdata_y  = build_mem_wdata_y;
                mem_wdata_z  = build_mem_wdata_z;
            end

            S_REDUCE_START: begin
                reduce_start = 1'b1;

                mem_valid    = reduce_mem_valid;
                mem_write_en = reduce_mem_write_en;
                mem_addr     = reduce_mem_addr;

                mem_wdata_x  = reduce_mem_wdata_x;
                mem_wdata_y  = reduce_mem_wdata_y;
                mem_wdata_z  = reduce_mem_wdata_z;
            end

            S_REDUCE_WAIT: begin
                mem_valid    = reduce_mem_valid;
                mem_write_en = reduce_mem_write_en;
                mem_addr     = reduce_mem_addr;

                mem_wdata_x  = reduce_mem_wdata_x;
                mem_wdata_y  = reduce_mem_wdata_y;
                mem_wdata_z  = reduce_mem_wdata_z;
            end

            default: begin
            end

        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;

            result_x <= '0;
            result_y <= '0;
            result_z <= '0;
        end else begin
            unique case (state)

                S_IDLE: begin
                    if (start) begin
                        result_x <= '0;
                        result_y <= '0;
                        result_z <= '0;
                        state    <= S_BUILD_START;
                    end
                end

                S_BUILD_START: begin
                    state <= S_BUILD_WAIT;
                end

                S_BUILD_WAIT: begin
                    if (build_done) begin
                        state <= S_REDUCE_START;
                    end
                end

                S_REDUCE_START: begin
                    state <= S_REDUCE_WAIT;
                end

                S_REDUCE_WAIT: begin
                    if (reduce_done) begin
                        result_x <= reduce_result_x;
                        result_y <= reduce_result_y;
                        result_z <= reduce_result_z;
                        state    <= S_DONE;
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

endmodule
