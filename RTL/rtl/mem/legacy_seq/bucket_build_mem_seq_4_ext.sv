`timescale 1ns/1ps

module bucket_build_mem_seq_4_ext #(
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

    output logic [2:0]          processed_count,
    output logic [2:0]          skipped_count,

    output logic [DATA_W-1:0]   last_x,
    output logic [DATA_W-1:0]   last_y,
    output logic [DATA_W-1:0]   last_z,

    // External bucket memory interface
    output logic                mem_valid,
    output logic                mem_write_en,
    output logic [ADDR_W-1:0]   mem_addr,

    output logic [DATA_W-1:0]   mem_wdata_x,
    output logic [DATA_W-1:0]   mem_wdata_y,
    output logic [DATA_W-1:0]   mem_wdata_z,

    input  logic                mem_ready,
    input  logic                mem_rvalid,

    input  logic [DATA_W-1:0]   mem_rdata_x,
    input  logic [DATA_W-1:0]   mem_rdata_y,
    input  logic [DATA_W-1:0]   mem_rdata_z
);

    typedef enum logic [3:0] {
        S_IDLE,
        S_CLEAR_START,
        S_CLEAR_WAIT,
        S_UPDATE_START,
        S_UPDATE_WAIT,
        S_DONE
    } state_t;

    state_t state;

    logic [1:0] point_idx;

    logic update_start;
    logic update_clear_all;
    logic [ADDR_W-1:0] update_bucket_id;
    logic [DATA_W-1:0] update_point_x;
    logic [DATA_W-1:0] update_point_y;

    logic update_busy;
    logic update_done;
    logic update_skipped;

    logic [DATA_W-1:0] update_last_x;
    logic [DATA_W-1:0] update_last_y;
    logic [DATA_W-1:0] update_last_z;

    bucket_update_seq_ext #(
        .ADDR_W(ADDR_W),
        .DATA_W(DATA_W),
        .DEPTH (DEPTH)
    ) u_update_ext (
        .clk          (clk),
        .rst_n        (rst_n),

        .start        (update_start),
        .clear_all    (update_clear_all),

        .bucket_id    (update_bucket_id),
        .point_x      (update_point_x),
        .point_y      (update_point_y),

        .busy         (update_busy),
        .done         (update_done),
        .skipped      (update_skipped),

        .last_x       (update_last_x),
        .last_y       (update_last_y),
        .last_z       (update_last_z),

        .mem_valid    (mem_valid),
        .mem_write_en (mem_write_en),
        .mem_addr     (mem_addr),

        .mem_wdata_x  (mem_wdata_x),
        .mem_wdata_y  (mem_wdata_y),
        .mem_wdata_z  (mem_wdata_z),

        .mem_ready    (mem_ready),
        .mem_rvalid   (mem_rvalid),

        .mem_rdata_x  (mem_rdata_x),
        .mem_rdata_y  (mem_rdata_y),
        .mem_rdata_z  (mem_rdata_z)
    );

    assign busy = (state != S_IDLE);
    assign done = (state == S_DONE);

    always_comb begin
        update_start     = 1'b0;
        update_clear_all = 1'b0;
        update_bucket_id = '0;
        update_point_x   = '0;
        update_point_y   = '0;

        unique case (state)

            S_CLEAR_START: begin
                update_start     = 1'b1;
                update_clear_all = 1'b1;
            end

            S_UPDATE_START: begin
                update_start     = 1'b1;
                update_clear_all = 1'b0;

                unique case (point_idx)
                    2'd0: begin
                        update_bucket_id = b0;
                        update_point_x   = p0_x;
                        update_point_y   = p0_y;
                    end

                    2'd1: begin
                        update_bucket_id = b1;
                        update_point_x   = p1_x;
                        update_point_y   = p1_y;
                    end

                    2'd2: begin
                        update_bucket_id = b2;
                        update_point_x   = p2_x;
                        update_point_y   = p2_y;
                    end

                    2'd3: begin
                        update_bucket_id = b3;
                        update_point_x   = p3_x;
                        update_point_y   = p3_y;
                    end

                    default: begin
                        update_bucket_id = '0;
                        update_point_x   = '0;
                        update_point_y   = '0;
                    end
                endcase
            end

            default: begin
            end

        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            point_idx       <= '0;
            processed_count <= '0;
            skipped_count   <= '0;

            last_x          <= '0;
            last_y          <= '0;
            last_z          <= '0;
        end else begin
            unique case (state)

                S_IDLE: begin
                    if (start) begin
                        point_idx       <= '0;
                        processed_count <= '0;
                        skipped_count   <= '0;

                        last_x          <= '0;
                        last_y          <= '0;
                        last_z          <= '0;

                        state           <= S_CLEAR_START;
                    end
                end

                S_CLEAR_START: begin
                    state <= S_CLEAR_WAIT;
                end

                S_CLEAR_WAIT: begin
                    if (update_done) begin
                        state <= S_UPDATE_START;
                    end
                end

                S_UPDATE_START: begin
                    state <= S_UPDATE_WAIT;
                end

                S_UPDATE_WAIT: begin
                    if (update_done) begin
                        processed_count <= processed_count + 1'b1;

                        if (update_skipped) begin
                            skipped_count <= skipped_count + 1'b1;
                        end

                        last_x <= update_last_x;
                        last_y <= update_last_y;
                        last_z <= update_last_z;

                        if (point_idx == 2'd3) begin
                            state <= S_DONE;
                        end else begin
                            point_idx <= point_idx + 1'b1;
                            state     <= S_UPDATE_START;
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

endmodule