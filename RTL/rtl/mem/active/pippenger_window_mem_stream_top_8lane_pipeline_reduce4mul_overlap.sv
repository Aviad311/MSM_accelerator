`timescale 1ns/1ps

module pippenger_window_mem_stream_top_8lane_pipeline_reduce4mul_overlap #(
    parameter int ADDR_W          = 8,
    parameter int DATA_W          = 256,
    parameter int DEPTH           = (1 << ADDR_W),
    parameter int SRAM_RD_LATENCY = 3,
    parameter int GEN_W           = 16,

    parameter int FIFO_DEPTH      = 16,
    parameter int SLOT_COUNT      = 16,
    parameter int MIX_CTX_COUNT   = 40,
    parameter int MUL_LATENCY     = 16
)(
    input  logic                clk,
    input  logic                rst_n,

    input  logic                start,

    input  logic                in_valid,
    output logic                in_ready,
    input  logic [ADDR_W-1:0]   in_bucket_id,
    input  logic [DATA_W-1:0]   in_point_x,
    input  logic [DATA_W-1:0]   in_point_y,
    input  logic                last_point,

    output logic                busy,
    output logic                done,

    output logic [DATA_W-1:0]   result_x,
    output logic [DATA_W-1:0]   result_y,
    output logic [DATA_W-1:0]   result_z
);

    localparam int NUM_BANKS   = 8;
    localparam int BANK_SEL_W  = $clog2(NUM_BANKS);
    localparam int BANK_ADDR_W = ADDR_W - BANK_SEL_W;
    localparam int BANK_DEPTH  = (1 << BANK_ADDR_W);

    typedef enum logic [3:0] {
        S_TAG_INIT,
        S_IDLE,
        S_BUILD_START,
        S_BUILD_WAIT,
        S_REDUCE_START,
        S_REDUCE_WAIT,
        S_DONE
    } state_t;

    state_t state;

    logic [GEN_W-1:0]       current_gen;
    logic [BANK_ADDR_W-1:0] tag_init_addr;

    // ------------------------------------------------------------------------
    // New 8-lane pipelined build path
    // ------------------------------------------------------------------------
    logic scheduler_in_ready;

    logic scheduler_out_valid;
    logic scheduler_out_ready;
    logic [ADDR_W-1:0] scheduler_out_bucket_id;
    logic scheduler_out_skipped;
    logic scheduler_out_direct_write;
    logic scheduler_out_mixed_add;
    logic [DATA_W-1:0] scheduler_out_x;
    logic [DATA_W-1:0] scheduler_out_y;
    logic [DATA_W-1:0] scheduler_out_z;

    logic [63:0] scheduler_total_enqueue_count;
    logic [63:0] scheduler_total_issue_count;
    logic [63:0] scheduler_total_completed_count;
    logic [63:0] scheduler_total_bypass_count;
    logic [63:0] scheduler_total_fifo_full_stall_count;
    logic [63:0] scheduler_total_direct_write_count;
    logic [63:0] scheduler_total_mixed_add_count;

    logic [NUM_BANKS-1:0][$clog2(FIFO_DEPTH+1)-1:0]
        scheduler_lane_fifo_occupancy;

    logic [NUM_BANKS-1:0][$clog2(SLOT_COUNT+1)-1:0]
        scheduler_lane_active_slots;

    logic [NUM_BANKS-1:0]                    build_mem_valid;
    logic [NUM_BANKS-1:0]                    build_mem_write_en;
    logic [NUM_BANKS-1:0][BANK_ADDR_W-1:0]   build_mem_addr;
    logic [NUM_BANKS-1:0][DATA_W-1:0]        build_mem_wdata_x;
    logic [NUM_BANKS-1:0][DATA_W-1:0]        build_mem_wdata_y;
    logic [NUM_BANKS-1:0][DATA_W-1:0]        build_mem_wdata_z;
    logic [NUM_BANKS-1:0]                    build_mem_tag_write_en;
    logic [NUM_BANKS-1:0][GEN_W-1:0]         build_mem_tag_wdata;

    logic [63:0] build_completed_base;
    logic [63:0] build_accepted_count;
    logic        build_last_seen;
    logic        build_done;

    wire build_input_fire =
        (state == S_BUILD_WAIT) &&
        in_valid &&
        scheduler_in_ready;

    assign scheduler_out_ready = 1'b1;

    assign build_done =
        build_last_seen &&
        ((scheduler_total_completed_count - build_completed_base) ==
         build_accepted_count);

    // ------------------------------------------------------------------------
    // Reduce path
    // ------------------------------------------------------------------------
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

    logic reduce_mem_tag_write_en;
    logic [GEN_W-1:0] reduce_mem_tag_wdata;

    logic reduce_mem_ready;
    logic reduce_mem_rvalid;

    logic [DATA_W-1:0] reduce_mem_rdata_x;
    logic [DATA_W-1:0] reduce_mem_rdata_y;
    logic [DATA_W-1:0] reduce_mem_rdata_z;
    logic [GEN_W-1:0]  reduce_mem_tag_rdata;

    logic [BANK_SEL_W-1:0] reduce_req_bank_sel;

    logic [SRAM_RD_LATENCY-1:0] reduce_resp_valid_pipe;
    logic [SRAM_RD_LATENCY-1:0][BANK_SEL_W-1:0]
        reduce_resp_bank_pipe;

    logic reduce_resp_valid_sel;
    logic [BANK_SEL_W-1:0] reduce_resp_bank_sel;

    assign reduce_req_bank_sel =
        reduce_mem_addr[BANK_SEL_W-1:0];

    assign reduce_resp_valid_sel =
        reduce_resp_valid_pipe[SRAM_RD_LATENCY-1];

    assign reduce_resp_bank_sel =
        reduce_resp_bank_pipe[SRAM_RD_LATENCY-1];

    // ------------------------------------------------------------------------
    // Physical bank interfaces
    // ------------------------------------------------------------------------
    logic [NUM_BANKS-1:0]                    mem_valid;
    logic [NUM_BANKS-1:0]                    mem_write_en;
    logic [NUM_BANKS-1:0][BANK_ADDR_W-1:0]   mem_addr;
    logic [NUM_BANKS-1:0][DATA_W-1:0]        mem_wdata_x;
    logic [NUM_BANKS-1:0][DATA_W-1:0]        mem_wdata_y;
    logic [NUM_BANKS-1:0][DATA_W-1:0]        mem_wdata_z;
    logic [NUM_BANKS-1:0]                    mem_tag_write_en;
    logic [NUM_BANKS-1:0][GEN_W-1:0]         mem_tag_wdata;

    logic [NUM_BANKS-1:0]                    mem_ready;
    logic [NUM_BANKS-1:0]                    mem_rvalid;
    logic [NUM_BANKS-1:0][DATA_W-1:0]        mem_rdata_x;
    logic [NUM_BANKS-1:0][DATA_W-1:0]        mem_rdata_y;
    logic [NUM_BANKS-1:0][DATA_W-1:0]        mem_rdata_z;
    logic [NUM_BANKS-1:0][GEN_W-1:0]         mem_tag_rdata;

    assign busy = (state != S_IDLE);
    assign done = (state == S_DONE);

    assign in_ready =
        (state == S_BUILD_WAIT) ? scheduler_in_ready : 1'b0;

    // ------------------------------------------------------------------------
    // Eight independent bucket SRAM banks
    // ------------------------------------------------------------------------
    genvar bi;
    generate
        for (bi = 0; bi < NUM_BANKS; bi = bi + 1) begin : g_bucket_mem
            bucket_mem_3coord #(
                .ADDR_W          (BANK_ADDR_W),
                .DATA_W          (DATA_W),
                .DEPTH           (BANK_DEPTH),
                .SRAM_RD_LATENCY (SRAM_RD_LATENCY),
                .GEN_W           (GEN_W)
            ) u_bucket_mem (
                .clk              (clk),
                .rst_n            (rst_n),
                .valid            (mem_valid[bi]),
                .write_en         (mem_write_en[bi]),
                .addr             (mem_addr[bi]),
                .wdata_x          (mem_wdata_x[bi]),
                .wdata_y          (mem_wdata_y[bi]),
                .wdata_z          (mem_wdata_z[bi]),
                .tag_write_en     (mem_tag_write_en[bi]),
                .tag_wdata        (mem_tag_wdata[bi]),
                .ready            (mem_ready[bi]),
                .rvalid           (mem_rvalid[bi]),
                .rdata_x          (mem_rdata_x[bi]),
                .rdata_y          (mem_rdata_y[bi]),
                .rdata_z          (mem_rdata_z[bi]),
                .tag_rdata        (mem_tag_rdata[bi])
            );
        end
    endgenerate

    // ------------------------------------------------------------------------
    // New build engine:
    //   8 lanes x one MixedAdd pipeline v2 x four multipliers = 32 multipliers
    // ------------------------------------------------------------------------
    bucket_update_scheduler_8lane_v1 #(
        .LANES            (NUM_BANKS),
        .GLOBAL_ADDR_W    (ADDR_W),
        .DATA_W           (DATA_W),
        .GEN_W            (GEN_W),
        .FIFO_DEPTH       (FIFO_DEPTH),
        .SLOT_COUNT       (SLOT_COUNT),
        .MIX_CTX_COUNT    (MIX_CTX_COUNT),
        .MUL_LATENCY      (MUL_LATENCY),
        .SKIP_ZERO_BUCKET (1'b1)
    ) u_build_scheduler_8lane (
        .clk                         (clk),
        .rst_n                       (rst_n),

        .in_valid                    (
            (state == S_BUILD_WAIT) ? in_valid : 1'b0
        ),
        .in_ready                    (scheduler_in_ready),
        .current_gen                 (current_gen),
        .in_bucket_id                (in_bucket_id),
        .in_point_x                  (in_point_x),
        .in_point_y                  (in_point_y),

        .out_valid                   (scheduler_out_valid),
        .out_ready                   (scheduler_out_ready),
        .out_bucket_id               (scheduler_out_bucket_id),
        .out_skipped                 (scheduler_out_skipped),
        .out_direct_write            (scheduler_out_direct_write),
        .out_mixed_add               (scheduler_out_mixed_add),
        .out_x                       (scheduler_out_x),
        .out_y                       (scheduler_out_y),
        .out_z                       (scheduler_out_z),

        .mem_valid                   (build_mem_valid),
        .mem_write_en                (build_mem_write_en),
        .mem_addr                    (build_mem_addr),
        .mem_wdata_x                 (build_mem_wdata_x),
        .mem_wdata_y                 (build_mem_wdata_y),
        .mem_wdata_z                 (build_mem_wdata_z),
        .mem_tag_write_en            (build_mem_tag_write_en),
        .mem_tag_wdata               (build_mem_tag_wdata),

        .mem_ready                   (mem_ready),
        .mem_rvalid                  (mem_rvalid),
        .mem_rdata_x                 (mem_rdata_x),
        .mem_rdata_y                 (mem_rdata_y),
        .mem_rdata_z                 (mem_rdata_z),
        .mem_tag_rdata               (mem_tag_rdata),

        .total_enqueue_count         (scheduler_total_enqueue_count),
        .total_issue_count           (scheduler_total_issue_count),
        .total_completed_count       (scheduler_total_completed_count),
        .total_bypass_count          (scheduler_total_bypass_count),
        .total_fifo_full_stall_count (
            scheduler_total_fifo_full_stall_count
        ),
        .total_direct_write_count    (
            scheduler_total_direct_write_count
        ),
        .total_mixed_add_count       (
            scheduler_total_mixed_add_count
        ),
        .lane_fifo_occupancy         (
            scheduler_lane_fifo_occupancy
        ),
        .lane_active_slots           (
            scheduler_lane_active_slots
        )
    );

    reduce_buckets_mem_4mul_overlap #(
        .ADDR_W (ADDR_W),
        .DATA_W (DATA_W),
        .GEN_W  (GEN_W)
    ) u_reduce (
        .clk              (clk),
        .rst_n            (rst_n),
        .start            (reduce_start),
        .current_gen      (current_gen),

        .busy             (reduce_busy),
        .done             (reduce_done),

        .result_x         (reduce_result_x),
        .result_y         (reduce_result_y),
        .result_z         (reduce_result_z),

        .mem_valid        (reduce_mem_valid),
        .mem_write_en     (reduce_mem_write_en),
        .mem_addr         (reduce_mem_addr),
        .mem_wdata_x      (reduce_mem_wdata_x),
        .mem_wdata_y      (reduce_mem_wdata_y),
        .mem_wdata_z      (reduce_mem_wdata_z),
        .mem_tag_write_en (reduce_mem_tag_write_en),
        .mem_tag_wdata    (reduce_mem_tag_wdata),

        .mem_ready        (reduce_mem_ready),
        .mem_rvalid       (reduce_mem_rvalid),

        .mem_rdata_x      (reduce_mem_rdata_x),
        .mem_rdata_y      (reduce_mem_rdata_y),
        .mem_rdata_z      (reduce_mem_rdata_z),
        .mem_tag_rdata    (reduce_mem_tag_rdata)
    );

    // Track which bank owns each reduce read response.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reduce_resp_valid_pipe <= '0;
            reduce_resp_bank_pipe  <= '0;
        end else begin
            reduce_resp_valid_pipe[0] <=
                ((state == S_REDUCE_START ||
                  state == S_REDUCE_WAIT) &&
                 reduce_mem_valid &&
                 !reduce_mem_write_en &&
                 reduce_mem_ready);

            reduce_resp_bank_pipe[0] <= reduce_req_bank_sel;

            for (int i = 1; i < SRAM_RD_LATENCY; i = i + 1) begin
                reduce_resp_valid_pipe[i] <=
                    reduce_resp_valid_pipe[i-1];

                reduce_resp_bank_pipe[i] <=
                    reduce_resp_bank_pipe[i-1];
            end
        end
    end

    assign reduce_mem_ready =
        mem_ready[reduce_req_bank_sel];

    assign reduce_mem_rvalid =
        reduce_resp_valid_sel &&
        mem_rvalid[reduce_resp_bank_sel];

    assign reduce_mem_rdata_x =
        mem_rdata_x[reduce_resp_bank_sel];

    assign reduce_mem_rdata_y =
        mem_rdata_y[reduce_resp_bank_sel];

    assign reduce_mem_rdata_z =
        mem_rdata_z[reduce_resp_bank_sel];

    assign reduce_mem_tag_rdata =
        mem_tag_rdata[reduce_resp_bank_sel];

    // ------------------------------------------------------------------------
    // SRAM ownership mux: init, build, or reduce
    // ------------------------------------------------------------------------
    always_comb begin
        for (int i = 0; i < NUM_BANKS; i = i + 1) begin
            mem_valid[i]        = 1'b0;
            mem_write_en[i]     = 1'b0;
            mem_addr[i]         = '0;
            mem_wdata_x[i]      = '0;
            mem_wdata_y[i]      = '0;
            mem_wdata_z[i]      = '0;
            mem_tag_write_en[i] = 1'b0;
            mem_tag_wdata[i]    = '0;
        end

        reduce_start = 1'b0;

        unique case (state)

            S_TAG_INIT: begin
                for (int i = 0; i < NUM_BANKS; i = i + 1) begin
                    mem_valid[i]        = 1'b1;
                    mem_write_en[i]     = 1'b0;
                    mem_addr[i]         = tag_init_addr;
                    mem_tag_write_en[i] = 1'b1;
                    mem_tag_wdata[i]    = '0;
                end
            end

            S_BUILD_START,
            S_BUILD_WAIT: begin
                for (int i = 0; i < NUM_BANKS; i = i + 1) begin
                    mem_valid[i]        = build_mem_valid[i];
                    mem_write_en[i]     = build_mem_write_en[i];
                    mem_addr[i]         = build_mem_addr[i];
                    mem_wdata_x[i]      = build_mem_wdata_x[i];
                    mem_wdata_y[i]      = build_mem_wdata_y[i];
                    mem_wdata_z[i]      = build_mem_wdata_z[i];
                    mem_tag_write_en[i] = build_mem_tag_write_en[i];
                    mem_tag_wdata[i]    = build_mem_tag_wdata[i];
                end
            end

            S_REDUCE_START,
            S_REDUCE_WAIT: begin
                if (state == S_REDUCE_START)
                    reduce_start = 1'b1;

                if (reduce_mem_valid) begin
                    mem_valid[reduce_req_bank_sel] =
                        reduce_mem_valid;

                    mem_write_en[reduce_req_bank_sel] =
                        reduce_mem_write_en;

                    mem_addr[reduce_req_bank_sel] =
                        reduce_mem_addr[ADDR_W-1:BANK_SEL_W];

                    mem_wdata_x[reduce_req_bank_sel] =
                        reduce_mem_wdata_x;

                    mem_wdata_y[reduce_req_bank_sel] =
                        reduce_mem_wdata_y;

                    mem_wdata_z[reduce_req_bank_sel] =
                        reduce_mem_wdata_z;

                    mem_tag_write_en[reduce_req_bank_sel] =
                        reduce_mem_tag_write_en;

                    mem_tag_wdata[reduce_req_bank_sel] =
                        reduce_mem_tag_wdata;
                end
            end

            default: begin
            end
        endcase
    end

    // ------------------------------------------------------------------------
    // Top-level control and per-build completion tracking
    // ------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state                <= S_TAG_INIT;
            tag_init_addr        <= '0;
            current_gen          <= {{(GEN_W-1){1'b0}}, 1'b1};

            build_completed_base <= 64'd0;
            build_accepted_count <= 64'd0;
            build_last_seen      <= 1'b0;

            result_x             <= '0;
            result_y             <= '0;
            result_z             <= '0;
        end else begin
            unique case (state)

                S_TAG_INIT: begin
                    if (&mem_ready) begin
                        if (tag_init_addr ==
                            {BANK_ADDR_W{1'b1}}) begin

                            state <= S_IDLE;
                        end else begin
                            tag_init_addr <=
                                tag_init_addr + 1'b1;
                        end
                    end
                end

                S_IDLE: begin
                    if (start) begin
                        current_gen <= current_gen + 1'b1;

                        build_completed_base <=
                            scheduler_total_completed_count;

                        build_accepted_count <= 64'd0;
                        build_last_seen      <= 1'b0;

                        result_x <= '0;
                        result_y <= '0;
                        result_z <= '0;

                        state <= S_BUILD_START;
                    end
                end

                S_BUILD_START: begin
                    // One setup cycle before accepting the stream.
                    state <= S_BUILD_WAIT;
                end

                S_BUILD_WAIT: begin
                    if (build_input_fire) begin
                        build_accepted_count <=
                            build_accepted_count + 64'd1;

                        if (last_point)
                            build_last_seen <= 1'b1;
                    end

                    if (build_done)
                        state <= S_REDUCE_START;
                end

                S_REDUCE_START: begin
                    state <= S_REDUCE_WAIT;
                end

                S_REDUCE_WAIT: begin
                    if (reduce_done) begin
                        result_x <= reduce_result_x;
                        result_y <= reduce_result_y;
                        result_z <= reduce_result_z;
                        state    <= S_DONE;
                    end
                end

                S_DONE: begin
                    state <= S_IDLE;
                end

                default: begin
                    state <= S_TAG_INIT;
                end
            endcase
        end
    end

endmodule