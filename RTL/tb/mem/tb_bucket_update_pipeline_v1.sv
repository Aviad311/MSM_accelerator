`timescale 1ns/1ps

module tb_bucket_update_pipeline_v1;

    localparam int ADDR_W         = 8;
    localparam int DATA_W         = 256;
    localparam int DEPTH          = (1 << ADDR_W);
    localparam int GEN_W          = 16;
    localparam int SLOT_COUNT     = 16;
    localparam int MIX_CTX_COUNT  = 40;
    localparam int SRAM_RD_LATENCY = 3;
    localparam int NUM_UPDATES    = 7;

    localparam logic [255:0] ZERO =
        256'h0000000000000000000000000000000000000000000000000000000000000000;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    localparam logic [255:0] GX_M =
        256'h9981E643E9089F48979F48C033FD129C231E295329BC66DBD7362E5A487E2097;

    localparam logic [255:0] GY_M =
        256'hCF3F851FD4A582D670B6B59AAC19C1368DFC5D5D1F1DC64DB15EA6D2D3DBABE2;

    localparam logic [255:0] G2_X =
        256'h7C75DD9524177D593C03889B8DCD9B1CB05FB7D2A3DA7FE8BA9F29B104E7DB13;

    localparam logic [255:0] G2_Y =
        256'h55DEBB381F4AD034CC27CB48A46449AAA87D43FDB563384B1CD20838E6FDDC9F;

    localparam logic [255:0] G2_Z =
        256'h9E7F0A3FA94B05ACE16D6B355833826D1BF8BABA3E3B8C9B62BD4DA6A7B75B95;

    localparam logic [255:0] G3_X =
        256'h019FA59F6F459FC6748FA0A875006844FC39BED026E15B2769CD0E0931000A12;

    localparam logic [255:0] G3_Y =
        256'hF03F524E8729A2D670F5F5BE0A33EEDC2FC8D898B67B2802B68EF68395ABD131;

    localparam logic [255:0] G3_Z =
        256'hC2C26ED3E5BE9201DB856E0C5E96B76D5D182C134369ED8ECD3F6A303370697B;

    logic clk;
    logic rst_n;

    logic                  in_valid;
    logic                  in_ready;
    logic [GEN_W-1:0]      current_gen;
    logic [ADDR_W-1:0]     in_bucket_id;
    logic [DATA_W-1:0]     in_point_x;
    logic [DATA_W-1:0]     in_point_y;

    logic                  out_valid;
    logic                  out_ready;
    logic [ADDR_W-1:0]     out_bucket_id;
    logic                  out_skipped;
    logic                  out_direct_write;
    logic                  out_mixed_add;
    logic [DATA_W-1:0]     out_x;
    logic [DATA_W-1:0]     out_y;
    logic [DATA_W-1:0]     out_z;

    logic                  mem_valid;
    logic                  mem_write_en;
    logic [ADDR_W-1:0]     mem_addr;
    logic [DATA_W-1:0]     mem_wdata_x;
    logic [DATA_W-1:0]     mem_wdata_y;
    logic [DATA_W-1:0]     mem_wdata_z;
    logic                  mem_tag_write_en;
    logic [GEN_W-1:0]      mem_tag_wdata;

    logic                  mem_ready;
    logic                  mem_rvalid;
    logic [DATA_W-1:0]     mem_rdata_x;
    logic [DATA_W-1:0]     mem_rdata_y;
    logic [DATA_W-1:0]     mem_rdata_z;
    logic [GEN_W-1:0]      mem_tag_rdata;

    logic [$clog2(SLOT_COUNT+1)-1:0] active_slots;
    logic [63:0] accepted_count;
    logic [63:0] completed_count;
    logic [63:0] same_bucket_stall_count;
    logic [63:0] direct_write_count;
    logic [63:0] mixed_add_count;

    logic [DATA_W-1:0] mem_x [0:DEPTH-1];
    logic [DATA_W-1:0] mem_y [0:DEPTH-1];
    logic [DATA_W-1:0] mem_z [0:DEPTH-1];
    logic [GEN_W-1:0]  mem_tag [0:DEPTH-1];

    logic [SRAM_RD_LATENCY-1:0] rd_valid_pipe;
    logic [ADDR_W-1:0] rd_addr_pipe [0:SRAM_RD_LATENCY-1];

    logic [ADDR_W-1:0] update_bucket [0:NUM_UPDATES-1];

    int cycle_count;
    int send_count;
    int recv_count;
    int input_stall_cycles;
    int same_bucket_input_stalls;
    bit completion_seen [0:NUM_UPDATES-1];

    assign mem_ready = 1'b1;

    assign mem_rvalid    = rd_valid_pipe[SRAM_RD_LATENCY-1];
    assign mem_rdata_x   = mem_x[rd_addr_pipe[SRAM_RD_LATENCY-1]];
    assign mem_rdata_y   = mem_y[rd_addr_pipe[SRAM_RD_LATENCY-1]];
    assign mem_rdata_z   = mem_z[rd_addr_pipe[SRAM_RD_LATENCY-1]];
    assign mem_tag_rdata = mem_tag[rd_addr_pipe[SRAM_RD_LATENCY-1]];

    bucket_update_pipeline_v1 #(
        .ADDR_W(ADDR_W),
        .DATA_W(DATA_W),
        .DEPTH(DEPTH),
        .GEN_W(GEN_W),
        .SLOT_COUNT(SLOT_COUNT),
        .MIX_CTX_COUNT(MIX_CTX_COUNT),
        .MUL_LATENCY(16),
        .SKIP_ZERO_BUCKET(1'b1)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),

        .in_valid(in_valid),
        .in_ready(in_ready),
        .current_gen(current_gen),
        .in_bucket_id(in_bucket_id),
        .in_point_x(in_point_x),
        .in_point_y(in_point_y),

        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_bucket_id(out_bucket_id),
        .out_skipped(out_skipped),
        .out_direct_write(out_direct_write),
        .out_mixed_add(out_mixed_add),
        .out_x(out_x),
        .out_y(out_y),
        .out_z(out_z),

        .mem_valid(mem_valid),
        .mem_write_en(mem_write_en),
        .mem_addr(mem_addr),
        .mem_wdata_x(mem_wdata_x),
        .mem_wdata_y(mem_wdata_y),
        .mem_wdata_z(mem_wdata_z),
        .mem_tag_write_en(mem_tag_write_en),
        .mem_tag_wdata(mem_tag_wdata),

        .mem_ready(mem_ready),
        .mem_rvalid(mem_rvalid),
        .mem_rdata_x(mem_rdata_x),
        .mem_rdata_y(mem_rdata_y),
        .mem_rdata_z(mem_rdata_z),
        .mem_tag_rdata(mem_tag_rdata),

        .active_slots(active_slots),
        .accepted_count(accepted_count),
        .completed_count(completed_count),
        .same_bucket_stall_count(same_bucket_stall_count),
        .direct_write_count(direct_write_count),
        .mixed_add_count(mixed_add_count)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    // ------------------------------------------------------------------------
    // Simple synchronous 1RW SRAM model with generation tags
    // ------------------------------------------------------------------------
    integer i;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_valid_pipe <= '0;

            for (i = 0; i < SRAM_RD_LATENCY; i = i + 1)
                rd_addr_pipe[i] <= '0;

            for (i = 0; i < DEPTH; i = i + 1) begin
                mem_x[i]   <= ZERO;
                mem_y[i]   <= ONE_M;
                mem_z[i]   <= ZERO;
                mem_tag[i] <= '0;
            end
        end else begin
            for (i = SRAM_RD_LATENCY-1; i > 0; i = i - 1) begin
                rd_valid_pipe[i] <= rd_valid_pipe[i-1];
                rd_addr_pipe[i]  <= rd_addr_pipe[i-1];
            end

            rd_valid_pipe[0] <= mem_valid && !mem_write_en && mem_ready;

            if (mem_valid && !mem_write_en && mem_ready)
                rd_addr_pipe[0] <= mem_addr;

            if (mem_valid && mem_write_en && mem_ready) begin
                mem_x[mem_addr] <= mem_wdata_x;
                mem_y[mem_addr] <= mem_wdata_y;
                mem_z[mem_addr] <= mem_wdata_z;

                if (mem_tag_write_en)
                    mem_tag[mem_addr] <= mem_tag_wdata;

                $display("[MEM_WRITE] cycle=%0d bucket=%0d X=%064h",
                         cycle_count, mem_addr, mem_wdata_x);
            end
        end
    end

    task automatic load_update(input int idx);
        begin
            in_bucket_id = update_bucket[idx];
            in_point_x   = GX_M;
            in_point_y   = GY_M;
        end
    endtask

    // ------------------------------------------------------------------------
    // Scoreboard / completion checks
    // ------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count             <= 0;
            recv_count              <= 0;
            input_stall_cycles      <= 0;
            same_bucket_input_stalls <= 0;

            for (int k = 0; k < NUM_UPDATES; k++)
                completion_seen[k] <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;

            if (in_valid && !in_ready) begin
                input_stall_cycles <= input_stall_cycles + 1;

                if (send_count == 3 || send_count == 5)
                    same_bucket_input_stalls <=
                        same_bucket_input_stalls + 1;
            end

            if (out_valid && out_ready) begin
                $display("[UPDATE_DONE] cycle=%0d bucket=%0d skip=%0b direct=%0b mixed=%0b active=%0d",
                         cycle_count, out_bucket_id, out_skipped,
                         out_direct_write, out_mixed_add, active_slots);

                if (out_bucket_id == 0) begin
                    if (!out_skipped)
                        $fatal(1, "Bucket zero completion was not marked skipped");

                    if (out_x !== ZERO || out_y !== ONE_M || out_z !== ZERO)
                        $fatal(1, "Bucket zero completion returned wrong infinity");
                end else if (out_direct_write) begin
                    if (out_x !== GX_M || out_y !== GY_M || out_z !== ONE_M)
                        $fatal(1, "Direct-write completion mismatch for bucket %0d",
                               out_bucket_id);
                end else if (out_mixed_add) begin
                    if (out_bucket_id == 1) begin
                        // Bucket 1 receives three G updates total. Depending on
                        // which repeated update this is, the result is 2G or 3G.
                        if (!((out_x === G2_X && out_y === G2_Y && out_z === G2_Z) ||
                              (out_x === G3_X && out_y === G3_Y && out_z === G3_Z)))
                            $fatal(1, "Bucket 1 mixed result is neither 2G nor 3G");
                    end else if (out_bucket_id == 2) begin
                        if (out_x !== G2_X || out_y !== G2_Y || out_z !== G2_Z)
                            $fatal(1, "Bucket 2 mixed result mismatch");
                    end
                end

                recv_count <= recv_count + 1;
            end
        end
    end

    initial begin
        rst_n       = 1'b0;
        in_valid    = 1'b0;
        current_gen = 16'h002A;
        in_bucket_id = '0;
        in_point_x  = ZERO;
        in_point_y  = ZERO;
        out_ready   = 1'b1;
        send_count  = 0;

        // Intentionally repeated buckets:
        //   bucket 1: G -> 2G -> 3G
        //   bucket 2: G -> 2G
        //   bucket 3: G
        //   bucket 0: skipped
        update_bucket[0] = 8'd1;
        update_bucket[1] = 8'd2;
        update_bucket[2] = 8'd3;
        update_bucket[3] = 8'd1;
        update_bucket[4] = 8'd0;
        update_bucket[5] = 8'd2;
        update_bucket[6] = 8'd1;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        $display("====================================================");
        $display(" tb_bucket_update_pipeline_v1 START");
        $display(" independent buckets + repeated-bucket hazards");
        $display("====================================================");

        while (send_count < NUM_UPDATES) begin
            @(negedge clk);
            in_valid = 1'b1;
            load_update(send_count);

            @(posedge clk);
            if (in_ready) begin
                $display("[UPDATE_SEND] cycle=%0d index=%0d bucket=%0d active=%0d",
                         cycle_count, send_count, update_bucket[send_count],
                         active_slots);
                send_count++;
            end
        end

        @(negedge clk);
        in_valid = 1'b0;
        in_bucket_id = '0;
        in_point_x = ZERO;
        in_point_y = ZERO;

        wait (recv_count == NUM_UPDATES);
        repeat (8) @(posedge clk);

        // Final memory contents.
        if (mem_tag[1] !== current_gen ||
            mem_x[1] !== G3_X ||
            mem_y[1] !== G3_Y ||
            mem_z[1] !== G3_Z)
            $fatal(1, "Final bucket 1 is not 3G");

        if (mem_tag[2] !== current_gen ||
            mem_x[2] !== G2_X ||
            mem_y[2] !== G2_Y ||
            mem_z[2] !== G2_Z)
            $fatal(1, "Final bucket 2 is not 2G");

        if (mem_tag[3] !== current_gen ||
            mem_x[3] !== GX_M ||
            mem_y[3] !== GY_M ||
            mem_z[3] !== ONE_M)
            $fatal(1, "Final bucket 3 is not G");

        if (mem_tag[0] === current_gen)
            $fatal(1, "Bucket zero should not have been written");

        if (accepted_count !== NUM_UPDATES)
            $fatal(1, "accepted_count mismatch: %0d", accepted_count);

        if (completed_count !== NUM_UPDATES)
            $fatal(1, "completed_count mismatch: %0d", completed_count);

        if (direct_write_count !== 3)
            $fatal(1, "direct_write_count expected 3, got %0d",
                   direct_write_count);

        if (mixed_add_count !== 3)
            $fatal(1, "mixed_add_count expected 3, got %0d",
                   mixed_add_count);

        if (same_bucket_stall_count == 0)
            $fatal(1, "Expected at least one same-bucket stall");

        $display("====================================================");
        $display(" BUCKET UPDATE PIPELINE V1 PASSED");
        $display(" accepted_count          = %0d", accepted_count);
        $display(" completed_count         = %0d", completed_count);
        $display(" direct_write_count      = %0d", direct_write_count);
        $display(" mixed_add_count         = %0d", mixed_add_count);
        $display(" same_bucket_stall_count = %0d", same_bucket_stall_count);
        $display(" input_stall_cycles      = %0d", input_stall_cycles);
        $display(" total_cycles            = %0d", cycle_count);
        $display("====================================================");

        $finish;
    end

    initial begin
        #5000000;
        $fatal(1, "Timeout in tb_bucket_update_pipeline_v1");
    end

endmodule