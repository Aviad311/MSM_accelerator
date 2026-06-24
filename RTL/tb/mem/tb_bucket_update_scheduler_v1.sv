`timescale 1ns/1ps

module tb_bucket_update_scheduler_v1;

    localparam int ADDR_W          = 8;
    localparam int DATA_W          = 256;
    localparam int DEPTH           = (1 << ADDR_W);
    localparam int GEN_W           = 16;
    localparam int FIFO_DEPTH      = 8;
    localparam int SLOT_COUNT      = 16;
    localparam int MIX_CTX_COUNT   = 40;
    localparam int SRAM_RD_LATENCY = 3;
    localparam int NUM_UPDATES     = 6;

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

    logic [$clog2(FIFO_DEPTH+1)-1:0] fifo_occupancy;
    logic [63:0] enqueue_count;
    logic [63:0] issue_count;
    logic [63:0] bypass_count;
    logic [63:0] fifo_full_stall_count;
    logic issue_pulse;
    logic [ADDR_W-1:0] issue_bucket_id;

    logic [$clog2(SLOT_COUNT+1)-1:0] active_slots;
    logic [63:0] accepted_count;
    logic [63:0] completed_count;
    logic [63:0] downstream_same_bucket_stall_count;
    logic [63:0] direct_write_count;
    logic [63:0] mixed_add_count;

    logic [DATA_W-1:0] mem_x [0:DEPTH-1];
    logic [DATA_W-1:0] mem_y [0:DEPTH-1];
    logic [DATA_W-1:0] mem_z [0:DEPTH-1];
    logic [GEN_W-1:0]  mem_tag [0:DEPTH-1];

    logic [SRAM_RD_LATENCY-1:0] rd_valid_pipe;
    logic [ADDR_W-1:0] rd_addr_pipe [0:SRAM_RD_LATENCY-1];

    logic [ADDR_W-1:0] updates [0:NUM_UPDATES-1];

    int cycle_count;
    int send_count;
    int recv_count;
    int issue_seen_count;
    int first_bucket1_issue_position;
    int second_bucket1_issue_position;
    int bucket4_issue_position;
    int bucket5_issue_position;

    assign mem_ready = 1'b1;

    assign mem_rvalid    = rd_valid_pipe[SRAM_RD_LATENCY-1];
    assign mem_rdata_x   = mem_x[rd_addr_pipe[SRAM_RD_LATENCY-1]];
    assign mem_rdata_y   = mem_y[rd_addr_pipe[SRAM_RD_LATENCY-1]];
    assign mem_rdata_z   = mem_z[rd_addr_pipe[SRAM_RD_LATENCY-1]];
    assign mem_tag_rdata = mem_tag[rd_addr_pipe[SRAM_RD_LATENCY-1]];

    bucket_update_scheduler_v1 #(
        .ADDR_W(ADDR_W),
        .DATA_W(DATA_W),
        .DEPTH(DEPTH),
        .GEN_W(GEN_W),
        .FIFO_DEPTH(FIFO_DEPTH),
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

        .fifo_occupancy(fifo_occupancy),
        .enqueue_count(enqueue_count),
        .issue_count(issue_count),
        .bypass_count(bypass_count),
        .fifo_full_stall_count(fifo_full_stall_count),
        .issue_pulse(issue_pulse),
        .issue_bucket_id(issue_bucket_id),

        .active_slots(active_slots),
        .accepted_count(accepted_count),
        .completed_count(completed_count),
        .downstream_same_bucket_stall_count(
            downstream_same_bucket_stall_count
        ),
        .direct_write_count(direct_write_count),
        .mixed_add_count(mixed_add_count)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

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

                $display("[SCHED_MEM_WRITE] cycle=%0d bucket=%0d",
                         cycle_count, mem_addr);
            end
        end
    end

    task automatic drive_update(input int idx);
        begin
            in_bucket_id = updates[idx];
            in_point_x   = GX_M;
            in_point_y   = GY_M;
        end
    endtask

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count                  <= 0;
            recv_count                   <= 0;
            issue_seen_count             <= 0;
            first_bucket1_issue_position <= -1;
            second_bucket1_issue_position <= -1;
            bucket4_issue_position       <= -1;
            bucket5_issue_position       <= -1;
        end else begin
            cycle_count <= cycle_count + 1;

            if (issue_pulse) begin
                $display("[SCHED_ISSUE] cycle=%0d position=%0d bucket=%0d fifo_occ=%0d",
                         cycle_count, issue_seen_count, issue_bucket_id,
                         fifo_occupancy);

                if (issue_bucket_id == 1) begin
                    if (first_bucket1_issue_position < 0)
                        first_bucket1_issue_position <= issue_seen_count;
                    else
                        second_bucket1_issue_position <= issue_seen_count;
                end

                if (issue_bucket_id == 4)
                    bucket4_issue_position <= issue_seen_count;

                if (issue_bucket_id == 5)
                    bucket5_issue_position <= issue_seen_count;

                issue_seen_count <= issue_seen_count + 1;
            end

            if (out_valid && out_ready) begin
                $display("[SCHED_DONE] cycle=%0d bucket=%0d direct=%0b mixed=%0b skipped=%0b",
                         cycle_count, out_bucket_id, out_direct_write,
                         out_mixed_add, out_skipped);
                recv_count <= recv_count + 1;
            end
        end
    end

    initial begin
        rst_n        = 1'b0;
        in_valid     = 1'b0;
        current_gen  = 16'h0033;
        in_bucket_id = '0;
        in_point_x   = ZERO;
        in_point_y   = ZERO;
        out_ready    = 1'b1;
        send_count   = 0;

        // The second bucket 1 entry is intentionally placed at the head of
        // the waiting queue. Buckets 4, 5, 2, and 0 must bypass it.
        updates[0] = 8'd1;
        updates[1] = 8'd1;
        updates[2] = 8'd4;
        updates[3] = 8'd5;
        updates[4] = 8'd2;
        updates[5] = 8'd0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        $display("====================================================");
        $display(" tb_bucket_update_scheduler_v1 START");
        $display(" repeated bucket at queue head + independent bypass");
        $display("====================================================");

        while (send_count < NUM_UPDATES) begin
            @(negedge clk);
            in_valid = 1'b1;
            drive_update(send_count);

            @(posedge clk);
            if (in_ready) begin
                $display("[SCHED_ENQUEUE] cycle=%0d index=%0d bucket=%0d fifo_occ=%0d",
                         cycle_count, send_count, updates[send_count],
                         fifo_occupancy);
                send_count++;
            end
        end

        @(negedge clk);
        in_valid = 1'b0;
        in_bucket_id = '0;

        wait (recv_count == NUM_UPDATES);
        repeat (8) @(posedge clk);

        if (first_bucket1_issue_position < 0 ||
            second_bucket1_issue_position < 0)
            $fatal(1, "Did not observe both bucket 1 issues");

        if (bucket4_issue_position < 0 || bucket5_issue_position < 0)
            $fatal(1, "Did not observe bucket 4/5 issues");

        if (!(bucket4_issue_position < second_bucket1_issue_position))
            $fatal(1, "Bucket 4 did not bypass blocked bucket 1");

        if (!(bucket5_issue_position < second_bucket1_issue_position))
            $fatal(1, "Bucket 5 did not bypass blocked bucket 1");

        if (bypass_count == 0)
            $fatal(1, "Expected non-zero bypass_count");

        // The scheduler must prevent blocked same-bucket requests from ever
        // being presented to the downstream engine.
        if (downstream_same_bucket_stall_count != 0)
            $fatal(1,
                "Downstream saw same-bucket stalls: %0d",
                downstream_same_bucket_stall_count);

        if (enqueue_count != NUM_UPDATES ||
            issue_count != NUM_UPDATES ||
            accepted_count != NUM_UPDATES ||
            completed_count != NUM_UPDATES)
            $fatal(1,
                "Count mismatch enq=%0d issue=%0d accepted=%0d completed=%0d",
                enqueue_count, issue_count, accepted_count, completed_count);

        if (fifo_full_stall_count != 0)
            $fatal(1, "Unexpected FIFO-full stalls: %0d",
                   fifo_full_stall_count);

        if (mem_tag[1] !== current_gen ||
            mem_x[1] !== G2_X ||
            mem_y[1] !== G2_Y ||
            mem_z[1] !== G2_Z)
            $fatal(1, "Final bucket 1 is not 2G");

        if (mem_tag[4] !== current_gen ||
            mem_x[4] !== GX_M ||
            mem_y[4] !== GY_M ||
            mem_z[4] !== ONE_M)
            $fatal(1, "Final bucket 4 is not G");

        if (mem_tag[5] !== current_gen ||
            mem_x[5] !== GX_M ||
            mem_y[5] !== GY_M ||
            mem_z[5] !== ONE_M)
            $fatal(1, "Final bucket 5 is not G");

        if (mem_tag[2] !== current_gen ||
            mem_x[2] !== GX_M ||
            mem_y[2] !== GY_M ||
            mem_z[2] !== ONE_M)
            $fatal(1, "Final bucket 2 is not G");

        if (mem_tag[0] === current_gen)
            $fatal(1, "Bucket zero should not have been written");

        $display("====================================================");
        $display(" BUCKET UPDATE SCHEDULER V1 PASSED");
        $display(" enqueue_count                    = %0d", enqueue_count);
        $display(" issue_count                      = %0d", issue_count);
        $display(" bypass_count                     = %0d", bypass_count);
        $display(" fifo_full_stall_count             = %0d", fifo_full_stall_count);
        $display(" downstream_same_bucket_stalls    = %0d",
                 downstream_same_bucket_stall_count);
        $display(" direct_write_count                = %0d", direct_write_count);
        $display(" mixed_add_count                   = %0d", mixed_add_count);
        $display(" total_cycles                      = %0d", cycle_count);
        $display("====================================================");

        $finish;
    end

    initial begin
        #5000000;
        $fatal(1, "Timeout in tb_bucket_update_scheduler_v1");
    end

endmodule