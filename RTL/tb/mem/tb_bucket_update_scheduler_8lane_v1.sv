`timescale 1ns/1ps

module tb_bucket_update_scheduler_8lane_v1;

    localparam int LANES           = 8;
    localparam int GLOBAL_ADDR_W   = 8;
    localparam int LANE_W          = $clog2(LANES);
    localparam int LOCAL_ADDR_W    = GLOBAL_ADDR_W - LANE_W;
    localparam int LOCAL_DEPTH     = (1 << LOCAL_ADDR_W);
    localparam int DATA_W          = 256;
    localparam int GEN_W           = 16;
    localparam int FIFO_DEPTH      = 16;
    localparam int SLOT_COUNT      = 16;
    localparam int MIX_CTX_COUNT   = 40;
    localparam int RD_LATENCY      = 3;
    localparam int NUM_UPDATES     = 24;

    localparam logic [255:0] ZERO =
        256'h0;
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

    logic clk, rst_n;
    logic in_valid, in_ready;
    logic [GEN_W-1:0] current_gen;
    logic [GLOBAL_ADDR_W-1:0] in_bucket_id;
    logic [DATA_W-1:0] in_point_x, in_point_y;

    logic out_valid, out_ready;
    logic [GLOBAL_ADDR_W-1:0] out_bucket_id;
    logic out_skipped, out_direct_write, out_mixed_add;
    logic [DATA_W-1:0] out_x, out_y, out_z;

    logic [LANES-1:0] mem_valid, mem_write_en;
    logic [LANES-1:0][LOCAL_ADDR_W-1:0] mem_addr;
    logic [LANES-1:0][DATA_W-1:0] mem_wdata_x, mem_wdata_y, mem_wdata_z;
    logic [LANES-1:0] mem_tag_write_en;
    logic [LANES-1:0][GEN_W-1:0] mem_tag_wdata;
    logic [LANES-1:0] mem_ready, mem_rvalid;
    logic [LANES-1:0][DATA_W-1:0] mem_rdata_x, mem_rdata_y, mem_rdata_z;
    logic [LANES-1:0][GEN_W-1:0] mem_tag_rdata;

    logic [63:0] total_enqueue_count, total_issue_count;
    logic [63:0] total_completed_count, total_bypass_count;
    logic [63:0] total_fifo_full_stall_count;
    logic [63:0] total_direct_write_count, total_mixed_add_count;
    logic [LANES-1:0][$clog2(FIFO_DEPTH+1)-1:0] lane_fifo_occupancy;
    logic [LANES-1:0][$clog2(SLOT_COUNT+1)-1:0] lane_active_slots;

    logic [DATA_W-1:0] mem_x [0:LANES-1][0:LOCAL_DEPTH-1];
    logic [DATA_W-1:0] mem_y [0:LANES-1][0:LOCAL_DEPTH-1];
    logic [DATA_W-1:0] mem_z [0:LANES-1][0:LOCAL_DEPTH-1];
    logic [GEN_W-1:0] mem_tag [0:LANES-1][0:LOCAL_DEPTH-1];

    logic [LANES-1:0][RD_LATENCY-1:0] rd_valid_pipe;
    logic [LANES-1:0][RD_LATENCY-1:0][LOCAL_ADDR_W-1:0] rd_addr_pipe;

    logic [GLOBAL_ADDR_W-1:0] updates [0:NUM_UPDATES-1];

    int cycle_count, send_count, recv_count;

    assign mem_ready = '1;

    genvar g;
    generate
        for (g = 0; g < LANES; g = g + 1) begin : G_MEM_OUT
            assign mem_rvalid[g]    = rd_valid_pipe[g][RD_LATENCY-1];
            assign mem_rdata_x[g]   = mem_x[g][rd_addr_pipe[g][RD_LATENCY-1]];
            assign mem_rdata_y[g]   = mem_y[g][rd_addr_pipe[g][RD_LATENCY-1]];
            assign mem_rdata_z[g]   = mem_z[g][rd_addr_pipe[g][RD_LATENCY-1]];
            assign mem_tag_rdata[g] = mem_tag[g][rd_addr_pipe[g][RD_LATENCY-1]];
        end
    endgenerate

    bucket_update_scheduler_8lane_v1 #(
        .LANES(LANES),
        .GLOBAL_ADDR_W(GLOBAL_ADDR_W),
        .DATA_W(DATA_W),
        .GEN_W(GEN_W),
        .FIFO_DEPTH(FIFO_DEPTH),
        .SLOT_COUNT(SLOT_COUNT),
        .MIX_CTX_COUNT(MIX_CTX_COUNT),
        .MUL_LATENCY(16)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid), .in_ready(in_ready),
        .current_gen(current_gen), .in_bucket_id(in_bucket_id),
        .in_point_x(in_point_x), .in_point_y(in_point_y),
        .out_valid(out_valid), .out_ready(out_ready),
        .out_bucket_id(out_bucket_id), .out_skipped(out_skipped),
        .out_direct_write(out_direct_write), .out_mixed_add(out_mixed_add),
        .out_x(out_x), .out_y(out_y), .out_z(out_z),
        .mem_valid(mem_valid), .mem_write_en(mem_write_en), .mem_addr(mem_addr),
        .mem_wdata_x(mem_wdata_x), .mem_wdata_y(mem_wdata_y),
        .mem_wdata_z(mem_wdata_z), .mem_tag_write_en(mem_tag_write_en),
        .mem_tag_wdata(mem_tag_wdata), .mem_ready(mem_ready),
        .mem_rvalid(mem_rvalid), .mem_rdata_x(mem_rdata_x),
        .mem_rdata_y(mem_rdata_y), .mem_rdata_z(mem_rdata_z),
        .mem_tag_rdata(mem_tag_rdata),
        .total_enqueue_count(total_enqueue_count),
        .total_issue_count(total_issue_count),
        .total_completed_count(total_completed_count),
        .total_bypass_count(total_bypass_count),
        .total_fifo_full_stall_count(total_fifo_full_stall_count),
        .total_direct_write_count(total_direct_write_count),
        .total_mixed_add_count(total_mixed_add_count),
        .lane_fifo_occupancy(lane_fifo_occupancy),
        .lane_active_slots(lane_active_slots)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    integer lane_i;
    integer pipe_i;
    integer addr_i;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_valid_pipe <= '0;
            rd_addr_pipe  <= '0;

            for (lane_i = 0; lane_i < LANES; lane_i = lane_i + 1)
                for (addr_i = 0; addr_i < LOCAL_DEPTH; addr_i = addr_i + 1) begin
                    mem_x[lane_i][addr_i]   <= ZERO;
                    mem_y[lane_i][addr_i]   <= ONE_M;
                    mem_z[lane_i][addr_i]   <= ZERO;
                    mem_tag[lane_i][addr_i] <= '0;
                end
        end else begin
            for (lane_i = 0; lane_i < LANES; lane_i = lane_i + 1) begin
                for (pipe_i = RD_LATENCY-1; pipe_i > 0; pipe_i = pipe_i - 1) begin
                    rd_valid_pipe[lane_i][pipe_i] <= rd_valid_pipe[lane_i][pipe_i-1];
                    rd_addr_pipe[lane_i][pipe_i]  <= rd_addr_pipe[lane_i][pipe_i-1];
                end

                rd_valid_pipe[lane_i][0] <=
                    mem_valid[lane_i] && !mem_write_en[lane_i] && mem_ready[lane_i];

                if (mem_valid[lane_i] && !mem_write_en[lane_i] && mem_ready[lane_i])
                    rd_addr_pipe[lane_i][0] <= mem_addr[lane_i];

                if (mem_valid[lane_i] && mem_write_en[lane_i] && mem_ready[lane_i]) begin
                    mem_x[lane_i][mem_addr[lane_i]] <= mem_wdata_x[lane_i];
                    mem_y[lane_i][mem_addr[lane_i]] <= mem_wdata_y[lane_i];
                    mem_z[lane_i][mem_addr[lane_i]] <= mem_wdata_z[lane_i];

                    if (mem_tag_write_en[lane_i])
                        mem_tag[lane_i][mem_addr[lane_i]] <= mem_tag_wdata[lane_i];
                end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 0;
            recv_count  <= 0;
        end else begin
            cycle_count <= cycle_count + 1;

            if (out_valid && out_ready) begin
                recv_count <= recv_count + 1;
                $display("[8L_DONE] cycle=%0d bucket=%0d direct=%0b mixed=%0b",
                         cycle_count, out_bucket_id,
                         out_direct_write, out_mixed_add);
            end
        end
    end

    initial begin
        rst_n = 1'b0;
        in_valid = 1'b0;
        current_gen = 16'h0088;
        in_bucket_id = '0;
        in_point_x = GX_M;
        in_point_y = GY_M;
        out_ready = 1'b1;
        send_count = 0;

        // First 8 updates: one bucket per lane (global IDs 8..15).
        // Second 8: repeat those same buckets -> one MixedAdd per lane.
        // Last 8: independent new bucket per lane (16..23).
        for (int k = 0; k < 8; k++) begin
            updates[k]      = 8 + k;
            updates[8+k]    = 8 + k;
            updates[16+k]   = 16 + k;
        end

        repeat (6) @(posedge clk);
        rst_n = 1'b1;
        repeat (3) @(posedge clk);

        $display("====================================================");
        $display(" tb_bucket_update_scheduler_8lane_v1 START");
        $display(" 8 lanes x 4 multipliers = 32 Montgomery multipliers");
        $display("====================================================");

        while (send_count < NUM_UPDATES) begin
            @(negedge clk);
            in_valid = 1'b1;
            in_bucket_id = updates[send_count];

            @(posedge clk);
            if (in_ready) begin
                $display("[8L_SEND] cycle=%0d idx=%0d bucket=%0d lane=%0d",
                         cycle_count, send_count, updates[send_count],
                         updates[send_count][LANE_W-1:0]);
                send_count++;
            end
        end

        @(negedge clk);
        in_valid = 1'b0;

        wait (recv_count == NUM_UPDATES);
        repeat (10) @(posedge clk);

        if (total_enqueue_count != NUM_UPDATES ||
            total_issue_count != NUM_UPDATES ||
            total_completed_count != NUM_UPDATES)
            $fatal(1,
                "Count mismatch enq=%0d issue=%0d completed=%0d",
                total_enqueue_count, total_issue_count,
                total_completed_count);

        if (total_direct_write_count != 16)
            $fatal(1, "Expected 16 direct writes, got %0d",
                   total_direct_write_count);

        if (total_mixed_add_count != 8)
            $fatal(1, "Expected 8 mixed adds, got %0d",
                   total_mixed_add_count);

        // Buckets 8..15 must contain 2G.
        for (int b = 8; b < 16; b++) begin
            int lane_num;
            int local_num;
            lane_num  = b & 7;
            local_num = b >> 3;

            if (mem_tag[lane_num][local_num] !== current_gen ||
                mem_x[lane_num][local_num] !== G2_X ||
                mem_y[lane_num][local_num] !== G2_Y ||
                mem_z[lane_num][local_num] !== G2_Z)
                $fatal(1, "Bucket %0d is not 2G", b);
        end

        // Buckets 16..23 must contain G.
        for (int b = 16; b < 24; b++) begin
            int lane_num;
            int local_num;
            lane_num  = b & 7;
            local_num = b >> 3;

            if (mem_tag[lane_num][local_num] !== current_gen ||
                mem_x[lane_num][local_num] !== GX_M ||
                mem_y[lane_num][local_num] !== GY_M ||
                mem_z[lane_num][local_num] !== ONE_M)
                $fatal(1, "Bucket %0d is not G", b);
        end

        $display("====================================================");
        $display(" BUCKET UPDATE SCHEDULER 8LANE V1 PASSED");
        $display(" total_cycles            = %0d", cycle_count);
        $display(" total_enqueue_count     = %0d", total_enqueue_count);
        $display(" total_issue_count       = %0d", total_issue_count);
        $display(" total_completed_count   = %0d", total_completed_count);
        $display(" total_direct_writes     = %0d", total_direct_write_count);
        $display(" total_mixed_adds        = %0d", total_mixed_add_count);
        $display(" total_bypass_count      = %0d", total_bypass_count);
        $display(" total_fifo_full_stalls  = %0d", total_fifo_full_stall_count);
        $display("====================================================");

        $finish;
    end

    initial begin
        #10000000;
        $fatal(1, "Timeout in tb_bucket_update_scheduler_8lane_v1");
    end

endmodule