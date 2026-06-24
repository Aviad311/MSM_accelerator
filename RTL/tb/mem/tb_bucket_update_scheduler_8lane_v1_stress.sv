`timescale 1ns/1ps

module tb_bucket_update_scheduler_8lane_v1_stress;

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
    localparam int MAX_UPDATES     = 256;

    localparam logic [255:0] ZERO =
        256'h0000000000000000000000000000000000000000000000000000000000000000;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    localparam logic [255:0] GX_M =
        256'h9981E643E9089F48979F48C033FD129C231E295329BC66DBD7362E5A487E2097;

    localparam logic [255:0] GY_M =
        256'hCF3F851FD4A582D670B6B59AAC19C1368DFC5D5D1F1DC64DB15EA6D2D3DBABE2;

    typedef enum int {
        CASE_UNIFORM = 0,
        CASE_HOT8    = 1,
        CASE_HOT4    = 2,
        CASE_HOT1    = 3
    } case_kind_t;

    logic clk;
    logic rst_n;

    logic in_valid;
    logic in_ready;
    logic [GEN_W-1:0] current_gen;
    logic [GLOBAL_ADDR_W-1:0] in_bucket_id;
    logic [DATA_W-1:0] in_point_x;
    logic [DATA_W-1:0] in_point_y;

    logic out_valid;
    logic out_ready;
    logic [GLOBAL_ADDR_W-1:0] out_bucket_id;
    logic out_skipped;
    logic out_direct_write;
    logic out_mixed_add;
    logic [DATA_W-1:0] out_x;
    logic [DATA_W-1:0] out_y;
    logic [DATA_W-1:0] out_z;

    logic [LANES-1:0] mem_valid;
    logic [LANES-1:0] mem_write_en;
    logic [LANES-1:0][LOCAL_ADDR_W-1:0] mem_addr;
    logic [LANES-1:0][DATA_W-1:0] mem_wdata_x;
    logic [LANES-1:0][DATA_W-1:0] mem_wdata_y;
    logic [LANES-1:0][DATA_W-1:0] mem_wdata_z;
    logic [LANES-1:0] mem_tag_write_en;
    logic [LANES-1:0][GEN_W-1:0] mem_tag_wdata;

    logic [LANES-1:0] mem_ready;
    logic [LANES-1:0] mem_rvalid;
    logic [LANES-1:0][DATA_W-1:0] mem_rdata_x;
    logic [LANES-1:0][DATA_W-1:0] mem_rdata_y;
    logic [LANES-1:0][DATA_W-1:0] mem_rdata_z;
    logic [LANES-1:0][GEN_W-1:0] mem_tag_rdata;

    logic [63:0] total_enqueue_count;
    logic [63:0] total_issue_count;
    logic [63:0] total_completed_count;
    logic [63:0] total_bypass_count;
    logic [63:0] total_fifo_full_stall_count;
    logic [63:0] total_direct_write_count;
    logic [63:0] total_mixed_add_count;

    logic [LANES-1:0][$clog2(FIFO_DEPTH+1)-1:0] lane_fifo_occupancy;
    logic [LANES-1:0][$clog2(SLOT_COUNT+1)-1:0] lane_active_slots;

    logic [DATA_W-1:0] mem_x [0:LANES-1][0:LOCAL_DEPTH-1];
    logic [DATA_W-1:0] mem_y [0:LANES-1][0:LOCAL_DEPTH-1];
    logic [DATA_W-1:0] mem_z [0:LANES-1][0:LOCAL_DEPTH-1];
    logic [GEN_W-1:0] mem_tag [0:LANES-1][0:LOCAL_DEPTH-1];

    logic [LANES-1:0][RD_LATENCY-1:0] rd_valid_pipe;
    logic [LANES-1:0][RD_LATENCY-1:0][LOCAL_ADDR_W-1:0] rd_addr_pipe;

    logic [GLOBAL_ADDR_W-1:0] vectors [0:MAX_UPDATES-1];

    int cycle_count;
    int case_start_cycle;
    int send_count;
    int recv_count;
    int input_stall_cycles;
    int max_lane_fifo [0:LANES-1];

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

            for (lane_i = 0; lane_i < LANES; lane_i = lane_i + 1) begin
                for (addr_i = 0; addr_i < LOCAL_DEPTH; addr_i = addr_i + 1) begin
                    mem_x[lane_i][addr_i]   <= ZERO;
                    mem_y[lane_i][addr_i]   <= ONE_M;
                    mem_z[lane_i][addr_i]   <= ZERO;
                    mem_tag[lane_i][addr_i] <= '0;
                end
            end
        end else begin
            for (lane_i = 0; lane_i < LANES; lane_i = lane_i + 1) begin
                for (pipe_i = RD_LATENCY-1; pipe_i > 0; pipe_i = pipe_i - 1) begin
                    rd_valid_pipe[lane_i][pipe_i] <=
                        rd_valid_pipe[lane_i][pipe_i-1];

                    rd_addr_pipe[lane_i][pipe_i] <=
                        rd_addr_pipe[lane_i][pipe_i-1];
                end

                rd_valid_pipe[lane_i][0] <=
                    mem_valid[lane_i] &&
                    !mem_write_en[lane_i] &&
                    mem_ready[lane_i];

                if (mem_valid[lane_i] &&
                    !mem_write_en[lane_i] &&
                    mem_ready[lane_i]) begin

                    rd_addr_pipe[lane_i][0] <= mem_addr[lane_i];
                end

                if (mem_valid[lane_i] &&
                    mem_write_en[lane_i] &&
                    mem_ready[lane_i]) begin

                    mem_x[lane_i][mem_addr[lane_i]] <= mem_wdata_x[lane_i];
                    mem_y[lane_i][mem_addr[lane_i]] <= mem_wdata_y[lane_i];
                    mem_z[lane_i][mem_addr[lane_i]] <= mem_wdata_z[lane_i];

                    if (mem_tag_write_en[lane_i]) begin
                        mem_tag[lane_i][mem_addr[lane_i]] <=
                            mem_tag_wdata[lane_i];
                    end
                end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count        <= 0;
            recv_count         <= 0;
            input_stall_cycles <= 0;

            for (int m = 0; m < LANES; m = m + 1)
                max_lane_fifo[m] <= 0;
        end else begin
            cycle_count <= cycle_count + 1;

            if (in_valid && !in_ready)
                input_stall_cycles <= input_stall_cycles + 1;

            if (out_valid && out_ready)
                recv_count <= recv_count + 1;

            for (int m = 0; m < LANES; m = m + 1) begin
                if (lane_fifo_occupancy[m] > max_lane_fifo[m])
                    max_lane_fifo[m] <= lane_fifo_occupancy[m];
            end
        end
    end

    task automatic build_case(
        input case_kind_t kind,
        input int         num_updates,
        output int        unique_buckets
    );
        int j;
        begin
            case (kind)
                CASE_UNIFORM: begin
                    for (j = 0; j < num_updates; j = j + 1)
                        vectors[j] = 1 + (j % 255);

                    unique_buckets =
                        (num_updates < 255) ? num_updates : 255;
                end

                CASE_HOT8: begin
                    // Buckets 1..8 map one hot bucket to each lane.
                    for (j = 0; j < num_updates; j = j + 1)
                        vectors[j] = 1 + (j % 8);

                    unique_buckets =
                        (num_updates < 8) ? num_updates : 8;
                end

                CASE_HOT4: begin
                    // Buckets 1..4 occupy four different lanes.
                    for (j = 0; j < num_updates; j = j + 1)
                        vectors[j] = 1 + (j % 4);

                    unique_buckets =
                        (num_updates < 4) ? num_updates : 4;
                end

                default: begin
                    for (j = 0; j < num_updates; j = j + 1)
                        vectors[j] = 8'd1;

                    unique_buckets = 1;
                end
            endcase
        end
    endtask

    task automatic run_case(
        input string      case_name,
        input case_kind_t kind,
        input int         num_updates,
        input logic [GEN_W-1:0] generation
    );
        int unique_buckets;
        int expected_direct;
        int expected_mixed;
        int total_case_cycles;
        begin
            build_case(kind, num_updates, unique_buckets);

            expected_direct = unique_buckets;
            expected_mixed  = num_updates - unique_buckets;

            rst_n        = 1'b0;
            in_valid     = 1'b0;
            current_gen  = generation;
            in_bucket_id = '0;
            in_point_x   = GX_M;
            in_point_y   = GY_M;
            out_ready    = 1'b1;
            send_count   = 0;

            repeat (6) @(posedge clk);
            rst_n = 1'b1;
            repeat (3) @(posedge clk);

            case_start_cycle = cycle_count;

            $display("====================================================");
            $display("[8L_STRESS_START] name=%s updates=%0d unique=%0d",
                     case_name, num_updates, unique_buckets);

            while (send_count < num_updates) begin
                @(negedge clk);

                in_valid     = 1'b1;
                in_bucket_id = vectors[send_count];
                in_point_x   = GX_M;
                in_point_y   = GY_M;

                @(posedge clk);

                if (in_ready)
                    send_count = send_count + 1;
            end

            @(negedge clk);
            in_valid     = 1'b0;
            in_bucket_id = '0;

            wait (recv_count == num_updates);
            repeat (10) @(posedge clk);

            total_case_cycles = cycle_count - case_start_cycle;

            if (total_enqueue_count != num_updates ||
                total_issue_count != num_updates ||
                total_completed_count != num_updates) begin

                $fatal(1,
                    "%s count mismatch enq=%0d issue=%0d completed=%0d",
                    case_name,
                    total_enqueue_count,
                    total_issue_count,
                    total_completed_count);
            end

            if (total_direct_write_count != expected_direct) begin
                $fatal(1,
                    "%s direct mismatch expected=%0d got=%0d",
                    case_name,
                    expected_direct,
                    total_direct_write_count);
            end

            if (total_mixed_add_count != expected_mixed) begin
                $fatal(1,
                    "%s mixed mismatch expected=%0d got=%0d",
                    case_name,
                    expected_mixed,
                    total_mixed_add_count);
            end

            $display("[8L_STRESS_PASS] name=%s", case_name);
            $display("  total_cycles           = %0d", total_case_cycles);
            $display("  cycles_per_update      = %0f",
                     (1.0 * total_case_cycles) / num_updates);
            $display("  total_bypass_count     = %0d",
                     total_bypass_count);
            $display("  total_fifo_full_stalls = %0d",
                     total_fifo_full_stall_count);
            $display("  input_stall_cycles     = %0d",
                     input_stall_cycles);
            $display("  direct_write_count     = %0d",
                     total_direct_write_count);
            $display("  mixed_add_count        = %0d",
                     total_mixed_add_count);

            for (int m = 0; m < LANES; m = m + 1) begin
                $display("  lane%0d_max_fifo        = %0d",
                         m, max_lane_fifo[m]);
            end
        end
    endtask

    initial begin
        rst_n        = 1'b0;
        in_valid     = 1'b0;
        current_gen  = '0;
        in_bucket_id = '0;
        in_point_x   = GX_M;
        in_point_y   = GY_M;
        out_ready    = 1'b1;

        run_case("uniform_256", CASE_UNIFORM, 256, 16'h0201);
        run_case("hot8_256",    CASE_HOT8,    256, 16'h0202);
        run_case("hot4_256",    CASE_HOT4,    256, 16'h0203);
        run_case("hot1_64",     CASE_HOT1,     64, 16'h0204);

        $display("====================================================");
        $display(" BUCKET UPDATE SCHEDULER 8LANE V1 STRESS PASSED");
        $display("====================================================");

        $finish;
    end

    initial begin
        #300000000;
        $fatal(1,
            "Timeout in tb_bucket_update_scheduler_8lane_v1_stress");
    end

endmodule


