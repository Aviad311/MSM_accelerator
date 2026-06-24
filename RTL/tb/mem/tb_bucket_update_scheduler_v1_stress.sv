`timescale 1ns/1ps

module tb_bucket_update_scheduler_v1_stress;

    localparam int ADDR_W = 8;
    localparam int DATA_W = 256;
    localparam int DEPTH = (1 << ADDR_W);
    localparam int GEN_W = 16;
    localparam int FIFO_DEPTH = 32;
    localparam int SLOT_COUNT = 24;
    localparam int MIX_CTX_COUNT = 48;
    localparam int SRAM_RD_LATENCY = 3;
    localparam int MAX_UPDATES = 256;

    localparam logic [255:0] ZERO =
        256'h0000000000000000000000000000000000000000000000000000000000000000;
    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;
    localparam logic [255:0] GX_M =
        256'h9981E643E9089F48979F48C033FD129C231E295329BC66DBD7362E5A487E2097;
    localparam logic [255:0] GY_M =
        256'hCF3F851FD4A582D670B6B59AAC19C1368DFC5D5D1F1DC64DB15EA6D2D3DBABE2;

    typedef enum int {CASE_UNIFORM, CASE_HOT1, CASE_HOT4, CASE_HOT8} case_kind_t;

    logic clk, rst_n;
    logic in_valid, in_ready;
    logic [GEN_W-1:0] current_gen;
    logic [ADDR_W-1:0] in_bucket_id;
    logic [DATA_W-1:0] in_point_x, in_point_y;

    logic out_valid, out_ready;
    logic [ADDR_W-1:0] out_bucket_id;
    logic out_skipped, out_direct_write, out_mixed_add;
    logic [DATA_W-1:0] out_x, out_y, out_z;

    logic mem_valid, mem_write_en;
    logic [ADDR_W-1:0] mem_addr;
    logic [DATA_W-1:0] mem_wdata_x, mem_wdata_y, mem_wdata_z;
    logic mem_tag_write_en;
    logic [GEN_W-1:0] mem_tag_wdata;
    logic mem_ready, mem_rvalid;
    logic [DATA_W-1:0] mem_rdata_x, mem_rdata_y, mem_rdata_z;
    logic [GEN_W-1:0] mem_tag_rdata;

    logic [$clog2(FIFO_DEPTH+1)-1:0] fifo_occupancy;
    logic [63:0] enqueue_count, issue_count, bypass_count, fifo_full_stall_count;
    logic issue_pulse;
    logic [ADDR_W-1:0] issue_bucket_id;

    logic [$clog2(SLOT_COUNT+1)-1:0] active_slots;
    logic [63:0] accepted_count, completed_count;
    logic [63:0] downstream_same_bucket_stall_count;
    logic [63:0] direct_write_count, mixed_add_count;

    logic [DATA_W-1:0] mem_x [0:DEPTH-1];
    logic [DATA_W-1:0] mem_y [0:DEPTH-1];
    logic [DATA_W-1:0] mem_z [0:DEPTH-1];
    logic [GEN_W-1:0] mem_tag [0:DEPTH-1];

    logic [SRAM_RD_LATENCY-1:0] rd_valid_pipe;
    logic [ADDR_W-1:0] rd_addr_pipe [0:SRAM_RD_LATENCY-1];
    logic [ADDR_W-1:0] vectors [0:MAX_UPDATES-1];

    int cycle_count, case_start_cycle, send_count, recv_count;
    int input_stall_cycles, max_fifo_occupancy;

    assign mem_ready = 1'b1;
    assign mem_rvalid = rd_valid_pipe[SRAM_RD_LATENCY-1];
    assign mem_rdata_x = mem_x[rd_addr_pipe[SRAM_RD_LATENCY-1]];
    assign mem_rdata_y = mem_y[rd_addr_pipe[SRAM_RD_LATENCY-1]];
    assign mem_rdata_z = mem_z[rd_addr_pipe[SRAM_RD_LATENCY-1]];
    assign mem_tag_rdata = mem_tag[rd_addr_pipe[SRAM_RD_LATENCY-1]];

    bucket_update_scheduler_v1 #(
        .ADDR_W(ADDR_W), .DATA_W(DATA_W), .DEPTH(DEPTH), .GEN_W(GEN_W),
        .FIFO_DEPTH(FIFO_DEPTH), .SLOT_COUNT(SLOT_COUNT),
        .MIX_CTX_COUNT(MIX_CTX_COUNT), .MUL_LATENCY(16),
        .SKIP_ZERO_BUCKET(1'b1)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid), .in_ready(in_ready), .current_gen(current_gen),
        .in_bucket_id(in_bucket_id), .in_point_x(in_point_x), .in_point_y(in_point_y),
        .out_valid(out_valid), .out_ready(out_ready), .out_bucket_id(out_bucket_id),
        .out_skipped(out_skipped), .out_direct_write(out_direct_write),
        .out_mixed_add(out_mixed_add), .out_x(out_x), .out_y(out_y), .out_z(out_z),
        .mem_valid(mem_valid), .mem_write_en(mem_write_en), .mem_addr(mem_addr),
        .mem_wdata_x(mem_wdata_x), .mem_wdata_y(mem_wdata_y), .mem_wdata_z(mem_wdata_z),
        .mem_tag_write_en(mem_tag_write_en), .mem_tag_wdata(mem_tag_wdata),
        .mem_ready(mem_ready), .mem_rvalid(mem_rvalid),
        .mem_rdata_x(mem_rdata_x), .mem_rdata_y(mem_rdata_y),
        .mem_rdata_z(mem_rdata_z), .mem_tag_rdata(mem_tag_rdata),
        .fifo_occupancy(fifo_occupancy), .enqueue_count(enqueue_count),
        .issue_count(issue_count), .bypass_count(bypass_count),
        .fifo_full_stall_count(fifo_full_stall_count),
        .issue_pulse(issue_pulse), .issue_bucket_id(issue_bucket_id),
        .active_slots(active_slots), .accepted_count(accepted_count),
        .completed_count(completed_count),
        .downstream_same_bucket_stall_count(downstream_same_bucket_stall_count),
        .direct_write_count(direct_write_count), .mixed_add_count(mixed_add_count)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    integer i;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_valid_pipe <= '0;
            for (i = 0; i < SRAM_RD_LATENCY; i = i + 1) rd_addr_pipe[i] <= '0;
            for (i = 0; i < DEPTH; i = i + 1) begin
                mem_x[i] <= ZERO; mem_y[i] <= ONE_M; mem_z[i] <= ZERO; mem_tag[i] <= '0;
            end
        end else begin
            for (i = SRAM_RD_LATENCY-1; i > 0; i = i - 1) begin
                rd_valid_pipe[i] <= rd_valid_pipe[i-1];
                rd_addr_pipe[i] <= rd_addr_pipe[i-1];
            end
            rd_valid_pipe[0] <= mem_valid && !mem_write_en && mem_ready;
            if (mem_valid && !mem_write_en && mem_ready) rd_addr_pipe[0] <= mem_addr;
            if (mem_valid && mem_write_en && mem_ready) begin
                mem_x[mem_addr] <= mem_wdata_x;
                mem_y[mem_addr] <= mem_wdata_y;
                mem_z[mem_addr] <= mem_wdata_z;
                if (mem_tag_write_en) mem_tag[mem_addr] <= mem_tag_wdata;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 0; recv_count <= 0;
            input_stall_cycles <= 0; max_fifo_occupancy <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (in_valid && !in_ready) input_stall_cycles <= input_stall_cycles + 1;
            if (fifo_occupancy > max_fifo_occupancy) max_fifo_occupancy <= fifo_occupancy;
            if (out_valid && out_ready) recv_count <= recv_count + 1;
        end
    end

    task automatic build_case(input case_kind_t kind, input int n, output int unique_buckets);
        int j;
        begin
            case (kind)
                CASE_UNIFORM: begin
                    for (j = 0; j < n; j = j + 1) vectors[j] = 1 + (j % 255);
                    unique_buckets = (n < 255) ? n : 255;
                end
                CASE_HOT1: begin
                    for (j = 0; j < n; j = j + 1) vectors[j] = 8'd1;
                    unique_buckets = 1;
                end
                CASE_HOT4: begin
                    for (j = 0; j < n; j = j + 1) vectors[j] = 1 + (j % 4);
                    unique_buckets = (n < 4) ? n : 4;
                end
                default: begin
                    for (j = 0; j < n; j = j + 1) vectors[j] = 1 + (j % 8);
                    unique_buckets = (n < 8) ? n : 8;
                end
            endcase
        end
    endtask

    task automatic run_case(
        input string case_name,
        input case_kind_t kind,
        input int n,
        input logic [GEN_W-1:0] generation
    );
        int unique_buckets, expected_direct, expected_mixed, total_case_cycles;
        begin
            build_case(kind, n, unique_buckets);
            expected_direct = unique_buckets;
            expected_mixed = n - unique_buckets;

            rst_n = 1'b0; in_valid = 1'b0; in_bucket_id = '0;
            in_point_x = GX_M; in_point_y = GY_M;
            current_gen = generation; out_ready = 1'b1; send_count = 0;

            repeat (6) @(posedge clk);
            rst_n = 1'b1;
            repeat (3) @(posedge clk);
            case_start_cycle = cycle_count;

            $display("====================================================");
            $display("[STRESS_CASE_START] name=%s updates=%0d unique=%0d",
                     case_name, n, unique_buckets);

            while (send_count < n) begin
                @(negedge clk);
                in_valid = 1'b1;
                in_bucket_id = vectors[send_count];
                @(posedge clk);
                if (in_ready) send_count = send_count + 1;
            end

            @(negedge clk);
            in_valid = 1'b0;
            in_bucket_id = '0;

            wait (recv_count == n);
            repeat (8) @(posedge clk);

            total_case_cycles = cycle_count - case_start_cycle;

            if (enqueue_count != n || issue_count != n ||
                accepted_count != n || completed_count != n)
                $fatal(1, "%s count mismatch enq=%0d issue=%0d accepted=%0d completed=%0d",
                       case_name, enqueue_count, issue_count, accepted_count, completed_count);

            if (direct_write_count != expected_direct)
                $fatal(1, "%s direct mismatch expected=%0d got=%0d",
                       case_name, expected_direct, direct_write_count);

            if (mixed_add_count != expected_mixed)
                $fatal(1, "%s mixed mismatch expected=%0d got=%0d",
                       case_name, expected_mixed, mixed_add_count);

            if (downstream_same_bucket_stall_count != 0)
                $fatal(1, "%s downstream same-bucket stalls=%0d",
                       case_name, downstream_same_bucket_stall_count);

            $display("[STRESS_CASE_PASS] name=%s", case_name);
            $display("  total_cycles           = %0d", total_case_cycles);
            $display("  cycles_per_update      = %0f", (1.0 * total_case_cycles) / n);
            $display("  bypass_count           = %0d", bypass_count);
            $display("  fifo_full_stall_count  = %0d", fifo_full_stall_count);
            $display("  input_stall_cycles     = %0d", input_stall_cycles);
            $display("  max_fifo_occupancy     = %0d", max_fifo_occupancy);
            $display("  direct_write_count     = %0d", direct_write_count);
            $display("  mixed_add_count        = %0d", mixed_add_count);
            $display("  downstream_same_stalls = %0d",
                     downstream_same_bucket_stall_count);
        end
    endtask

    initial begin
        rst_n = 1'b0; in_valid = 1'b0; current_gen = '0;
        in_bucket_id = '0; in_point_x = GX_M; in_point_y = GY_M; out_ready = 1'b1;

        run_case("uniform_256", CASE_UNIFORM, 256, 16'h0101);
        run_case("hot8_256", CASE_HOT8, 256, 16'h0102);
        run_case("hot4_256", CASE_HOT4, 256, 16'h0103);
        run_case("hot1_64", CASE_HOT1, 64, 16'h0104);

        $display("====================================================");
        $display(" BUCKET UPDATE SCHEDULER V1 STRESS PASSED");
        $display("====================================================");
        $finish;
    end

    initial begin
        #200000000;
        $fatal(1, "Timeout in tb_bucket_update_scheduler_v1_stress");
    end

endmodule