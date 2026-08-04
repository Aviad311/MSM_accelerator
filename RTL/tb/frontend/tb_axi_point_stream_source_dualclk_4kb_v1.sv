`timescale 1ns/1ps

module tb_axi_point_stream_source_dualclk_4kb_v1;

    localparam int GLOBAL_ADDR_W    = 16;
    localparam int DATA_W           = 256;
    localparam int ASYNC_FIFO_DEPTH = 64;
    localparam int MAX_BURST_BEATS  = 64;
    localparam int NUM_POINTS       = 160;
    localparam int TOTAL_BEATS      = 2 * NUM_POINTS;
    localparam int BYTES_PER_BEAT   = 64;

    // Starts 128 bytes before a 4KB boundary.
    // First burst must therefore contain exactly 2 beats.
    localparam logic [63:0] BASE_ADDR =
        64'h0000_0000_0200_0F80;

    localparam time AXI_CLK_PERIOD = 2ns;
    localparam time MSM_CLK_PERIOD = 4ns;

    logic axi_clk, axi_rst_n;
    logic msm_clk, msm_rst_n;

    logic start_axi;
    logic [63:0] base_addr;
    logic [31:0] logical_point_count;
    logic axi_busy, axi_done;

    logic [63:0] m_axi_araddr;
    logic [7:0]  m_axi_arlen;
    logic [2:0]  m_axi_arsize;
    logic [1:0]  m_axi_arburst;
    logic        m_axi_arvalid;
    logic        m_axi_arready;

    logic [511:0] m_axi_rdata;
    logic [1:0]   m_axi_rresp;
    logic         m_axi_rlast;
    logic         m_axi_rvalid;
    logic         m_axi_rready;

    logic                     point_valid;
    logic                     point_ready;
    logic [GLOBAL_ADDR_W-1:0] point_bucket_id;
    logic [DATA_W-1:0]        point_x;
    logic [DATA_W-1:0]        point_y;
    logic                     point_last;

    logic [511:0] axi_mem [0:TOTAL_BEATS-1];

    integer error_count;
    integer ar_count;
    integer r_count;
    integer accepted_count;
    integer max_consecutive_accepts;
    integer consecutive_accepts;
    integer expected_index;

    logic        slave_active_q;
    logic [63:0] slave_addr_q;
    logic [8:0]  slave_beats_left_q;

    function automatic logic [DATA_W-1:0] mk_x(input int unsigned idx);
        logic [DATA_W-1:0] v;
        begin
            v = '0;
            v[31:0] = 32'h1000_0000 + idx;
            v[255:224] = 32'hA5A5_A5A5;
            return v;
        end
    endfunction

    function automatic logic [DATA_W-1:0] mk_y(input int unsigned idx);
        logic [DATA_W-1:0] v;
        begin
            v = '0;
            v[31:0] = 32'h2000_0000 + idx;
            v[255:224] = 32'h5A5A_5A5A;
            return v;
        end
    endfunction

    function automatic logic [GLOBAL_ADDR_W-1:0] mk_bucket(
        input int unsigned idx
    );
        return 16'((idx * 31 + 7) % 65535 + 1);
    endfunction

    initial axi_clk = 1'b0;
    always #(AXI_CLK_PERIOD/2) axi_clk = ~axi_clk;

    initial msm_clk = 1'b0;
    always #(MSM_CLK_PERIOD/2) msm_clk = ~msm_clk;

    axi_point_stream_source_dualclk_v2 #(
        .GLOBAL_ADDR_W    (GLOBAL_ADDR_W),
        .DATA_W           (DATA_W),
        .ASYNC_FIFO_DEPTH (ASYNC_FIFO_DEPTH),
        .MAX_BURST_BEATS  (MAX_BURST_BEATS)
    ) dut (
        .axi_clk             (axi_clk),
        .axi_rst_n           (axi_rst_n),
        .msm_clk             (msm_clk),
        .msm_rst_n           (msm_rst_n),
        .start_axi           (start_axi),
        .base_addr           (base_addr),
        .logical_point_count (logical_point_count),
        .axi_busy            (axi_busy),
        .axi_done            (axi_done),

        .m_axi_araddr        (m_axi_araddr),
        .m_axi_arlen         (m_axi_arlen),
        .m_axi_arsize        (m_axi_arsize),
        .m_axi_arburst       (m_axi_arburst),
        .m_axi_arvalid       (m_axi_arvalid),
        .m_axi_arready       (m_axi_arready),

        .m_axi_rdata         (m_axi_rdata),
        .m_axi_rresp         (m_axi_rresp),
        .m_axi_rlast         (m_axi_rlast),
        .m_axi_rvalid        (m_axi_rvalid),
        .m_axi_rready        (m_axi_rready),

        .point_valid         (point_valid),
        .point_ready         (point_ready),
        .point_bucket_id     (point_bucket_id),
        .point_x             (point_x),
        .point_y             (point_y),
        .point_last          (point_last)
    );

    initial begin : init_memory
        integer i;
        for (i = 0; i < NUM_POINTS; i = i + 1) begin
            axi_mem[2*i] = {mk_y(i), mk_x(i)};
            axi_mem[2*i+1] = '0;
            axi_mem[2*i+1][GLOBAL_ADDR_W-1:0] = mk_bucket(i);
            axi_mem[2*i+1][GLOBAL_ADDR_W] = (i == NUM_POINTS-1);
        end
    end

    assign m_axi_arready = !slave_active_q;

    always @(posedge axi_clk or negedge axi_rst_n) begin : axi_slave
        integer mem_index;
        integer burst_bytes;
        logic [12:0] low_sum;

        if (!axi_rst_n) begin
            slave_active_q     <= 1'b0;
            slave_addr_q       <= '0;
            slave_beats_left_q <= '0;
            m_axi_rdata        <= '0;
            m_axi_rresp        <= 2'b00;
            m_axi_rlast        <= 1'b0;
            m_axi_rvalid       <= 1'b0;
            ar_count           <= 0;
            r_count            <= 0;
        end else begin
            if (m_axi_arvalid && m_axi_arready) begin
                burst_bytes = ({1'b0, m_axi_arlen} + 9'd1) * BYTES_PER_BEAT;
                low_sum = {1'b0, m_axi_araddr[11:0]} + burst_bytes;

                if (low_sum > 13'd4096) begin
                    error_count++;
                    $error("[4KB] burst crosses boundary addr=%h len=%0d",
                           m_axi_araddr, m_axi_arlen + 1);
                end

                if (m_axi_arsize !== 3'b110) begin
                    error_count++;
                    $error("[4KB] ARSIZE mismatch");
                end

                if (m_axi_arburst !== 2'b01) begin
                    error_count++;
                    $error("[4KB] ARBURST mismatch");
                end

                if (m_axi_araddr[5:0] != 0) begin
                    error_count++;
                    $error("[4KB] ARADDR not 64-byte aligned");
                end

                if (ar_count == 0 && (m_axi_arlen != 8'd1)) begin
                    error_count++;
                    $error("[4KB] first burst should be 2 beats, got %0d",
                           m_axi_arlen + 1);
                end

                mem_index = (m_axi_araddr - BASE_ADDR) >> 6;
                m_axi_rdata <= axi_mem[mem_index];
                m_axi_rresp <= 2'b00;
                m_axi_rlast <= (m_axi_arlen == 0);
                m_axi_rvalid <= 1'b1;

                slave_active_q <= 1'b1;
                slave_addr_q <= m_axi_araddr;
                slave_beats_left_q <= {1'b0, m_axi_arlen} + 9'd1;
                ar_count <= ar_count + 1;
            end

            if (m_axi_rvalid && m_axi_rready) begin
                r_count <= r_count + 1;

                if (slave_beats_left_q == 1) begin
                    slave_active_q <= 1'b0;
                    slave_beats_left_q <= '0;
                    m_axi_rvalid <= 1'b0;
                    m_axi_rlast <= 1'b0;
                end else begin
                    slave_addr_q <= slave_addr_q + BYTES_PER_BEAT;
                    slave_beats_left_q <= slave_beats_left_q - 1'b1;

                    mem_index =
                        ((slave_addr_q + BYTES_PER_BEAT) - BASE_ADDR) >> 6;
                    m_axi_rdata <= axi_mem[mem_index];
                    m_axi_rlast <= (slave_beats_left_q == 2);
                    m_axi_rvalid <= 1'b1;
                end
            end
        end
    end

    assign point_ready = 1'b1;

    always @(posedge msm_clk or negedge msm_rst_n) begin
        if (!msm_rst_n) begin
            accepted_count <= 0;
            expected_index <= 0;
            consecutive_accepts <= 0;
            max_consecutive_accepts <= 0;
        end else begin
            if (point_valid && point_ready) begin
                if (point_x !== mk_x(expected_index)) begin
                    error_count++;
                    $error("[4KB] X mismatch idx=%0d", expected_index);
                end

                if (point_y !== mk_y(expected_index)) begin
                    error_count++;
                    $error("[4KB] Y mismatch idx=%0d", expected_index);
                end

                if (point_bucket_id !== mk_bucket(expected_index)) begin
                    error_count++;
                    $error("[4KB] bucket mismatch idx=%0d", expected_index);
                end

                if (point_last !== (expected_index == NUM_POINTS-1)) begin
                    error_count++;
                    $error("[4KB] last mismatch idx=%0d", expected_index);
                end

                accepted_count <= accepted_count + 1;
                expected_index <= expected_index + 1;
                consecutive_accepts <= consecutive_accepts + 1;

                if (consecutive_accepts + 1 > max_consecutive_accepts)
                    max_consecutive_accepts <= consecutive_accepts + 1;
            end else begin
                consecutive_accepts <= 0;
            end
        end
    end

    initial begin : test_sequence
        error_count = 0;
        axi_rst_n = 1'b0;
        msm_rst_n = 1'b0;
        start_axi = 1'b0;
        base_addr = BASE_ADDR;
        logical_point_count = NUM_POINTS;

        repeat (8) @(posedge axi_clk);
        axi_rst_n = 1'b1;

        repeat (4) @(posedge msm_clk);
        msm_rst_n = 1'b1;

        repeat (4) @(posedge axi_clk);
        @(negedge axi_clk);
        start_axi = 1'b1;
        @(negedge axi_clk);
        start_axi = 1'b0;

        wait (accepted_count == NUM_POINTS);
        repeat (8) @(posedge msm_clk);

        if (r_count != TOTAL_BEATS) begin
            error_count++;
            $error("[4KB] expected %0d beats, got %0d",
                   TOTAL_BEATS, r_count);
        end

        // 320 beats from address ...0F80:
        // 2 beats to boundary, then 4 full 64-beat bursts, then 62 beats.
        if (ar_count != 6) begin
            error_count++;
            $error("[4KB] expected 6 bursts, got %0d", ar_count);
        end

        if (max_consecutive_accepts < 64) begin
            error_count++;
            $error("[4KB] II=1 run too short: expected at least 64, got %0d",
                   max_consecutive_accepts);
        end

        if (error_count == 0) begin
            $display("");
            $display("============================================================");
            $display("[TB_4KB] AXI 4KB BURST SPLITTER PASSED");
            $display("[TB_4KB] base address               = %h", BASE_ADDR);
            $display("[TB_4KB] logical points             = %0d", NUM_POINTS);
            $display("[TB_4KB] total beats                = %0d", r_count);
            $display("[TB_4KB] bursts                     = %0d", ar_count);
            $display("[TB_4KB] max consecutive accepts    = %0d",
                     max_consecutive_accepts);
            $display("[TB_4KB] verified no 4KB crossing, CDC, ordering, II=1");
            $display("============================================================");
        end else begin
            $fatal(1, "[TB_4KB] FAILED with %0d errors", error_count);
        end

        $finish;
    end

    initial begin : watchdog
        #1ms;
        $fatal(1, "[TB_4KB] WATCHDOG TIMEOUT");
    end

endmodule