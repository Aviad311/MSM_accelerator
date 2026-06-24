`timescale 1ns / 1ps

module axi_point_fetch_fifo #(
    parameter int FIFO_DEPTH      = 64,  // Must be a power of 2 (e.g., 16, 32, 64, 128)
    parameter int MAX_BURST_BEATS = 16   // Maximum AXI4 burst length (1 to 256)
)(
    input  logic         clk,
    input  logic         rst_n,

    // --- Control & Status Interface ---
    input  logic         start,
    input  logic [63:0]  base_addr,
    input  logic [31:0]  total_points,
    output logic         busy,
    output logic         done,

    // --- AXI4 Read Address Channel (AR) ---
    output logic [63:0]  m_axi_araddr,
    output logic [7:0]   m_axi_arlen,
    output logic [2:0]   m_axi_arsize,
    output logic [1:0]   m_axi_arburst,
    output logic         m_axi_arvalid,
    input  logic         m_axi_arready,

    // --- AXI4 Read Data Channel (R) ---
    input  logic [511:0] m_axi_rdata,
    input  logic [1:0]   m_axi_rresp,
    input  logic         m_axi_rlast,
    input  logic         m_axi_rvalid,
    output logic         m_axi_rready,

    // --- Downstream FIFO Interface ---
    input  logic         pop,
    output logic [511:0] dout,
    output logic         empty,
    output logic         full,
    output logic         almost_full
);

    // =========================================================================
    // Parameters & Typdefs
    // =========================================================================
    localparam int ADDR_W = $clog2(FIFO_DEPTH);
    typedef logic [ADDR_W:0] fifo_cnt_t; // 1 extra bit for full/empty wrap detection

    localparam fifo_cnt_t MAX_FIFO_CNT = FIFO_DEPTH[ADDR_W:0];

    // =========================================================================
    // Internal Signals
    // =========================================================================
    // FIFO Pointers & Counters
    fifo_cnt_t    wr_ptr, rd_ptr;
    fifo_cnt_t    fifo_count;
    fifo_cnt_t    in_flight_beats;
    fifo_cnt_t    committed_space;
    fifo_cnt_t    safe_space;

    // AXI tracking
    logic [63:0]  cur_araddr;
    logic [31:0]  points_left_to_req;
    logic [31:0]  points_left_to_rx;
    logic [8:0]   ar_beats_to_send; // 9 bits to safely hold up to 256

    logic         fifo_push;
    logic         ar_fire;
    logic         r_fire;

    // FSM
    typedef enum logic {
        IDLE   = 1'b0,
        ACTIVE = 1'b1
    } state_e;

    state_e state;

    // =========================================================================
    // Static AXI Assignments
    // =========================================================================
    assign m_axi_arsize  = 3'b110; // 64 Bytes = 512 bits per beat
    assign m_axi_arburst = 2'b01;  // INCR burst type
    assign m_axi_rready  = ~full;  // Hard protection (though FSM guarantees no overflow)

    assign ar_fire = m_axi_arvalid & m_axi_arready;
    assign r_fire  = m_axi_rvalid  & m_axi_rready;
    assign fifo_push = r_fire;

    // =========================================================================
    // FIFO Status Math
    // =========================================================================
    assign fifo_count      = wr_ptr - rd_ptr;
    assign full            = (fifo_count == MAX_FIFO_CNT);
    assign empty           = (fifo_count == '0);
    assign almost_full     = (fifo_count >= (MAX_FIFO_CNT - 1'b1));

    assign committed_space = fifo_count + in_flight_beats;
    assign safe_space      = MAX_FIFO_CNT - committed_space;

    // =========================================================================
    // AXI Master Burst Decision Logic (Combinational)
    // =========================================================================
    logic take_max_burst;
    logic take_tail_burst;
    logic [7:0] chosen_burst_len;

    // Can we fit a full max-burst inside the available safe room?
    assign take_max_burst  = (points_left_to_req >= MAX_BURST_BEATS) && 
                             (safe_space >= MAX_BURST_BEATS[ADDR_W:0]);

    // Can we fit the tail (the remainder) inside the available safe room?
    assign take_tail_burst = (points_left_to_req > 0) && 
                             (points_left_to_req < MAX_BURST_BEATS) && 
                             (safe_space >= points_left_to_req[ADDR_W:0]);

    assign chosen_burst_len = take_max_burst ? MAX_BURST_BEATS[7:0] : points_left_to_req[7:0];

    // =========================================================================
    // FSM & AXI Address (AR) Channel Issuing
    // =========================================================================
    assign ar_beats_to_send = 9'(m_axi_arlen) + 9'd1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state              <= IDLE;
            m_axi_arvalid      <= 1'b0;
            m_axi_arlen        <= '0;
            m_axi_araddr       <= '0;
            cur_araddr         <= '0;
            points_left_to_req <= '0;
            busy               <= 1'b0;
            done               <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    if (start && total_points > 0) begin
                        state              <= ACTIVE;
                        cur_araddr         <= base_addr;
                        points_left_to_req <= total_points;
                        busy               <= 1'b1;
                        done               <= 1'b0;
                    end
                end

                ACTIVE: begin
                    // 1. Issue new AR transaction if line is free and we have safe space
                    if (!m_axi_arvalid) begin
                        if (take_max_burst || take_tail_burst) begin
                            m_axi_arvalid <= 1'b1;
                            m_axi_arlen   <= chosen_burst_len - 8'd1; // AXI spec: Length is N-1
                            m_axi_araddr  <= cur_araddr;
                        end
                    end else if (m_axi_arready) begin
                        m_axi_arvalid      <= 1'b0;
                        // Advance address by (Beats * 64 Bytes)
                        cur_araddr         <= cur_araddr + ({55'd0, ar_beats_to_send} << 6); 
                        points_left_to_req <= points_left_to_req - {23'd0, ar_beats_to_send};
                    end

                    // 2. Check complete transfer termination
                    if (points_left_to_rx == 0) begin
                        state <= IDLE;
                        busy  <= 1'b0;
                        done  <= 1'b1;
                    end
                end
            endcase
        end
    end

    // =========================================================================
    // Tracking In-Flight Beats & Data Rx
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_flight_beats   <= '0;
            points_left_to_rx <= '0;
        end else if (state == IDLE && start) begin
            in_flight_beats   <= '0;
            points_left_to_rx <= total_points;
        end else begin
            // In-flight tracker (Handles simultaneous AR fire and R fire safely)
            case ({ar_fire, r_fire})
                2'b10: in_flight_beats <= in_flight_beats + ar_beats_to_send[ADDR_W:0];
                2'b01: in_flight_beats <= in_flight_beats - 1'b1;
                2'b11: in_flight_beats <= in_flight_beats + ar_beats_to_send[ADDR_W:0] - 1'b1;
                default: ; // hold stable
            endcase

            if (r_fire) begin
                points_left_to_rx <= points_left_to_rx - 1'b1;
            end
        end
    end

    // =========================================================================
    // FIFO Storage & Pointers
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= '0;
            rd_ptr <= '0;
        end else if (state == IDLE && start) begin
            wr_ptr <= '0; // Auto-flush FIFO on new start
            rd_ptr <= '0;
        end else begin
            if (fifo_push)      wr_ptr <= wr_ptr + 1'b1;
            if (pop && !empty)  rd_ptr <= rd_ptr + 1'b1;
        end
    end

    // --- RAM Array ---
    logic [511:0] sram_array [0:FIFO_DEPTH-1];

    always_ff @(posedge clk) begin
        if (fifo_push) begin
            sram_array[wr_ptr[ADDR_W-1:0]] <= m_axi_rdata;
        end
    end

    // ASIC CRITICAL SYNTHESIS NOTE: 
    // This continuous assignment infers a Lookahead (FWFT) FIFO via Flip-Flops.
    // When replacing 'sram_array' with a real Foundry Dual-Port SRAM macro, 
    // the read will become synchronous (1 cycle latency). You will need to attach 
    // a 1-stage Skid Buffer here to preserve the zero-latency 'dout' behavior.
    assign dout = sram_array[rd_ptr[ADDR_W-1:0]];

endmodule