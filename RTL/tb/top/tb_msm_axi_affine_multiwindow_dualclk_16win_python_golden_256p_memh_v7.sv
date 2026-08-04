

`timescale 1ns/1ps

module tb_msm_axi_affine_multiwindow_dualclk_16win_python_golden_256p_memh_v7;

    // ------------------------------------------------------------------------
    // Parameters & Golden Constants
    // ------------------------------------------------------------------------
    localparam int DATA_W = 256;
    localparam int AXI_DATA_W = 512;
    localparam int ADDR_W = 16;
    
    // Base address for the 8192 AXI beats (derived from the logs)
    localparam logic [63:0] AXI_BASE_ADDR = 64'h0000000002078000;
    
    // Golden Expected Results
    localparam logic [DATA_W-1:0] MW_EXPECTED_X = 256'h5ba002fcd2ff6d0f266010e9cd0474086e089a79bf8cf1b92bc52965c6407cf2;
    localparam logic [DATA_W-1:0] MW_EXPECTED_Y = 256'h5fa9a5b618ceb06dfd62ec7dc69be62280f858111be15ed1f7a41ffd9f58b15c;

    // ------------------------------------------------------------------------
    // Clocks and Resets
    // ------------------------------------------------------------------------
    logic msm_clk;
    logic msm_rst_n;
    logic axi_clk;
    logic axi_rst_n;

    initial begin
        msm_clk = 0;
        forever #5 msm_clk = ~msm_clk;   // 100 MHz
    end

    initial begin
        axi_clk = 0;
        forever #4 axi_clk = ~axi_clk;   // 125 MHz
    end

    // ------------------------------------------------------------------------
    // Memory Model (Loaded from MEMH)
    // ------------------------------------------------------------------------
    logic [AXI_DATA_W-1:0] axi_mem_array [0:8191];

    // ------------------------------------------------------------------------
    // AXI Interface Signals
    // ------------------------------------------------------------------------
    logic [63:0] m_axi_araddr;
    logic [7:0]  m_axi_arlen;
    logic [2:0]  m_axi_arsize;
    logic [1:0]  m_axi_arburst;
    logic        m_axi_arvalid;
    logic        m_axi_arready;

    logic [AXI_DATA_W-1:0] m_axi_rdata;
    logic [1:0]            m_axi_rresp;
    logic                  m_axi_rlast;
    logic                  m_axi_rvalid;
    logic                  m_axi_rready;

    // ------------------------------------------------------------------------
    // DUT Control and Outputs
    // ------------------------------------------------------------------------
    logic start;
    logic busy;
    logic done;
    logic [DATA_W-1:0] final_x;
    logic [DATA_W-1:0] final_y;
    logic [DATA_W-1:0] final_z;

    // ------------------------------------------------------------------------
    // AXI Slave Read Model
    // ------------------------------------------------------------------------
    logic        slave_active_q;
    logic [63:0] slave_addr_q;
    logic [8:0]  slave_beats_left_q;
    logic [31:0] gap_lfsr_q;
    
    int ar_burst_count;
    int r_beat_count;
    int error_count;

    // Using 'always' instead of 'always_ff' to prevent *MULAXX errors.
    always @(posedge axi_clk or negedge axi_rst_n) begin
        if (!axi_rst_n) begin
            slave_active_q     <= 1'b0;
            slave_addr_q       <= '0;
            slave_beats_left_q <= '0;
            gap_lfsr_q         <= 32'h1ACE_B00C; // Seed
            m_axi_arready      <= 1'b0;
            m_axi_rvalid       <= 1'b0;
            m_axi_rlast        <= 1'b0;
            m_axi_rdata        <= '0;
            m_axi_rresp        <= 2'b00;
            ar_burst_count     <= 0;
            r_beat_count       <= 0;
        end else begin
            // LFSR for optional backpressure (randomizes RVALID)
            gap_lfsr_q <= {gap_lfsr_q[30:0], gap_lfsr_q[31] ^ gap_lfsr_q[21] ^ gap_lfsr_q[1] ^ gap_lfsr_q[0]};

            // Always accept AR if not currently servicing a burst
            m_axi_arready <= !slave_active_q;

            // AR Handshake
            if (m_axi_arvalid && !slave_active_q) begin
                slave_active_q     <= 1'b1;
                slave_addr_q       <= m_axi_araddr;
                slave_beats_left_q <= m_axi_arlen + 1;
                ar_burst_count     <= ar_burst_count + 1;
                m_axi_arready      <= 1'b0;
            end

            // R Handshake and Burst Generation
            if (slave_active_q) begin
                if (!m_axi_rvalid || (m_axi_rvalid && m_axi_rready)) begin
                    // Add slight random backpressure using LFSR
                    if ((gap_lfsr_q[1:0] == 2'b00) && (r_beat_count > 0)) begin
                        m_axi_rvalid <= 1'b0;
                    end else begin
                        m_axi_rvalid <= 1'b1;
                        m_axi_rdata  <= axi_mem_array[(slave_addr_q - AXI_BASE_ADDR) / 64];
                        m_axi_rlast  <= (slave_beats_left_q == 1);
                        m_axi_rresp  <= 2'b00; // OKAY
                        
                        r_beat_count <= r_beat_count + 1;
                        slave_addr_q <= slave_addr_q + 64; // Advance by 64 bytes
                        slave_beats_left_q <= slave_beats_left_q - 1;

                        if (slave_beats_left_q == 1) begin
                            slave_active_q <= 1'b0;
                        end
                    end
                end
            end else begin
                m_axi_rvalid <= 1'b0;
                m_axi_rlast  <= 1'b0;
            end
        end
    end

    // ------------------------------------------------------------------------
    // DUT Instantiation
    // ------------------------------------------------------------------------
    msm_axi_affine_multiwindow_top_dualclk_v1 dut (
        .msm_clk        (msm_clk),
        .msm_rst_n      (msm_rst_n),
        .axi_clk        (axi_clk),
        .axi_rst_n      (axi_rst_n),
        
        .start          (start),
        .base_addr      (AXI_BASE_ADDR),
        .logical_points_per_window(32'd256),
        .busy           (busy),
        .done           (done),
        .window_index   (), // Unconnected
        .result_x       (final_x),
        .result_y       (final_y),
        .result_z       (final_z),

        .m_axi_araddr   (m_axi_araddr),
        .m_axi_arlen    (m_axi_arlen),
        .m_axi_arsize   (m_axi_arsize),
        .m_axi_arburst  (m_axi_arburst),
        .m_axi_arvalid  (m_axi_arvalid),
        .m_axi_arready  (m_axi_arready),
        .m_axi_rdata    (m_axi_rdata),
        .m_axi_rresp    (m_axi_rresp),
        .m_axi_rlast    (m_axi_rlast),
        .m_axi_rvalid   (m_axi_rvalid),
        .m_axi_rready   (m_axi_rready),

        .converter_busy (),
        .converter_pending_count (),
        .converter_result_count ()
    );

    // ------------------------------------------------------------------------
    // Progress Monitor
    // ------------------------------------------------------------------------
    longint msm_cycle;
    logic is_running;

    always_ff @(posedge msm_clk or negedge msm_rst_n) begin
        if (!msm_rst_n) begin
            msm_cycle <= 0;
            is_running <= 0;
        end else begin
            msm_cycle <= msm_cycle + 1;
            if (start) is_running <= 1;
            if (done)  is_running <= 0;
        end
    end

    always @(posedge msm_clk) begin
        if (is_running && (msm_cycle % 100 == 0)) begin
            $display("[MEMH_V7_PROGRESS] time=%0t msm_cycle=%0d window=%0d", 
                      $time, msm_cycle, dut.u_controller.current_window);
            $display("ar=%0d r=%0d accepted=%0d completed=%0d last=%0d", 
                      ar_burst_count, r_beat_count, 
                      dut.u_controller.u_window.build_accepted_count, 
                      dut.u_controller.u_window.scheduler_total_completed_count, 
                      dut.u_controller.u_window.build_last_seen);
            $display("window_state=%0d controller_state=%0d\n", 
                      dut.u_controller.u_window.state, dut.u_controller.state);
        end
    end

    // ------------------------------------------------------------------------
    // Main Stimulus & Checks
    // ------------------------------------------------------------------------
    initial begin
        error_count = 0;
        start = 0;
        msm_rst_n = 0;
        axi_rst_n = 0;

        // Load the compact MEMH file
        $display("[TB] Loading compact MEMH memory...");
        $readmemh("vectors/multiwindow_w16_python_golden_affine_256p_axi.memh", axi_mem_array);
        
        #100;
        msm_rst_n = 1;
        axi_rst_n = 1;
        #100;
        
        @(posedge msm_clk);
        start = 1;
        @(posedge msm_clk);
        start = 0;

        $display("[TB] Start signaled. Waiting for completion...");

        // Wait for done or timeout
        fork
            begin
                wait(done);
                $display("\n[TB] MSM Completed at cycle %0d", msm_cycle);
            end
            begin
                #1000000000; // Watchdog Timeout (~1M cycles)
                $display("\n[ERROR] Watchdog Timeout! Simulation stuck.");
                $finish;
            end
        join_any
        disable fork;

        // Golden Checks
        if (final_x !== MW_EXPECTED_X) begin
            $display("[FAIL] X mismatch!");
            $display("Expected: %x", MW_EXPECTED_X);
            $display("Got     : %x", final_x);
            error_count++;
        end

        if (final_y !== MW_EXPECTED_Y) begin
            $display("[FAIL] Y mismatch!");
            $display("Expected: %x", MW_EXPECTED_Y);
            $display("Got     : %x", final_y);
            error_count++;
        end

        if (error_count == 0) begin
            $display("\n=============================================");
            $display(" [SUCCESS] 256-point Affine AXI Simulation PASS");
            $display("=============================================");
        end else begin
            $display("\n[TEST FAILED] with %0d errors.", error_count);
        end

        $finish;
    end

endmodule
