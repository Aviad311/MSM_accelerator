`timescale 1ns/1ps

module simple_sync_sram_1rw #(
    parameter int ADDR_W     = 4,
    parameter int DATA_W     = 256,
    parameter int DEPTH      = (1 << ADDR_W),
    parameter int RD_LATENCY = 1
)(
    input  logic                clk,
    input  logic                rst_n,

    input  logic                valid,
    input  logic                write_en,
    input  logic [ADDR_W-1:0]   addr,
    input  logic [DATA_W-1:0]   wdata,

    output logic                ready,
    output logic                rvalid,
    output logic [DATA_W-1:0]   rdata
);

    // Behavioral 1RW synchronous SRAM model.
    //
    // Contract:
    // - ready is always high.
    // - write is accepted on posedge clk when valid && write_en.
    // - read is accepted on posedge clk when valid && !write_en.
    // - rvalid/rdata are returned after RD_LATENCY cycles.
    //
    // RD_LATENCY=1:
    //   read request sampled on clock N,
    //   rvalid/rdata are visible after that clock edge.
    //
    // RD_LATENCY=2:
    //   read request sampled on clock N,
    //   rvalid/rdata become valid one extra clock later.
    //
    // This is a temporary behavioral SRAM model.
    // The real SRAM macro wrapper will replace this later.

    logic [DATA_W-1:0] mem [0:DEPTH-1];

    assign ready = 1'b1;

    generate

        if (RD_LATENCY == 1) begin : GEN_RD_LATENCY_1

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    rvalid <= 1'b0;
                    rdata  <= '0;
                end else begin
                    rvalid <= 1'b0;

                    if (valid) begin
                        if (write_en) begin
                            mem[addr] <= wdata;
                        end else begin
                            rdata  <= mem[addr];
                            rvalid <= 1'b1;
                        end
                    end
                end
            end

        end else begin : GEN_RD_LATENCY_N

            logic [RD_LATENCY-1:0] rd_valid_pipe;
            logic [DATA_W-1:0]     rd_data_pipe [0:RD_LATENCY-1];

            integer i;

            // Output is directly driven from the last pipeline stage.
            // Do NOT register it again, otherwise actual latency becomes RD_LATENCY+1.
            assign rvalid = rd_valid_pipe[RD_LATENCY-1];
            assign rdata  = rd_data_pipe [RD_LATENCY-1];

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    rd_valid_pipe <= '0;

                    for (i = 0; i < RD_LATENCY; i = i + 1) begin
                        rd_data_pipe[i] <= '0;
                    end
                end else begin
                    // Stage 0 captures the new read request.
                    rd_valid_pipe[0] <= valid && !write_en;

                    if (valid && write_en) begin
                        mem[addr]       <= wdata;
                        rd_data_pipe[0] <= '0;
                    end else if (valid && !write_en) begin
                        rd_data_pipe[0] <= mem[addr];
                    end else begin
                        rd_data_pipe[0] <= '0;
                    end

                    // Shift read response pipeline.
                    for (i = 1; i < RD_LATENCY; i = i + 1) begin
                        rd_valid_pipe[i] <= rd_valid_pipe[i-1];
                        rd_data_pipe[i]  <= rd_data_pipe[i-1];
                    end
                end
            end

        end

    endgenerate

endmodule