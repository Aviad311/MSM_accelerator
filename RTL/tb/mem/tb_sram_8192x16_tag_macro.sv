`timescale 1ns/1ps

module tb_sram_8192x16_tag_macro;

    logic        clk;
    logic        rst_n;
    logic        en;
    logic        we;
    logic [12:0] addr;
    logic [15:0] wdata;
    logic [15:0] rdata;
    logic        rvalid;

    sram_8192x16_tag_macro dut (
        .clk    (clk),
        .rst_n  (rst_n),
        .en     (en),
        .we     (we),
        .addr   (addr),
        .wdata  (wdata),
        .rdata  (rdata),
        .rvalid (rvalid)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    task automatic write_word(
        input logic [12:0] a,
        input logic [15:0] d
    );
        begin
            @(negedge clk);
            en    = 1'b1;
            we    = 1'b1;
            addr  = a;
            wdata = d;

            @(negedge clk);
            en    = 1'b0;
            we    = 1'b0;
            addr  = '0;
            wdata = '0;
        end
    endtask

    task automatic read_check(
        input logic [12:0] a,
        input logic [15:0] exp
    );
        begin
            @(negedge clk);
            en   = 1'b1;
            we   = 1'b0;
            addr = a;

            @(posedge clk);
            #0.1;

            if (rvalid !== 1'b1) begin
                $fatal(
                    1,
                    "[TB_TAG] rvalid missing for addr=%0d",
                    a
                );
            end

            if (rdata !== exp) begin
                $display(
                    "[TB_TAG] READ FAILED addr=%0d expected=%04h got=%04h",
                    a,
                    exp,
                    rdata
                );
                $fatal(1, "[TB_TAG] Tag SRAM mismatch.");
            end

            $display(
                "[TB_TAG] READ PASSED addr=%0d data=%04h",
                a,
                rdata
            );

            @(negedge clk);
            en   = 1'b0;
            we   = 1'b0;
            addr = '0;
        end
    endtask

    initial begin
        en    = 1'b0;
        we    = 1'b0;
        addr  = '0;
        wdata = '0;

        rst_n = 1'b0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        write_word(13'd0,    16'h0001);
        write_word(13'd1,    16'h1234);
        write_word(13'd4095, 16'hABCD);
        write_word(13'd8191, 16'hFFFF);

        read_check(13'd0,    16'h0001);
        read_check(13'd1,    16'h1234);
        read_check(13'd4095, 16'hABCD);
        read_check(13'd8191, 16'hFFFF);

        $display("");
        $display("[TB_TAG] sram_8192x16_tag_macro PASSED");
        $display("");

        #20;
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "[TB_TAG] Watchdog timeout.");
    end

endmodule