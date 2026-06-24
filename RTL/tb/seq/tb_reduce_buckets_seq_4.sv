`timescale 1ns/1ps

module tb_reduce_buckets_seq_4;

    localparam int WIDTH = 256;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    // bucket1 = 2G, Jacobian Montgomery
    localparam logic [255:0] B1_X =
        256'h7C75DD9524177D593C03889B8DCD9B1CB05FB7D2A3DA7FE8BA9F29B104E7DB13;

    localparam logic [255:0] B1_Y =
        256'h55DEBB381F4AD034CC27CB48A46449AAA87D43FDB563384B1CD20838E6FDDC9F;

    localparam logic [255:0] B1_Z =
        256'h9E7F0A3FA94B05ACE16D6B355833826D1BF8BABA3E3B8C9B62BD4DA6A7B75B95;

    // bucket2 = 2G, affine-as-Jacobian Montgomery
    localparam logic [255:0] B2_X =
        256'hF918623CCBA0EE23CE0B62E1E014040471354AFC88B285A04E0640C981048D2C;

    localparam logic [255:0] B2_Y =
        256'h3C7F7712157B93134B3A0F64BDA2CC6584FD25167DC75CE17D12D622FFACCFBF;

    localparam logic [255:0] B2_Z =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    // bucket3 = 3G, affine-as-Jacobian Montgomery
    localparam logic [255:0] B3_X =
        256'h9497730FCDF4C0AD5940D07385985972066CEAFB22EB7BC42379D4BBD5FEA781;

    localparam logic [255:0] B3_Y =
        256'h3EC28DCD9215EC76CC6048BD84885650AC4964CDC5A1F91FAF18B0B0613F55A9;

    localparam logic [255:0] B3_Z =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    // Expected reduction result:
    //
    // running_sum = INF
    // result      = INF
    //
    // i=3: running_sum = B3,          result = B3
    // i=2: running_sum = B3+B2,       result = B3 + (B3+B2)
    // i=1: running_sum = B3+B2+B1,    result = previous + running_sum
    //
    // With bucket1=2G, bucket2=2G, bucket3=3G:
    // result = 1*B1 + 2*B2 + 3*B3
    //        = 2G + 4G + 9G
    //        = 15G
    //
    // Expected below is in Jacobian Montgomery coordinates for this reduction flow.
    localparam logic [255:0] EXP_X =
        256'h095BC488048E05A5732C475C3A609EFCC38EC30F0B30A04E778684E3DD149772;

    localparam logic [255:0] EXP_Y =
        256'hA95C53D653DEE15BE8482AD23B040A470B14DC6069A4204C751D6C1C6D8FDC7D;

    localparam logic [255:0] EXP_Z =
        256'h93F59AC795686CC45912CC7F9918DE3F914DBDA84AD1331E5C2FD8DA0EBF9998;

    logic clk = 1'b0;
    logic rst_n;

    logic start;
    logic busy;
    logic done;

    logic [WIDTH-1:0] result_x;
    logic [WIDTH-1:0] result_y;
    logic [WIDTH-1:0] result_z;

    always #5 clk = ~clk;

    reduce_buckets_seq_4 #(
        .WIDTH(WIDTH)
    ) dut (
        .clk   (clk),
        .rst_n (rst_n),

        .start (start),

        .b1_x (B1_X),
        .b1_y (B1_Y),
        .b1_z (B1_Z),

        .b2_x (B2_X),
        .b2_y (B2_Y),
        .b2_z (B2_Z),

        .b3_x (B3_X),
        .b3_y (B3_Y),
        .b3_z (B3_Z),

        .busy (busy),
        .done (done),

        .result_x (result_x),
        .result_y (result_y),
        .result_z (result_z)
    );

    task automatic run_reduce_check;
        int cycles;

        begin
            @(posedge clk);

            if (busy) begin
                $fatal(1, "Tried to start while busy");
            end

            start <= 1'b1;

            @(posedge clk);
            start <= 1'b0;

            cycles = 0;
            while (!done) begin
                @(posedge clk);
                cycles++;

                if (cycles > 5000) begin
                    $fatal(1, "TIMEOUT waiting for done");
                end
            end

            $display("[reduce_buckets_seq_4] latency = %0d cycles", cycles);
            $display("[reduce_buckets_seq_4] X = %h", result_x);
            $display("[reduce_buckets_seq_4] Y = %h", result_y);
            $display("[reduce_buckets_seq_4] Z = %h", result_z);

            if (result_x !== EXP_X || result_y !== EXP_Y || result_z !== EXP_Z) begin
                $display("EXPECTED X = %h", EXP_X);
                $display("GOT      X = %h", result_x);
                $display("EXPECTED Y = %h", EXP_Y);
                $display("GOT      Y = %h", result_y);
                $display("EXPECTED Z = %h", EXP_Z);
                $display("GOT      Z = %h", result_z);
                $fatal(1, "reduce_buckets_seq_4 mismatch");
            end

            $display("[reduce_buckets_seq_4] PASSED");

            @(posedge clk);
        end
    endtask

    initial begin
        rst_n = 1'b0;
        start = 1'b0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        $display("==============================================");
        $display(" tb_reduce_buckets_seq_4 START");
        $display("==============================================");

        run_reduce_check();

        $display("==============================================");
        $display(" tb_reduce_buckets_seq_4 PASSED");
        $display("==============================================");

        $finish;
    end

endmodule