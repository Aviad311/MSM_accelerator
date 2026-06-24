`timescale 1ns/1ps

module bucket_update_seq #(
    parameter int ADDR_W = 4,
    parameter int DATA_W = 256,
    parameter int DEPTH  = (1 << ADDR_W)
)(
    input  logic                clk,
    input  logic                rst_n,

    input  logic                start,
    input  logic                clear_all,

    input  logic [ADDR_W-1:0]   bucket_id,
    input  logic [DATA_W-1:0]   point_x,
    input  logic [DATA_W-1:0]   point_y,

    output logic                busy,
    output logic                done,
    output logic                skipped,

    output logic [DATA_W-1:0]   last_x,
    output logic [DATA_W-1:0]   last_y,
    output logic [DATA_W-1:0]   last_z
);

    localparam logic [255:0] ZERO =
        256'h0000000000000000000000000000000000000000000000000000000000000000;

    localparam logic [255:0] ONE_M =
        256'h00000000000000000000000000000000000000000000000000000001000003D1;

    typedef enum logic [3:0] {
        S_IDLE,
        S_CLEAR_WRITE,
        S_READ_BUCKET,
        S_WAIT_BUCKET,
        S_ADD_START,
        S_ADD_WAIT,
        S_WRITE_BUCKET,
        S_DONE
    } state_t;

    state_t state;

    logic [ADDR_W-1:0] bucket_id_r;
    logic [ADDR_W-1:0] clear_addr_r;

    logic [DATA_W-1:0] point_x_r;
    logic [DATA_W-1:0] point_y_r;

    logic [DATA_W-1:0] bucket_x_r;
    logic [DATA_W-1:0] bucket_y_r;
    logic [DATA_W-1:0] bucket_z_r;

    logic mem_valid;
    logic mem_write_en;
    logic [ADDR_W-1:0] mem_addr;

    logic [DATA_W-1:0] mem_wdata_x;
    logic [DATA_W-1:0] mem_wdata_y;
    logic [DATA_W-1:0] mem_wdata_z;

    logic mem_ready;
    logic mem_rvalid;

    logic [DATA_W-1:0] mem_rdata_x;
    logic [DATA_W-1:0] mem_rdata_y;
    logic [DATA_W-1:0] mem_rdata_z;

    logic add_start;
    logic add_busy;
    logic add_done;

    logic [DATA_W-1:0] add_x3;
    logic [DATA_W-1:0] add_y3;
    logic [DATA_W-1:0] add_z3;

    bucket_mem_3coord #(
        .ADDR_W(ADDR_W),
        .DATA_W(DATA_W),
        .DEPTH (DEPTH)
    ) u_bucket_mem (
        .clk      (clk),
        .rst_n    (rst_n),

        .valid    (mem_valid),
        .write_en (mem_write_en),
        .addr     (mem_addr),

        .wdata_x  (mem_wdata_x),
        .wdata_y  (mem_wdata_y),
        .wdata_z  (mem_wdata_z),

        .ready    (mem_ready),
        .rvalid   (mem_rvalid),

        .rdata_x  (mem_rdata_x),
        .rdata_y  (mem_rdata_y),
        .rdata_z  (mem_rdata_z)
    );
    

    jacobian_mixed_add_seq u_mixed_add (
        .clk    (clk),
        .rst_n  (rst_n),
        .start  (add_start),

        .X1     (bucket_x_r),
        .Y1     (bucket_y_r),
        .Z1     (bucket_z_r),

        .X2     (point_x_r),
        .Y2     (point_y_r),

        .busy   (add_busy),
        .done   (add_done),

        .X3     (add_x3),
        .Y3     (add_y3),
        .Z3     (add_z3)
    );

    assign busy = (state != S_IDLE);
    assign done = (state == S_DONE);

    always_comb begin
        mem_valid    = 1'b0;
        mem_write_en = 1'b0;
        mem_addr     = '0;

        mem_wdata_x  = '0;
        mem_wdata_y  = '0;
        mem_wdata_z  = '0;

        add_start    = 1'b0;

        unique case (state)

            S_CLEAR_WRITE: begin
                mem_valid    = 1'b1;
                mem_write_en = 1'b1;
                mem_addr     = clear_addr_r;

                mem_wdata_x  = ZERO;
                mem_wdata_y  = ONE_M;
                mem_wdata_z  = ZERO;
            end

            S_READ_BUCKET: begin
                mem_valid    = 1'b1;
                mem_write_en = 1'b0;
                mem_addr     = bucket_id_r;
            end

            S_ADD_START: begin
                add_start = 1'b1;
            end

            S_WRITE_BUCKET: begin
                mem_valid    = 1'b1;
                mem_write_en = 1'b1;
                mem_addr     = bucket_id_r;

                mem_wdata_x  = add_x3;
                mem_wdata_y  = add_y3;
                mem_wdata_z  = add_z3;
            end

            default: begin
            end

        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;

            bucket_id_r  <= '0;
            clear_addr_r <= '0;

            point_x_r    <= '0;
            point_y_r    <= '0;

            bucket_x_r   <= '0;
            bucket_y_r   <= ONE_M;
            bucket_z_r   <= '0;

            last_x       <= '0;
            last_y       <= ONE_M;
            last_z       <= '0;

            skipped      <= 1'b0;
        end else begin
            skipped <= 1'b0;

            unique case (state)

                S_IDLE: begin
                    if (start && clear_all) begin
                        clear_addr_r <= '0;
                        state        <= S_CLEAR_WRITE;
                    end else if (start) begin
                        bucket_id_r <= bucket_id;
                        point_x_r   <= point_x;
                        point_y_r   <= point_y;

                        if (bucket_id == '0) begin
                            last_x  <= ZERO;
                            last_y  <= ONE_M;
                            last_z  <= ZERO;
                            skipped <= 1'b1;
                            state   <= S_DONE;
                        end else begin
                            state <= S_READ_BUCKET;
                        end
                    end
                end

                S_CLEAR_WRITE: begin
                    if (clear_addr_r == ADDR_W'(DEPTH-1)) begin
                        state <= S_DONE;
                    end else begin
                        clear_addr_r <= clear_addr_r + 1'b1;
                        state        <= S_CLEAR_WRITE;
                    end
                end

                S_READ_BUCKET: begin
                    state <= S_WAIT_BUCKET;
                end

                S_WAIT_BUCKET: begin
                    if (mem_rvalid) begin
                        bucket_x_r <= mem_rdata_x;
                        bucket_y_r <= mem_rdata_y;
                        bucket_z_r <= mem_rdata_z;
                        state      <= S_ADD_START;
                    end
                end

                S_ADD_START: begin
                    state <= S_ADD_WAIT;
                end

                S_ADD_WAIT: begin
                    if (add_done) begin
                        last_x <= add_x3;
                        last_y <= add_y3;
                        last_z <= add_z3;
                        state  <= S_WRITE_BUCKET;
                    end
                end

                S_WRITE_BUCKET: begin
                    state <= S_DONE;
                end

                S_DONE: begin
                    state <= S_IDLE;
                end

                default: begin
                    state <= S_IDLE;
                end

            endcase
        end
    end

endmodule