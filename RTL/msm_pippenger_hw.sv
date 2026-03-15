```systemverilog
//-----------------------------------------------------------------------------
// Module: msm_pippenger_tiled_pingpong
// Description: HW-aware Pippenger MSM implementation.
// Translates the tiled, ping-pong bucket-buffering logic from Python to RTL.
//-----------------------------------------------------------------------------

module msm_pippenger_tiled_pingpong #(
    parameter int W             = 16,        // Window size
    parameter int TILE_SIZE     = 4,         // Number of windows per pass (tile)
    parameter int SCALAR_BITS   = 256,       // Bits per scalar
    parameter int COORD_WIDTH   = 256,       // Field element width
    parameter logic [COORD_WIDTH-1:0] ONE_M = 256'd1 // Montgomery representation of 1
)(
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     start,
    
    // Scalar and Affine Point streaming (Pre-converted to Montgomery Affine)
    input  logic [SCALAR_BITS-1:0]   in_scalar,
    input  logic [COORD_WIDTH-1:0]   in_point_x,
    input  logic [COORD_WIDTH-1:0]   in_point_y,
    input  logic                     in_valid,
    input  logic [31:0]              n_points,
    
    // Result (Jacobian Montgomery)
    output logic [COORD_WIDTH-1:0]   res_x,
    output logic [COORD_WIDTH-1:0]   res_y,
    output logic [COORD_WIDTH-1:0]   res_z,
    output logic                     done
);

    //-------------------------------------------------------------------------
    // Localparams and Types
    //-------------------------------------------------------------------------
    localparam int NUM_WINDOWS = (SCALAR_BITS + W - 1) / W;
    localparam int NUM_BUCKETS = 1 << W;

    typedef enum logic [3:0] {
        ST_IDLE,
        ST_BUILD_FIRST_TILE,
        ST_PIPELINE_START,
        ST_PIPELINE_RUN,
        ST_REDUCE_LAST,
        ST_DONE
    } state_t;

    state_t state;

    //-------------------------------------------------------------------------
    // Bucket RAM Signals (Ping/Pong)
    //-------------------------------------------------------------------------
    // In a real HW implementation, these would be external BRAMs or DDR controllers.
    // Represented here as logic for the translation flow.
    logic [COORD_WIDTH-1:0] bucket_mem_x [0:1][0:TILE_SIZE-1][0:NUM_BUCKETS-1];
    logic [COORD_WIDTH-1:0] bucket_mem_y [0:1][0:TILE_SIZE-1][0:NUM_BUCKETS-1];
    logic [COORD_WIDTH-1:0] bucket_mem_z [0:1][0:TILE_SIZE-1][0:NUM_BUCKETS-1];

    logic ping_pong_sel; // 0 for buf0, 1 for buf1

    //-------------------------------------------------------------------------
    // Internal Registers for Pippenger Logic
    //-------------------------------------------------------------------------
    logic [COORD_WIDTH-1:0] R_x, R_y, R_z;
    logic [COORD_WIDTH-1:0] running_x, running_y, running_z;
    logic [COORD_WIDTH-1:0] result_x, result_y, result_z;
    
    logic [31:0] point_cnt;
    logic [7:0]  win_idx_cnt;
    logic [W:0]  bucket_cnt; // W+1 to handle range down to 0
    logic [7:0]  tile_win_hi, tile_win_lo;
    logic        first_window;

    //-------------------------------------------------------------------------
    // Arithmetic Units Interface (Combinational Logic / Tasks)
    //-------------------------------------------------------------------------
    // These blocks represent the logic of jacobian_add, double, and mixed_add.
    // In a high-performance design, these would be pipelined modules.
    
    function automatic void jacobian_add(
        input  logic [COORD_WIDTH-1:0] ax, ay, az,
        input  logic [COORD_WIDTH-1:0] bx, by, bz,
        output logic [COORD_WIDTH-1:0] rx, ry, rz
    );
        // Logical flow for Jacobian addition
        // If A is INF (az==0), return B. If B is INF (bz==0), return A.
        if (az == 0) begin rx = bx; ry = by; rz = bz; end
        else if (bz == 0) begin rx = ax; ry = ay; rz = az; end
        else begin
            // ... implementation of jacobian_add logic ...
            rx = 0; ry = 0; rz = 0; // Placeholder
        end
    endfunction

    function automatic void jacobian_double(
        input  logic [COORD_WIDTH-1:0] ax, ay, az,
        output logic [COORD_WIDTH-1:0] rx, ry, rz
    );
        // ... implementation of jacobian_double logic ...
        rx = 0; ry = 0; rz = 0; // Placeholder
    endfunction

    function automatic void jacobian_mixed_add_mont(
        input  logic [COORD_WIDTH-1:0] jx, jy, jz,
        input  logic [COORD_WIDTH-1:0] ax, ay,
        output logic [COORD_WIDTH-1:0] rx, ry, rz
    );
        // If Jacobian bucket is INF (jz==0), result is (ax, ay, ONE_M)
        if (jz == 0) begin
            rx = ax; ry = ay; rz = ONE_M;
        end else begin
            // ... implementation of jacobian_mixed_add_mont ...
            rx = 0; ry = 0; rz = 0; // Placeholder
        end
    endfunction

    //-------------------------------------------------------------------------
    // FSM Control Logic
    //-------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            ping_pong_sel <= 0;
            done <= 0;
            R_x <= 0; R_y <= 0; R_z <= 0;
            first_window <= 1;
        end else begin
            case (state)
                ST_IDLE: begin
                    done <= 0;
                    if (start) begin
                        state <= ST_BUILD_FIRST_TILE;
                        tile_win_hi <= NUM_WINDOWS - 1;
                        tile_win_lo <= (NUM_WINDOWS - 1 >= TILE_SIZE) ? (NUM_WINDOWS - TILE_SIZE) : 0;
                        ping_pong_sel <= 0; // Start with buf0
                        point_cnt <= 0;
                        first_window <= 1;
                        // Initial result is INF
                        R_x <= 0; R_y <= 0; R_z <= 0;
                    end
                end

                // BUILD phase: Streaming points into bucket buffers
                ST_BUILD_FIRST_TILE: begin
                    if (in_valid) begin
                        // For each window in current tile, calculate bucket index
                        for (int i = 0; i < TILE_SIZE; i++) begin
                            int win = tile_win_lo + i;
                            logic [W-1:0] b;
                            b = (in_scalar >> (win * W)) & ((1 << W) - 1);
                            
                            if (b != 0) begin
                                jacobian_mixed_add_mont(
                                    bucket_mem_x[ping_pong_sel][i][b],
                                    bucket_mem_y[ping_pong_sel][i][b],
                                    bucket_mem_z[ping_pong_sel][i][b],
                                    in_point_x, in_point_y,
                                    bucket_mem_x[ping_pong_sel][i][b],
                                    bucket_mem_y[ping_pong_sel][i][b],
                                    bucket_mem_z[ping_pong_sel][i][b]
                                );
                            end
                        end
                        
                        if (point_cnt == n_points - 1) begin
                            state <= ST_PIPELINE_START;
                            point_cnt <= 0;
                        end else begin
                            point_cnt <= point_cnt + 1;
                        end
                    end
                end

                ST_PIPELINE_START: begin
                    // Prepare next tile boundaries
                    ping_pong_sel <= ~ping_pong_sel; // Swap Build/Reduce buffers
                    if (tile_win_lo == 0) begin
                        state <= ST_REDUCE_LAST;
                    end else begin
                        tile_win_hi <= tile_win_lo - 1;
                        tile_win_lo <= (tile_win_lo >= TILE_SIZE) ? (tile_win_lo - TILE_SIZE) : 0;
                        state <= ST_PIPELINE_RUN;
                    end
                    // Init reduction counters
                    win_idx_cnt <= TILE_SIZE - 1; 
                    bucket_cnt <= NUM_BUCKETS - 1;
                    running_x <= 0; running_y <= 0; running_z <= 0;
                    result_x <= 0; result_y <= 0; result_z <= 0;
                end

                // Pipeline Loop: Reduce previous tile while building current tile
                ST_PIPELINE_RUN: begin
                    // -- Reduction Logic (from ping_pong_sel ^ 1) --
                    // shift_window(R, W) logic
                    if (bucket_cnt == NUM_BUCKETS - 1) begin
                        if (!first_window) begin
                            // R = 2^W * R
                            for (int k = 0; k < W; k++) begin
                                jacobian_double(R_x, R_y, R_z, R_x, R_y, R_z);
                            end
                        end else begin
                            first_window <= 0;
                        end
                    end

                    // reduce_buckets_pippenger logic
                    if (bucket_cnt > 0) begin
                        if (bucket_mem_z[~ping_pong_sel][win_idx_cnt][bucket_cnt] != 0) begin
                            jacobian_add(
                                running_x, running_y, running_z,
                                bucket_mem_x[~ping_pong_sel][win_idx_cnt][bucket_cnt],
                                bucket_mem_y[~ping_pong_sel][win_idx_cnt][bucket_cnt],
                                bucket_mem_z[~ping_pong_sel][win_idx_cnt][bucket_cnt],
                                running_x, running_y, running_z
                            );
                        end
                        jacobian_add(result_x, result_y, result_z, running_x, running_y, running_z, result_x, result_y, result_z);
                        bucket_cnt <= bucket_cnt - 1;
                    end else if (win_idx_cnt > 0) begin
                        // Add window sum to total R
                        jacobian_add(R_x, R_y, R_z, result_x, result_y, result_z, R_x, R_y, R_z);
                        // Move to next window in tile
                        win_idx_cnt <= win_idx_cnt - 1;
                        bucket_cnt <= NUM_BUCKETS - 1;
                        running_x <= 0; running_y <= 0; running_z <= 0;
                        result_x <= 0; result_y <= 0; result_z <= 0;
                        // Prepare for next window's shift
                        for (int k = 0; k < W; k++) begin
                             jacobian_double(R_x, R_y, R_z, R_x, R_y, R_z);
                        end
                    end

                    // -- Build Logic (into ping_pong_sel) --
                    // (Similar to ST_BUILD_FIRST_TILE, concurrent processing)
                    if (in_valid) begin
                        // Point processing...
                        if (point_cnt == n_points - 1) begin
                            // Check if reduction also finished, then loop
                        end
                    end
                end

                ST_REDUCE_LAST: begin
                    // Final reduction for the last pending tile in buffer
                    // ... (Implementation of final reduction loop) ...
                    state <= ST_DONE;
                end

                ST_DONE: begin
                    res_x <= R_x;
                    res_y <= R_y;
                    res_z <= R_z;
                    done <= 1;
                    state <= ST_IDLE;
                end
            endcase
        end
    end

endmodule
```
