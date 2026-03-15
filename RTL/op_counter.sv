```systemverilog
// ==========================================================
// Op Counter Module
// Translated from hardware-aware Python code
// ==========================================================

module op_counter #(
    parameter int COUNTER_WIDTH = 32
)(
    input  logic                     clk,
    input  logic                     rst_n,            // Asynchronous active-low reset
    input  logic                     i_reset_counters, // Synchronous clear (reset_counters function)

    // Point operation increment enables
    input  logic                     i_affine_add_en,
    input  logic                     i_jacobian_add_en,
    input  logic                     i_jacobian_mixed_add_en,
    input  logic                     i_jacobian_double_en,
    input  logic                     i_extended_add_en,
    input  logic                     i_extended_mixed_add_en,
    input  logic                     i_extended_double_en,

    // Field operation increment enables
    input  logic                     i_field_add_en,
    input  logic                     i_field_sub_en,
    input  logic                     i_field_mul_en,
    input  logic                     i_field_inv_en,
    input  logic                     i_field_neg_en,

    // Counter outputs (Global variables in Python)
    output logic [COUNTER_WIDTH-1:0] o_affine_add_count,
    output logic [COUNTER_WIDTH-1:0] o_jacobian_add_count,
    output logic [COUNTER_WIDTH-1:0] o_jacobian_mixed_add_count,
    output logic [COUNTER_WIDTH-1:0] o_jacobian_double_count,
    output logic [COUNTER_WIDTH-1:0] o_extended_add_count,
    output logic [COUNTER_WIDTH-1:0] o_extended_mixed_add_count,
    output logic [COUNTER_WIDTH-1:0] o_extended_double_count,

    output logic [COUNTER_WIDTH-1:0] o_field_add_count,
    output logic [COUNTER_WIDTH-1:0] o_field_sub_count,
    output logic [COUNTER_WIDTH-1:0] o_field_mul_count,
    output logic [COUNTER_WIDTH-1:0] o_field_inv_count,
    output logic [COUNTER_WIDTH-1:0] o_field_neg_count,
    
    // Derived metric (total_field_ops in print_counters)
    output logic [COUNTER_WIDTH-1:0] o_total_field_ops
);

    // Internal registers for counters
    logic [COUNTER_WIDTH-1:0] affine_add_count;
    logic [COUNTER_WIDTH-1:0] jacobian_add_count;
    logic [COUNTER_WIDTH-1:0] jacobian_mixed_add_count;
    logic [COUNTER_WIDTH-1:0] jacobian_double_count;
    logic [COUNTER_WIDTH-1:0] extended_add_count;
    logic [COUNTER_WIDTH-1:0] extended_mixed_add_count;
    logic [COUNTER_WIDTH-1:0] extended_double_count;
    logic [COUNTER_WIDTH-1:0] field_add_count;
    logic [COUNTER_WIDTH-1:0] field_sub_count;
    logic [COUNTER_WIDTH-1:0] field_mul_count;
    logic [COUNTER_WIDTH-1:0] field_inv_count;
    logic [COUNTER_WIDTH-1:0] field_neg_count;

    // Sequential logic: Handles initialization (global assignments), 
    // the reset_counters() function, and increment operations.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            affine_add_count           <= '0;
            jacobian_add_count         <= '0;
            jacobian_mixed_add_count   <= '0;
            jacobian_double_count      <= '0;
            extended_add_count         <= '0;
            extended_mixed_add_count   <= '0;
            extended_double_count      <= '0;
            field_add_count            <= '0;
            field_sub_count            <= '0;
            field_mul_count            <= '0;
            field_inv_count            <= '0;
            field_neg_count            <= '0;
        end else if (i_reset_counters) begin
            affine_add_count           <= '0;
            jacobian_add_count         <= '0;
            jacobian_mixed_add_count   <= '0;
            jacobian_double_count      <= '0;
            extended_add_count         <= '0;
            extended_mixed_add_count   <= '0;
            extended_double_count      <= '0;
            field_add_count            <= '0;
            field_sub_count            <= '0;
            field_mul_count            <= '0;
            field_inv_count            <= '0;
            field_neg_count            <= '0;
        end else begin
            // Point operation increments
            if (i_affine_add_en)         affine_add_count         <= affine_add_count + 1'b1;
            if (i_jacobian_add_en)       jacobian_add_count       <= jacobian_add_count + 1'b1;
            if (i_jacobian_mixed_add_en) jacobian_mixed_add_count <= jacobian_mixed_add_count + 1'b1;
            if (i_jacobian_double_en)    jacobian_double_count    <= jacobian_double_count + 1'b1;
            if (i_extended_add_en)       extended_add_count       <= extended_add_count + 1'b1;
            if (i_extended_mixed_add_en) extended_mixed_add_count <= extended_mixed_add_count + 1'b1;
            if (i_extended_double_en)    extended_double_count    <= extended_double_count + 1'b1;
            
            // Field operation increments
            if (i_field_add_en)          field_add_count          <= field_add_count + 1'b1;
            if (i_field_sub_en)          field_sub_count          <= field_sub_count + 1'b1;
            if (i_field_mul_en)          field_mul_count          <= field_mul_count + 1'b1;
            if (i_field_inv_en)          field_inv_count          <= field_inv_count + 1'b1;
            if (i_field_neg_en)          field_neg_count          <= field_neg_count + 1'b1;
        end
    end

    // Combinational logic: Total field operations (from print_counters title/calculation)
    always_comb begin
        o_total_field_ops = field_add_count + field_sub_count + field_mul_count + 
                           field_inv_count + field_neg_count;
    end

    // Output port mapping
    assign o_affine_add_count         = affine_add_count;
    assign o_jacobian_add_count       = jacobian_add_count;
    assign o_jacobian_mixed_add_count = jacobian_mixed_add_count;
    assign o_jacobian_double_count    = jacobian_double_count;
    assign o_extended_add_count       = extended_add_count;
    assign o_extended_mixed_add_count = extended_mixed_add_count;
    assign o_extended_double_count    = extended_double_count;
    assign o_field_add_count          = field_add_count;
    assign o_field_sub_count          = field_sub_count;
    assign o_field_mul_count          = field_mul_count;
    assign o_field_inv_count          = field_inv_count;
    assign o_field_neg_count          = field_neg_count;

endmodule
```
