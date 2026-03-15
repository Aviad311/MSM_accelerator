`timescale 1ns / 1ps

/**
 * Module: msm_reference
 * 
 * Direct translation of the Python MSM reference model into synthesizable SystemVerilog.
 * Implements windowed Multi-Scalar Multiplication (MSM) with bucket accumulation and reduction.
 * 
 * Logic Flow:
 * 1. Pre-process: Convert affine points to Montgomery affine.
 * 2. Window Loop: Iterate from MSB to LSB windows.
 * 3. Double: Perform W doublings on the current result.
 * 4. Build Buckets: Aggregate points into buckets based on scalar window values.
 * 5. Reduce Buckets: Sum weighted buckets (i * bucket[i]) using repeated addition.
 * 6. Accumulate: Add bucket sum to the running result.
 */

module msm_reference #(
    parameter int FIELD_WIDTH  = 256,
    parameter int SCALAR_WIDTH = 256,
    parameter int W            = 16,     // Window size
    parameter int N_POINTS     = 1024,   // Number of points
    parameter [FIELD_WIDTH-1:0] ONE_M = 256'h0000000000000000000000000000000000000000000000000
