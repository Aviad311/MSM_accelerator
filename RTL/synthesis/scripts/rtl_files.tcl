# ============================================================
# MSM Accelerator
# Active RTL source list for Cadence Genus
# Stage 1: RTL elaboration
# ============================================================

set RTL_ROOT [file normalize "."]

set RTL_FILES [list \
    $RTL_ROOT/rtl/montgomery/secp256k1_montgomery_mul.sv \
    $RTL_ROOT/rtl/montgomery/field_mul_seq.sv \
    \
    $RTL_ROOT/rtl/seq/jacobian_double_seq.sv \
    $RTL_ROOT/rtl/seq/jacobian_mixed_add_pipeline_v2.sv \
    $RTL_ROOT/rtl/seq/jacobian_add_4mul_seq.sv \
    \
    $RTL_ROOT/rtl/mem/active/bucket_update_pipeline_v1.sv \
    $RTL_ROOT/rtl/mem/active/bucket_update_scheduler_v1.sv \
    $RTL_ROOT/rtl/mem/active/bucket_update_scheduler_8lane_v1.sv \
    $RTL_ROOT/rtl/mem/active/reduce_buckets_mem_4mul_overlap.sv \
    \
    $RTL_ROOT/rtl/mem/active/sram_8192x256_macro.sv \
    $RTL_ROOT/rtl/mem/active/sram_8192x16_tag_macro.sv \
    $RTL_ROOT/rtl/mem/active/bucket_mem_3coord_sram_macro_v2.sv \
    \
    $RTL_ROOT/rtl/mem/active/pippenger_window_mem_stream_top_8lane_pipeline_reduce4mul_overlap_sram_macro_v2.sv \
]

puts "============================================================"
puts "MSM synthesis RTL file list"
puts "RTL root: $RTL_ROOT"
puts "Number of RTL files: [llength $RTL_FILES]"
puts "============================================================"

foreach rtl_file $RTL_FILES {
    if {![file exists $rtl_file]} {
        puts "ERROR: Missing RTL file:"
        puts "       $rtl_file"
        error "RTL source file not found"
    }

    puts "RTL: $rtl_file"
}