# ============================================================
# MSM Accelerator
# RTL file list for 16-window multiwindow controller
# ============================================================

set RTL_FILES [list \
    rtl/montgomery/secp256k1_montgomery_mul.sv \
    rtl/montgomery/field_mul_seq.sv \
    rtl/seq/jacobian_double_seq.sv \
    rtl/seq/jacobian_mixed_add_pipeline_v2.sv \
    rtl/seq/jacobian_add_4mul_seq.sv \
    rtl/mem/active/bucket_update_pipeline_v1.sv \
    rtl/mem/active/bucket_update_scheduler_v1.sv \
    rtl/mem/active/bucket_update_scheduler_8lane_v1.sv \
    rtl/mem/active/reduce_buckets_mem_4mul_overlap.sv \
    rtl/mem/active/sram_8192x256_macro.sv \
    rtl/mem/active/sram_8192x16_tag_macro.sv \
    rtl/mem/active/bucket_mem_3coord_sram_macro_v2.sv \
    rtl/mem/active/pippenger_window_mem_stream_top_8lane_pipeline_reduce4mul_overlap_sram_macro_v2.sv \
    rtl/mem/active/msm_multiwindow_controller_v1.sv \
]