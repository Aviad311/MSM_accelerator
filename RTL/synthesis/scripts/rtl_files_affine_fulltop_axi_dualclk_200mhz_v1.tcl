# rtl_files_affine_fulltop_axi_dualclk_200mhz_v1.tcl
# Clean RTL file list for the affine full-top AXI dual-clock 200MHz elaboration.
#
# This list targets the module used by most final End-to-End affine TBs:
#   msm_axi_affine_multiwindow_top_dualclk_v1
#
# Expected chain:
#   msm_axi_affine_multiwindow_top_dualclk_v1
#     -> axi_point_stream_source_dualclk_v1
#          -> async_fifo_gray_v1
#     -> msm_multiwindow_controller_v1
#          -> pippenger_window_mem_stream_top_8lane_pipeline_reduce4mul_overlap_sram_macro_v2
#               -> bucket_mem_3coord_sram_macro_v2
#               -> bucket_update_scheduler_8lane_v1
#               -> reduce_buckets_mem_4mul_overlap
#               -> arithmetic modules
#
# Alternative versions intentionally excluded:
#   - axi_point_stream_source_dualclk_v2.sv
#   - axi_point_stream_source_v2.sv
#   - msm_axi_multiwindow_top_v2.sv
#   - msm_axi_multiwindow_top_dualclk_v1.sv
#   - msm_affine_frontend_top.sv
#   - msm_point_streamer_block.sv
#   - point_to_montgomery_stream.sv
#   - axi_point_fetch_fifo.sv
#   - bucket_mem_3coord.sv
#   - bucket_mem_3coord_sram_macro_deep_v1.sv
#   - pippenger_window_mem_stream_top_param_lanes_sram_macro_v1.sv
#   - pippenger_window_mem_stream_top_8lane_pipeline_reduce4mul_overlap.sv
#   - simple_sync_sram_1rw.sv
#   - SRAM macro Verilog models under rtl/mem/sram_macro/*/verilog
#
# SRAM macros are provided through Liberty files in the elaboration/synthesis script.

read_hdl -sv rtl/montgomery/secp256k1_montgomery_mul.sv
read_hdl -sv rtl/montgomery/field_mul_seq.sv

read_hdl -sv rtl/seq/jacobian_double_seq.sv
read_hdl -sv rtl/seq/jacobian_add_4mul_seq.sv
read_hdl -sv rtl/seq/jacobian_mixed_add_pipeline_v2.sv

read_hdl -sv rtl/mem/active/sram_8192x16_tag_macro.sv
read_hdl -sv rtl/mem/active/sram_8192x256_macro.sv
read_hdl -sv rtl/mem/active/bucket_mem_3coord_sram_macro_v2.sv
read_hdl -sv rtl/mem/active/bucket_update_pipeline_v1.sv
read_hdl -sv rtl/mem/active/bucket_update_scheduler_v1.sv
read_hdl -sv rtl/mem/active/bucket_update_scheduler_8lane_v1.sv
read_hdl -sv rtl/mem/active/reduce_buckets_mem_4mul_overlap.sv
read_hdl -sv rtl/mem/active/pippenger_window_mem_stream_top_8lane_pipeline_reduce4mul_overlap_sram_macro_v2.sv
read_hdl -sv rtl/mem/active/msm_multiwindow_controller_v1.sv

read_hdl -sv rtl/frontend/async_fifo_gray_v1.sv
read_hdl -sv rtl/frontend/axi_point_stream_source_dualclk_v1.sv
read_hdl -sv rtl/frontend/point_to_montgomery_stream.sv
read_hdl -sv rtl/frontend/msm_axi_affine_multiwindow_top_dualclk_v1.sv