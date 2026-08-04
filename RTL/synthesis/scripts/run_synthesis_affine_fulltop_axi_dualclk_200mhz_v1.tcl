# run_synthesis_affine_fulltop_axi_dualclk_200mhz_v1.tcl
# Affine full-top AXI dual-clock 200MHz synthesis flow.
#
# Starts from the elaborated + constrained DB:
#   synthesis/checkpoints/affine_fulltop_axi_dualclk_200mhz_v1/
#     affine_fulltop_axi_dualclk_elaborated_constrained_200mhz_v1.db
#
# Important methodology:
#   syn_generic -> write_db
#   syn_map     -> write_db
#   syn_opt     -> write_db
#
# This avoids losing several days of work if a report command fails after synthesis.

set TOP_MODULE msm_axi_affine_multiwindow_top_dualclk_v1

set SCRIPT_DIR [file dirname [file normalize [info script]]]
set SYNTH_DIR  [file normalize "$SCRIPT_DIR/.."]
set RTL_ROOT   [file normalize "$SYNTH_DIR/.."]

set RUN_NAME       "affine_fulltop_axi_dualclk_200mhz_synthesis_v1"
set REPORT_DIR     "$SYNTH_DIR/reports/$RUN_NAME"
set OUTPUT_DIR     "$SYNTH_DIR/outputs/$RUN_NAME"
set CHECKPOINT_DIR "$SYNTH_DIR/checkpoints/affine_fulltop_axi_dualclk_200mhz_v1"

set INPUT_DB   "$CHECKPOINT_DIR/affine_fulltop_axi_dualclk_elaborated_constrained_200mhz_v1.db"
set GENERIC_DB "$CHECKPOINT_DIR/affine_fulltop_axi_dualclk_after_generic_200mhz_v1.db"
set MAP_DB     "$CHECKPOINT_DIR/affine_fulltop_axi_dualclk_after_map_200mhz_v1.db"
set OPT_DB     "$CHECKPOINT_DIR/affine_fulltop_axi_dualclk_after_opt_200mhz_v1.db"

set FINAL_NETLIST "$OUTPUT_DIR/msm_axi_affine_multiwindow_top_dualclk_v1_mapped_200mhz_v1.v"
set FINAL_SDC     "$OUTPUT_DIR/msm_axi_affine_multiwindow_top_dualclk_v1_mapped_200mhz_v1.sdc"
set FINAL_DB      "$OUTPUT_DIR/msm_axi_affine_multiwindow_top_dualclk_v1_mapped_200mhz_v1.db"

file mkdir $REPORT_DIR
file mkdir $OUTPUT_DIR
file mkdir $CHECKPOINT_DIR

cd $RTL_ROOT

# --------------------------------------------------------------------
# Effort / HDL settings
# --------------------------------------------------------------------
set_db syn_generic_effort medium
set_db syn_map_effort     medium
set_db syn_opt_effort     medium

# Keep same loop setting used by the successful elaboration flow.
set_db hdl_max_loop_limit 20000

# --------------------------------------------------------------------
# Libraries
# --------------------------------------------------------------------
set STD_CELL_LIB "/tech/tsmc/65LP/dig_libs/CORE/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn65lp_200a/tcbn65lptc.lib"
set SRAM64_LIB "$RTL_ROOT/rtl/mem/sram_macro/s8192x64/liberty/s8192x64_tt_1p20v_1p20v_25c.lib"
set SRAM16_LIB "$RTL_ROOT/rtl/mem/sram_macro/s8192x16/liberty/s8192x16_tt_1p20v_1p20v_25c.lib"

foreach lib_file [list $STD_CELL_LIB $SRAM64_LIB $SRAM16_LIB] {
    if {![file exists $lib_file]} {
        puts "ERROR: Missing Liberty file: $lib_file"
        error "Technology library file not found"
    }
}

if {![file exists $INPUT_DB]} {
    puts "ERROR: Missing input checkpoint: $INPUT_DB"
    error "Input checkpoint not found"
}

read_libs [list $STD_CELL_LIB $SRAM64_LIB $SRAM16_LIB]
read_db $INPUT_DB

if {[llength [get_designs $TOP_MODULE]] == 0} {
    puts "ERROR: Expected top design was not found after read_db: $TOP_MODULE"
    error "Wrong or incomplete checkpoint"
}

current_design $TOP_MODULE

# --------------------------------------------------------------------
# Restored-design sanity reports
# --------------------------------------------------------------------
check_design -all      > $REPORT_DIR/check_design_restored.rpt
report_clocks          > $REPORT_DIR/clocks_restored.rpt
check_timing           > $REPORT_DIR/check_timing_restored.rpt
report_timing -lint    > $REPORT_DIR/timing_lint_restored.rpt

# --------------------------------------------------------------------
# syn_generic
# --------------------------------------------------------------------
puts "============================================================"
puts "Starting syn_generic for $RUN_NAME"
puts "============================================================"

syn_generic

write_db $GENERIC_DB

check_design -all > $REPORT_DIR/check_design_after_generic.rpt
report_area       > $REPORT_DIR/area_after_generic.rpt
report_timing     > $REPORT_DIR/timing_after_generic.rpt
report_qor        > $REPORT_DIR/qor_after_generic.rpt

puts "============================================================"
puts "syn_generic completed"
puts "Wrote DB: $GENERIC_DB"
puts "============================================================"

# --------------------------------------------------------------------
# syn_map
# --------------------------------------------------------------------
puts "============================================================"
puts "Starting syn_map for $RUN_NAME"
puts "============================================================"

syn_map

write_db $MAP_DB

check_design -all > $REPORT_DIR/check_design_after_map.rpt
report_area       > $REPORT_DIR/area_after_map.rpt
report_area -depth 100 -show_module_names -show_full_names > $REPORT_DIR/area_hierarchy_after_map.rpt
report_timing     > $REPORT_DIR/timing_after_map.rpt
report_qor        > $REPORT_DIR/qor_after_map.rpt

puts "============================================================"
puts "syn_map completed"
puts "Wrote DB: $MAP_DB"
puts "============================================================"

# --------------------------------------------------------------------
# syn_opt
# --------------------------------------------------------------------
puts "============================================================"
puts "Starting syn_opt for $RUN_NAME"
puts "============================================================"

syn_opt

# Save immediately after syn_opt, before any heavy/optional report.
write_db $OPT_DB

check_design -all > $REPORT_DIR/check_design_after_opt.rpt
report_area       > $REPORT_DIR/area_after_opt.rpt
report_area -depth 100 -show_module_names -show_full_names > $REPORT_DIR/area_hierarchy_after_opt.rpt
report_timing     > $REPORT_DIR/timing_after_opt.rpt
report_qor        > $REPORT_DIR/qor_after_opt.rpt

puts "============================================================"
puts "syn_opt completed"
puts "Wrote DB: $OPT_DB"
puts "============================================================"

# --------------------------------------------------------------------
# Final outputs
# --------------------------------------------------------------------
write_hdl > $FINAL_NETLIST
write_sdc > $FINAL_SDC
write_db  $FINAL_DB

puts "============================================================"
puts "Affine full-top AXI dual-clock 200MHz synthesis completed"
puts "TOP:          $TOP_MODULE"
puts "GENERIC_DB:   $GENERIC_DB"
puts "MAP_DB:       $MAP_DB"
puts "OPT_DB:       $OPT_DB"
puts "NETLIST:      $FINAL_NETLIST"
puts "SDC:          $FINAL_SDC"
puts "FINAL_DB:     $FINAL_DB"
puts "REPORT_DIR:   $REPORT_DIR"
puts "============================================================"