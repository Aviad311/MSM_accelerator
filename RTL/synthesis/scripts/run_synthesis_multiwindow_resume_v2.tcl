# ============================================================
# MSM Accelerator
# Cadence Genus - Resume 16-Window Synthesis from Elaborated DB
# ============================================================

set TOP_MODULE msm_multiwindow_controller_v1

set SCRIPT_DIR [file dirname [file normalize [info script]]]
set SYNTH_DIR  [file normalize "$SCRIPT_DIR/.."]
set RTL_ROOT   [file normalize "$SYNTH_DIR/.."]

set REPORT_DIR "$SYNTH_DIR/reports/multiwindow_synthesis_resume_v2"
set OUTPUT_DIR "$SYNTH_DIR/outputs/multiwindow_synthesis_resume_v2"
set CHECKPOINT_DIR "$SYNTH_DIR/checkpoints/multiwindow_synthesis_v1"

set INPUT_DB   "$CHECKPOINT_DIR/multiwindow_elaborated_constrained_v1.db"
set GENERIC_DB "$CHECKPOINT_DIR/multiwindow_after_generic_v2.db"
set MAP_DB     "$CHECKPOINT_DIR/multiwindow_after_map_v2.db"
set OPT_DB     "$CHECKPOINT_DIR/multiwindow_after_opt_v2.db"

file mkdir $REPORT_DIR
file mkdir $OUTPUT_DIR
file mkdir $CHECKPOINT_DIR

puts ""
puts "============================================================"
puts "MSM 16-Window Synthesis Resume"
puts "============================================================"
puts "TOP_MODULE     = $TOP_MODULE"
puts "INPUT_DB       = $INPUT_DB"
puts "REPORT_DIR     = $REPORT_DIR"
puts "OUTPUT_DIR     = $OUTPUT_DIR"
puts "CHECKPOINT_DIR = $CHECKPOINT_DIR"
puts "============================================================"

cd $RTL_ROOT

set_db syn_generic_effort medium
set_db syn_map_effort medium
set_db syn_opt_effort medium

set STD_CELL_LIB \
    "/tech/tsmc/65LP/dig_libs/CORE/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn65lp_200a/tcbn65lptc.lib"
set SRAM64_LIB \
    "$RTL_ROOT/rtl/mem/sram_macro/s8192x64/liberty/s8192x64_tt_1p20v_1p20v_25c.lib"
set SRAM16_LIB \
    "$RTL_ROOT/rtl/mem/sram_macro/s8192x16/liberty/s8192x16_tt_1p20v_1p20v_25c.lib"
set LIB_FILES [list $STD_CELL_LIB $SRAM64_LIB $SRAM16_LIB]

foreach lib_file $LIB_FILES {
    if {![file exists $lib_file]} {
        puts "ERROR: Missing Liberty file: $lib_file"
        error "Technology library file not found"
    }
}

if {![file exists $INPUT_DB]} {
    puts "ERROR: Missing checkpoint: $INPUT_DB"
    error "Input checkpoint not found"
}

puts "Loading technology libraries..."
read_libs $LIB_FILES

puts "Restoring elaborated constrained checkpoint..."
read_db $INPUT_DB

if {[llength [get_designs $TOP_MODULE]] == 0} {
    puts "ERROR: Top design '$TOP_MODULE' not found after read_db"
    error "Unexpected checkpoint contents"
}

puts "Restored design: [get_object_name [current_design]]"

report_clocks > $REPORT_DIR/clocks_restored.rpt
check_design -all > $REPORT_DIR/check_design_restored.rpt
report_timing -lint > $REPORT_DIR/timing_lint_restored.rpt
report_messages > $REPORT_DIR/messages_restored.rpt

puts ""
puts "============================================================"
puts "Running syn_generic"
puts "============================================================"
syn_generic
puts "syn_generic completed"

# Save before reports so a report error cannot destroy the expensive stage.
write_db $GENERIC_DB
puts "Saved generic checkpoint: $GENERIC_DB"

report_area > $REPORT_DIR/area_after_generic.rpt
report_area -depth 100 -show_module_names -show_full_names \
    > $REPORT_DIR/area_hierarchy_after_generic.rpt
report_timing > $REPORT_DIR/timing_after_generic.rpt
report_qor > $REPORT_DIR/qor_after_generic.rpt
report_messages > $REPORT_DIR/messages_after_generic.rpt

puts ""
puts "============================================================"
puts "Running syn_map"
puts "============================================================"
syn_map
puts "syn_map completed"

write_db $MAP_DB
puts "Saved mapped checkpoint: $MAP_DB"

report_area > $REPORT_DIR/area_after_map.rpt
report_area -depth 100 -show_module_names -show_full_names \
    > $REPORT_DIR/area_hierarchy_after_map.rpt
report_timing > $REPORT_DIR/timing_after_map.rpt
report_timing -max_paths 50 > $REPORT_DIR/timing_top50_after_map.rpt
report_qor > $REPORT_DIR/qor_after_map.rpt
report_gates > $REPORT_DIR/gates_after_map.rpt
report_messages > $REPORT_DIR/messages_after_map.rpt

puts ""
puts "============================================================"
puts "Running syn_opt"
puts "============================================================"
syn_opt
puts "syn_opt completed"

write_db $OPT_DB
puts "Saved optimized checkpoint: $OPT_DB"

check_design -all > $REPORT_DIR/check_design_post_synth.rpt
report_area > $REPORT_DIR/area_final.rpt
report_area -depth 100 -show_module_names -show_full_names \
    > $REPORT_DIR/area_hierarchy_final.rpt
report_timing > $REPORT_DIR/timing_final.rpt
report_timing -max_paths 50 > $REPORT_DIR/timing_top50_final.rpt
report_qor > $REPORT_DIR/qor_final.rpt
report_power > $REPORT_DIR/power_final.rpt
report_gates > $REPORT_DIR/gates_final.rpt
report_hierarchy > $REPORT_DIR/hierarchy_final.rpt
report_design_rules > $REPORT_DIR/design_rules_final.rpt
report_messages > $REPORT_DIR/messages_final.rpt

write_hdl > $OUTPUT_DIR/${TOP_MODULE}_mapped_v2.v
write_sdc > $OUTPUT_DIR/${TOP_MODULE}_mapped_v2.sdc
write_db $OUTPUT_DIR/${TOP_MODULE}_mapped_v2.db

puts ""
puts "============================================================"
puts "MSM 16-window synthesis completed successfully"
puts "============================================================"
puts "Generic checkpoint: $GENERIC_DB"
puts "Mapped checkpoint : $MAP_DB"
puts "Optimized checkpoint: $OPT_DB"
puts "Final DB: $OUTPUT_DIR/${TOP_MODULE}_mapped_v2.db"
puts "============================================================"

exit