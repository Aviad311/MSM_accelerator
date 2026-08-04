
# run_elaborate_affine_fulltop_axi400_msm200_v1.tcl
# Affine full-top AXI dual-clock elaboration flow.
#
# Clock intent:
#   axi_clk = 400MHz => 2.500ns
#   msm_clk = 200MHz => 5.000ns
#
# CDC intent:
#   axi_clk and msm_clk are asynchronous clock domains.
#   Cross-domain timing is cut explicitly with set_clock_groups and set_false_path.
#
# TOP:
#   msm_axi_affine_multiwindow_top_dualclk_v1
#
# Output checkpoint:
#   synthesis/checkpoints/affine_fulltop_axi400_msm200_v1/
#     affine_fulltop_axi400_msm200_elaborated_constrained_v1.db

set TOP_MODULE msm_axi_affine_multiwindow_top_dualclk_v1

set SCRIPT_DIR [file dirname [file normalize [info script]]]
set SYNTH_DIR  [file normalize "$SCRIPT_DIR/.."]
set RTL_ROOT   [file normalize "$SYNTH_DIR/.."]

set RUN_NAME       "affine_fulltop_axi400_msm200_elaborate_v1"
set REPORT_DIR     "$SYNTH_DIR/reports/$RUN_NAME"
set CHECKPOINT_DIR "$SYNTH_DIR/checkpoints/affine_fulltop_axi400_msm200_v1"
set DB_OUT         "$CHECKPOINT_DIR/affine_fulltop_axi400_msm200_elaborated_constrained_v1.db"

file mkdir $REPORT_DIR
file mkdir $CHECKPOINT_DIR

cd $RTL_ROOT

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

read_libs [list $STD_CELL_LIB $SRAM64_LIB $SRAM16_LIB]

# Required for large generate/for loops in bucket scheduler.
set_db hdl_max_loop_limit 20000

# --------------------------------------------------------------------
# RTL
# --------------------------------------------------------------------
source synthesis/scripts/rtl_files_affine_fulltop_axi_dualclk_200mhz_v1.tcl

# --------------------------------------------------------------------
# Elaborate
# --------------------------------------------------------------------
elaborate $TOP_MODULE

if {[llength [get_designs $TOP_MODULE]] == 0} {
    puts "ERROR: Expected top design was not found after elaborate: $TOP_MODULE"
    error "Top design was not elaborated"
}

current_design $TOP_MODULE

# --------------------------------------------------------------------
# Basic design check before constraints
# --------------------------------------------------------------------
check_design -all > $REPORT_DIR/check_design_elab.rpt

# --------------------------------------------------------------------
# Clocks / Constraints
# --------------------------------------------------------------------
set AXI_CLK_PORT axi_clk
set MSM_CLK_PORT msm_clk

set AXI_CLK_PERIOD_NS 2.500
set MSM_CLK_PERIOD_NS 5.000

if {[llength [get_ports $AXI_CLK_PORT]] == 0} {
    puts "ERROR: Missing expected AXI clock port: $AXI_CLK_PORT"
    puts "Available ports:"
    puts [get_object_name [get_ports *]]
    error "Missing AXI clock port"
}

if {[llength [get_ports $MSM_CLK_PORT]] == 0} {
    puts "ERROR: Missing expected MSM clock port: $MSM_CLK_PORT"
    puts "Available ports:"
    puts [get_object_name [get_ports *]]
    error "Missing MSM clock port"
}

create_clock -name axi_clk -period $AXI_CLK_PERIOD_NS [get_ports $AXI_CLK_PORT]
create_clock -name msm_clk -period $MSM_CLK_PERIOD_NS [get_ports $MSM_CLK_PORT]

# Declare the two domains asynchronous.
set_clock_groups -asynchronous \
  -group [get_clocks axi_clk] \
  -group [get_clocks msm_clk]

# Explicitly cut cross-clock timing as well, to keep reports unambiguous.
set_false_path -from [get_clocks axi_clk] -to [get_clocks msm_clk]
set_false_path -from [get_clocks msm_clk] -to [get_clocks axi_clk]

# Basic IO delays relative to AXI clock.
set_input_delay  0.5 -clock axi_clk [remove_from_collection [all_inputs] [get_ports [list $AXI_CLK_PORT $MSM_CLK_PORT]]]
set_output_delay 0.5 -clock axi_clk [all_outputs]

# --------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------
report_clocks > $REPORT_DIR/clocks_elab.rpt
check_timing  > $REPORT_DIR/check_timing_elab.rpt
report_timing -lint > $REPORT_DIR/timing_lint_elab.rpt
check_design -all > $REPORT_DIR/check_design_elab_after_constraints.rpt

# Cross-clock sanity reports. These should not show real analyzed timing paths.
report_timing -from [get_clocks axi_clk] -to [get_clocks msm_clk] -max_paths 5 > $REPORT_DIR/timing_axi_to_msm_cut_check.rpt
report_timing -from [get_clocks msm_clk] -to [get_clocks axi_clk] -max_paths 5 > $REPORT_DIR/timing_msm_to_axi_cut_check.rpt

# --------------------------------------------------------------------
# Checkpoint
# --------------------------------------------------------------------
write_db $DB_OUT

puts "============================================================"
puts "Affine full-top AXI400/MSM200 elaboration completed"
puts "TOP:        $TOP_MODULE"
puts "AXI clock:  $AXI_CLK_PERIOD_NS ns / 400MHz"
puts "MSM clock:  $MSM_CLK_PERIOD_NS ns / 200MHz"
puts "DB_OUT:     $DB_OUT"
puts "REPORTS:    $REPORT_DIR"
puts "============================================================"