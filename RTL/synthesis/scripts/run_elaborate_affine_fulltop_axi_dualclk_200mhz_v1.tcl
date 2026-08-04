# run_elaborate_affine_fulltop_axi_dualclk_200mhz_v1.tcl
# Affine full-top AXI dual-clock 200MHz elaboration flow.
#
# Purpose:
#   Elaborate the module used by most final affine End-to-End TBs:
#     msm_axi_affine_multiwindow_top_dualclk_v1
#
# Chain:
#   AXI/frontend -> dual-clock FIFO/source -> multiwindow controller -> 8-lane backend
#
# Output checkpoint:
#   synthesis/checkpoints/affine_fulltop_axi_dualclk_200mhz_v1/
#     affine_fulltop_axi_dualclk_elaborated_constrained_200mhz_v1.db
#
# This script only elaborates and writes a constrained DB.
# It does not run syn_generic/syn_map/syn_opt.

set TOP_MODULE msm_axi_affine_multiwindow_top_dualclk_v1

set SCRIPT_DIR [file dirname [file normalize [info script]]]
set SYNTH_DIR  [file normalize "$SCRIPT_DIR/.."]
set RTL_ROOT   [file normalize "$SYNTH_DIR/.."]

set REPORT_DIR     "$SYNTH_DIR/reports/affine_fulltop_axi_dualclk_200mhz_elaborate_v1"
set CHECKPOINT_DIR "$SYNTH_DIR/checkpoints/affine_fulltop_axi_dualclk_200mhz_v1"
set DB_OUT         "$CHECKPOINT_DIR/affine_fulltop_axi_dualclk_elaborated_constrained_200mhz_v1.db"

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
# Basic design checks before constraints
# --------------------------------------------------------------------
check_design -all > $REPORT_DIR/check_design_elab.rpt

# --------------------------------------------------------------------
# Clocks / Constraints
# Expected top clock ports:
#   axi_clk
#   msm_clk
#
# If the top module uses different names, update these variables.
# --------------------------------------------------------------------
set AXI_CLK_PORT axi_clk
set MSM_CLK_PORT msm_clk
set CLOCK_PERIOD_NS 5.000

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

create_clock -name axi_clk -period $CLOCK_PERIOD_NS [get_ports $AXI_CLK_PORT]
create_clock -name msm_clk -period $CLOCK_PERIOD_NS [get_ports $MSM_CLK_PORT]

set_clock_groups -asynchronous \
  -group [get_clocks axi_clk] \
  -group [get_clocks msm_clk]

# Basic IO delays relative to axi_clk.
set_input_delay  0.5 -clock axi_clk [remove_from_collection [all_inputs] [get_ports [list $AXI_CLK_PORT $MSM_CLK_PORT]]]
set_output_delay 0.5 -clock axi_clk [all_outputs]

# --------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------
report_clocks > $REPORT_DIR/clocks_elab.rpt
check_timing  > $REPORT_DIR/check_timing_elab.rpt
report_timing -lint > $REPORT_DIR/timing_lint_elab.rpt
check_design -all > $REPORT_DIR/check_design_elab_after_constraints.rpt

# --------------------------------------------------------------------
# Checkpoint
# --------------------------------------------------------------------
write_db $DB_OUT

puts "============================================================"
puts "Affine full-top AXI dual-clock 200MHz elaboration completed"
puts "TOP:      $TOP_MODULE"
puts "DB_OUT:   $DB_OUT"
puts "REPORTS:  $REPORT_DIR"
puts "============================================================"