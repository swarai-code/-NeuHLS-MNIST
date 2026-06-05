# run_hls_full.tcl — Vitis HLS synthesis for the full hybrid MLP.
#   vitis_hls -f run_hls_full.tcl
# Requires fc1_params.h and fc2_params.h (run export_weights.py first).

open_project hls_mlp_full
set_top mlp_top
add_files mlp_top.cpp
add_files symbolic_head.cpp
add_files -tb mlp_tb.cpp

open_solution "solution1" -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default   ;# 100 MHz

csim_design
csynth_design
# cosim_design
export_design -format ip_catalog

close_project
exit
