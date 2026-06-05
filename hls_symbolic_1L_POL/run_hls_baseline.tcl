# run_hls_baseline.tcl — Vitis HLS synthesis for the dense BL-MLP baseline.
#   vitis_hls -f run_hls_baseline.tcl
# Requires baseline_params.h (run export_baseline.py on baseline.pt first).

open_project hls_mlp_baseline
set_top mlp_baseline_top
add_files mlp_baseline.cpp
add_files -tb mlp_baseline_tb.cpp

open_solution "solution1" -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default   ;# 100 MHz

csim_design
csynth_design
export_design -format ip_catalog

close_project
exit
