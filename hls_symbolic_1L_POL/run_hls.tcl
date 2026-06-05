# run_hls.tcl — Vitis HLS synthesis for the 1L_POL symbolic head
# Target matches the paper: XC7Z020 (Zynq-7020), 100 MHz.
#   vitis_hls -f run_hls.tcl

open_project hls_symbolic_head_1L_POL
set_top symbolic_head
add_files symbolic_head.cpp
add_files -tb symbolic_head_tb.cpp

open_solution "solution1" -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default   ;# 10 ns = 100 MHz

csim_design          ;# C simulation (functional check)
csynth_design        ;# synthesis -> LUT/FF/DSP/latency estimates
# cosim_design       ;# uncomment for RTL co-sim (slow)
export_design -format ip_catalog

close_project
exit
