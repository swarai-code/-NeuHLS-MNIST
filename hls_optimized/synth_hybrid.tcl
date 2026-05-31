open_project -reset hybrid_1L_SRL
set_top hybrid_1L_SRL
add_files hybrid_1L_SRL.cpp
add_files -tb {}
open_solution -reset sol1
set_part {xc7a35tcpg236-1}
create_clock -period 10 -name default
config_dataflow -default_channel fifo -fifo_depth 2
csynth_design
export_design -rtl verilog -format ip_catalog
close_project
