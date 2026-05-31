open_project -reset baseline_mlp
set_top baseline_mlp
add_files baseline_mlp.cpp
add_files -tb {}
open_solution -reset sol1
set_part {xc7a35tcpg236-1}
create_clock -period 10 -name default
config_dataflow -default_channel fifo -fifo_depth 2
csynth_design
export_design -rtl verilog -format ip_catalog
close_project
