open_project -reset baseline_mlp
set_top baseline_mlp
add_files baseline_mlp.cpp
open_solution -reset sol1
set_part {xc7a35tcpg236-1}
create_clock -period 10 -name default
config_dataflow -default_channel fifo -fifo_depth 2
csynth_design
close_project
