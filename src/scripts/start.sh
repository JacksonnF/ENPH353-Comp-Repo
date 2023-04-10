#!/bin/bash

pkill -f ros
pkill -f gazebo
pkill -f gz

source ~/ros_ws/devel/setup.bash
(cd ~/ros_ws/src/2022_competition/enph353/enph353_utils/scripts && ./run_sim.sh -vpg)


 