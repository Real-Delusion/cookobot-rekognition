cmake_minimum_required(VERSION 2.8.3)
project(prediction_action)

find_package(catkin REQUIRED COMPONENTS
  rospy
  actionlib_msgs
)

## Generate actions in the 'action' folder
add_action_files(
  DIRECTORY action
  FILES Prediction.action
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs  # Or other packages containing msgs
  actionlib_msgs
)


###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS rospy
  actionlib_msgs
)