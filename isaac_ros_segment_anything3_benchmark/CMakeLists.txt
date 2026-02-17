# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_segment_anything3_benchmark)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Install Python scripts
install(PROGRAMS
  scripts/isaac_ros_segment_anything3_graph.py
  scripts/sam3_monitor_node.py
  scripts/generate_test_rosbag.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

# Install results and datasets directories (with .gitkeep placeholder)
install(FILES results/.gitkeep DESTINATION share/${PROJECT_NAME}/results)
install(FILES datasets/.gitkeep DESTINATION share/${PROJECT_NAME}/datasets)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()
