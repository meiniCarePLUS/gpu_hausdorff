#	State Key Lab of CAD&CG Zhejiang Unv.
#
#	Author: 
#          Yicun Zheng (3130104113@zju.edu.cn)
#          Haoran Sun (hrsun@zju.edu.cn)
#          Jin Huang (hj@cad.zju.edu.cn)
#
#	Copyright (c) 2004-2021 <Jin Huang>
#	All rights reserved.
#
#	Licensed under the MIT License.

add_executable(hausdorff main.cpp)
target_link_libraries(hausdorff zycgeo_hausdorff zjucad::matrix meshio cxxopts zycgeo_gpu_bounds zycgeo_lbvh cudart)
target_include_directories(hausdorff PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(test_lbvh test_lbvh.cu)
target_link_libraries(test_lbvh zycgeo_lbvh cudart)
target_include_directories(test_lbvh PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(test_gpu_hausdorff test_gpu_hausdorff.cu)
target_link_libraries(test_gpu_hausdorff zycgeo_gpu_bounds zycgeo_lbvh cudart)
target_include_directories(test_gpu_hausdorff PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(test_initial_bounds test_initial_bounds.cu)
target_link_libraries(test_initial_bounds zycgeo_gpu_bounds zycgeo_lbvh cudart)
target_include_directories(test_initial_bounds PRIVATE ${CMAKE_SOURCE_DIR}/src)