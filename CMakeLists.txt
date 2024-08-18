# # 定义最低的 CMake 版本
# cmake_minimum_required(VERSION 3.10)

# # 定义工程名称
# project(MyProject)

# # 设置 C++ 标准
# set(CMAKE_CXX_STANDARD 14)

# # 设置项目的可执行文件输出路径
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# # 包含头文件目录
# include_directories(include)

# # 链接库目录
# link_directories(lib)

# find_package(ZLIB REQUIRED)
# include_directories(${ZLIB_INCLUDE_DIRS})

# # 添加可执行文件
# # add_executable(readdata src/read_data.cpp)
# add_executable(estimate src/estimate2.cpp)

# # 链接 cnpy 库
# target_link_libraries(estimate cnpy ${ZLIB_LIBRARIES})


cmake_minimum_required(VERSION 3.10)
project(FastPropeller)

# 设置所需的C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 找到Ceres库
find_package(Ceres REQUIRED)

# 找到Eigen库
find_package(Eigen3 REQUIRED)

find_package(OpenCV REQUIRED)
# 设置库和头文件的路径
include_directories(${CERES_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加源码文件
add_executable(estimate src/estimate_cost.cpp)

# 链接需要的库
target_link_libraries(estimate ${CERES_LIBRARIES} ${EIGEN3_LIBRARIES} ${OpenCV_LIBS})