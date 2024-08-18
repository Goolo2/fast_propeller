# 定义最低的 CMake 版本
cmake_minimum_required(VERSION 3.10)

# 定义工程名称
project(MyProject)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)

# 设置项目的可执行文件输出路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 包含头文件目录
include_directories(include)

# 链接库目录
link_directories(lib)

find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})

# 添加可执行文件
add_executable(readdata src/read_data.cpp)

# 链接 cnpy 库
target_link_libraries(readdata cnpy ${ZLIB_LIBRARIES})