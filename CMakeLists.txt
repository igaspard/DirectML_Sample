cmake_minimum_required(VERSION 3.19)

project(HelloDML VERSION 0.0.1 LANGUAGES CXX)

include(FetchContent)
include(cmake/helper_platform.cmake)

include(cmake/wil.cmake)
add_wil_target(wil CACHE_PREFIX DXD)

include(cmake/d3d12.cmake)
add_d3d12_target(d3d12 CACHE_PREFIX DXD)

include(cmake/directml.cmake)
add_directml_target(directml CACHE_PREFIX DXD)

set(CMAKE_CXX_STANDARD 17)

get_target_property(directml_config directml DX_COMPONENT_CONFIG)
get_target_property(d3d12_config d3d12 DX_COMPONENT_CONFIG)

add_executable(hello_dml src/DirectMLProcessor.cpp src/main.cpp) 

include_directories(include)

target_link_libraries(hello_dml wil d3d12 directml dxcore d3d12.lib directml.lib)
