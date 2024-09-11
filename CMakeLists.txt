cmake_minimum_required(VERSION 3.19)

project(HelloDML VERSION 0.0.1 LANGUAGES CXX)

# To avoid runtime dependency on MSVC runtime DLLs
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded$<$<CONFIG:Debug>:Debug>)

include(FetchContent)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

set(TARGET_ARCH ${CMAKE_CXX_COMPILER_ARCHITECTURE_ID})
if(TARGET_ARCH STREQUAL AMD64)
    set(TARGET_ARCH x64)
endif()

# -----------------------------------------------------------------------------
# wil - Windows-related helper types/macros
# -----------------------------------------------------------------------------
FetchContent_Declare(
    wil
    URL https://github.com/microsoft/wil/archive/refs/tags/v1.0.240803.1.zip
    URL_HASH SHA256=353D2D7F2ACEA5642689A1BA85213C1AC6816457038B54AC02158B893E0F389F
)

FetchContent_GetProperties(wil)
if(NOT wil_POPULATED)
    FetchContent_Populate(wil)
endif()

add_library(wil INTERFACE)
target_include_directories(wil INTERFACE "${wil_SOURCE_DIR}/include")

# -----------------------------------------------------------------------------
# directx-headers - to get the latest DXCore header with GENERIC_ML GUID
# -----------------------------------------------------------------------------
FetchContent_Declare(
    dxheaders
    GIT_REPOSITORY https://github.com/microsoft/DirectX-Headers
    GIT_TAG de28d93dfa9ebf3e473127c1c657e1920a5345ee # v1.613.1
)

FetchContent_MakeAvailable(dxheaders)

set(ort_bin_dir "${ort_SOURCE_DIR}/runtimes/win-${TARGET_ARCH}/native")

add_library(dxheaders INTERFACE)
target_include_directories(dxheaders INTERFACE ${dxheaders_SOURCE_DIR}/include/directx)
target_link_libraries(dxheaders INTERFACE Microsoft::DirectX-Guids)

# -----------------------------------------------------------------------------
# directml
# -----------------------------------------------------------------------------
FetchContent_Declare(
    dml
    URL https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.15.2
    URL_HASH SHA256=9F07482559087088A4DBA4AE76EEEEE1FAD3F7077A92CCFBDB439C6BC2964C09
)

FetchContent_MakeAvailable(dml)

set(dml_bin_dir "${dml_SOURCE_DIR}/bin/${TARGET_ARCH}-win")

add_library(dml INTERFACE)
target_include_directories(dml INTERFACE "${dml_SOURCE_DIR}/include")
target_link_libraries(dml INTERFACE "${dml_bin_dir}/directml.lib")

# -----------------------------------------------------------------------------
# DirectMLX Header
# -----------------------------------------------------------------------------
FetchContent_Declare(
    directmlx
    URL https://raw.githubusercontent.com/microsoft/DirectML/91cc5e5e823d582938c3407ec65e8e4a96e020a1/Libraries/DirectMLX.h
    DOWNLOAD_NO_EXTRACT 1
)
FetchContent_MakeAvailable(directmlx)
target_include_directories(dml INTERFACE ${directmlx_SOURCE_DIR})

# -----------------------------------------------------------------------------
# main sample source
# -----------------------------------------------------------------------------

add_executable(hello_dml hello_dml.cpp DirectMLProcessor.cpp ) 
target_link_libraries(hello_dml PRIVATE wil dml d3d12 dxcore dxheaders)
target_compile_features(hello_dml PRIVATE cxx_std_17)