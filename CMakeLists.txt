cmake_minimum_required(VERSION 3.21)

project(llama_cpp)

option(LLAMA_BUILD "Build llama.cpp shared library and install alongside python package" OFF)
option(LLAMA_USE_EXTERNAL "Use externally built llama.cpp library" ON)
option(MTMD_BUILD "Build mtmd shared library and install alongside python package" OFF)

# External llama.cpp configuration
set(LLAMA_CPP_LIB_PATH "" CACHE STRING "Path to external llama.cpp library directory")
set(LLAMA_CPP_INCLUDE_PATH "" CACHE STRING "Path to external llama.cpp include directory")

function(llama_cpp_python_install_target target)
    if(NOT TARGET ${target})
        return()
    endif()

    install(
        TARGETS ${target}
        LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
        RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
        ARCHIVE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
        FRAMEWORK DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
        RESOURCE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
    )
    install(
        TARGETS ${target}
        LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib
        RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib
        ARCHIVE DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib
        FRAMEWORK DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib
        RESOURCE DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib
    )
    set_target_properties(${target} PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
    if(UNIX)
        if(APPLE)
            set_target_properties(${target} PROPERTIES
                INSTALL_RPATH "@loader_path"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
        else()
            set_target_properties(${target} PROPERTIES
                INSTALL_RPATH "$ORIGIN"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
        endif()
    endif()
endfunction()

function(llama_cpp_python_install_external_library lib_path)
    get_filename_component(lib_name ${lib_path} NAME)
    
    install(
        FILES ${lib_path}
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
    )
    install(
        FILES ${lib_path}
        DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib
    )
    
    message(STATUS "Will install external library: ${lib_path}")
endfunction()

if (LLAMA_USE_EXTERNAL)
    message(STATUS "Using external llama.cpp library")
    
    # Find external llama.cpp library
    if(LLAMA_CPP_LIB_PATH)
        set(LLAMA_LIBRARY_SEARCH_PATHS ${LLAMA_CPP_LIB_PATH})
    else()
        # Default search paths for common installation locations
        set(LLAMA_LIBRARY_SEARCH_PATHS
            /usr/local/lib
            /usr/lib
            /opt/homebrew/lib
            /opt/local/lib
            ${CMAKE_PREFIX_PATH}/lib
        )
    endif()

    # Find the llama library
    find_library(LLAMA_LIBRARY
        NAMES llama libllama
        PATHS ${LLAMA_LIBRARY_SEARCH_PATHS}
        NO_DEFAULT_PATH
    )

    if(NOT LLAMA_LIBRARY)
        find_library(LLAMA_LIBRARY
            NAMES llama libllama
        )
    endif()

    if(NOT LLAMA_LIBRARY)
        message(FATAL_ERROR "External llama.cpp library not found. Please specify LLAMA_CPP_LIB_PATH or install llama.cpp in a standard location.")
    endif()

    message(STATUS "Found llama library: ${LLAMA_LIBRARY}")

    # Find GGML library (may be separate or combined with llama)
    find_library(GGML_LIBRARY
        NAMES ggml libggml
        PATHS ${LLAMA_LIBRARY_SEARCH_PATHS}
        NO_DEFAULT_PATH
    )

    if(NOT GGML_LIBRARY)
        find_library(GGML_LIBRARY
            NAMES ggml libggml
        )
    endif()

    if(GGML_LIBRARY)
        message(STATUS "Found ggml library: ${GGML_LIBRARY}")
        llama_cpp_python_install_external_library(${GGML_LIBRARY})
    else()
        message(STATUS "GGML library not found separately, assuming it's included in llama library")
    endif()

    # Install the external libraries
    llama_cpp_python_install_external_library(${LLAMA_LIBRARY})

    # Find and install backend libraries if they exist separately
    set(BACKEND_LIBRARIES
        ggml-base ggml-amx ggml-blas ggml-cann ggml-cpu ggml-cuda 
        ggml-hip ggml-kompute ggml-metal ggml-musa ggml-opencl 
        ggml-rpc ggml-sycl ggml-vulkan ggml-webgpu
    )

    foreach(backend ${BACKEND_LIBRARIES})
        find_library(${backend}_LIBRARY
            NAMES ${backend} lib${backend}
            PATHS ${LLAMA_LIBRARY_SEARCH_PATHS}
            NO_DEFAULT_PATH
        )
        
        if(${backend}_LIBRARY)
            message(STATUS "Found ${backend} library: ${${backend}_LIBRARY}")
            llama_cpp_python_install_external_library(${${backend}_LIBRARY})
        endif()
    endforeach()

elseif (LLAMA_BUILD)
    message(STATUS "Building llama.cpp from vendored source")
    
    set(BUILD_SHARED_LIBS "On")

    set(CMAKE_SKIP_BUILD_RPATH FALSE)

    # When building, don't use the install RPATH already
    # (but later on when installing)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
 
    # Add the automatically determined parts of the RPATH
    # which point to directories outside the build tree to the install RPATH
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_SKIP_RPATH FALSE)

    # Enable building of the common library
    set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build llama.cpp common library" FORCE)

    # Disable building curl support
    set(LLAMA_CURL OFF CACHE BOOL "llama.cpp: enable curl" FORCE)

    # Architecture detection and settings for Apple platforms
    if (APPLE)
        # Get the target architecture
        execute_process(
            COMMAND uname -m
            OUTPUT_VARIABLE HOST_ARCH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        # If CMAKE_OSX_ARCHITECTURES is not set, use the host architecture
        if(NOT CMAKE_OSX_ARCHITECTURES)
            set(CMAKE_OSX_ARCHITECTURES ${HOST_ARCH} CACHE STRING "Build architecture for macOS" FORCE)
        endif()

        message(STATUS "Host architecture: ${HOST_ARCH}")
        message(STATUS "Target architecture: ${CMAKE_OSX_ARCHITECTURES}")

        # Configure based on target architecture
        if(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
            # Intel Mac settings
            set(GGML_AVX "OFF" CACHE BOOL "ggml: enable AVX" FORCE)
            set(GGML_AVX2 "OFF" CACHE BOOL "ggml: enable AVX2" FORCE)
            set(GGML_FMA "OFF" CACHE BOOL "ggml: enable FMA" FORCE)
            set(GGML_F16C "OFF" CACHE BOOL "ggml: enable F16C" FORCE)
        endif()

        # Metal settings (enable for both architectures)
        set(GGML_METAL "ON" CACHE BOOL "ggml: enable Metal" FORCE)
        set(GGML_METAL_EMBED_LIBRARY "ON" CACHE BOOL "ggml: embed metal library" FORCE)
    endif()

    add_subdirectory(vendor/llama.cpp)

    if (WIN32)
        if (TARGET llama)
            set_target_properties(llama PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
        endif()
    endif()

    llama_cpp_python_install_target(llama)
    llama_cpp_python_install_target(ggml)

    llama_cpp_python_install_target(ggml-base)

    llama_cpp_python_install_target(ggml-amx)
    llama_cpp_python_install_target(ggml-blas)
    llama_cpp_python_install_target(ggml-cann)
    llama_cpp_python_install_target(ggml-cpu)
    llama_cpp_python_install_target(ggml-cuda)
    llama_cpp_python_install_target(ggml-hip)
    llama_cpp_python_install_target(ggml-kompute)
    llama_cpp_python_install_target(ggml-metal)
    llama_cpp_python_install_target(ggml-musa)
    llama_cpp_python_install_target(ggml-opencl)
    llama_cpp_python_install_target(ggml-rpc)
    llama_cpp_python_install_target(ggml-sycl)
    llama_cpp_python_install_target(ggml-vulkan)
    llama_cpp_python_install_target(ggml-webgpu)

    # Workaround for Windows + CUDA https://github.com/abetlen/llama-cpp-python/issues/563
    if (WIN32)
        install(
            FILES $<TARGET_RUNTIME_DLLS:llama>
            DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
        )
        install(
            FILES $<TARGET_RUNTIME_DLLS:llama>
            DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib
        )
        install(
            FILES $<TARGET_RUNTIME_DLLS:ggml>
            DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
        )
        install(
            FILES $<TARGET_RUNTIME_DLLS:ggml>
            DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib
        )
    endif()

    if (MTMD_BUILD)
        add_subdirectory(vendor/llama.cpp/tools/mtmd)

        if (LLAMA_CUBLAS OR LLAMA_CUDA)
            add_compile_definitions(GGML_USE_CUBLAS)
            add_compile_definitions(GGML_USE_CUDA)
        endif()

        if (LLAMA_METAL)
            add_compile_definitions(GGML_USE_METAL)
        endif()

        if (WIN32)
            set_target_properties(mtmd PROPERTIES CUDA_ARCHITECTURES OFF)
        endif()

        llama_cpp_python_install_target(mtmd)

        if (WIN32)
            install(FILES $<TARGET_RUNTIME_DLLS:mtmd> DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib)
            install(FILES $<TARGET_RUNTIME_DLLS:mtmd> DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib)
        endif()
    endif()
else()
    message(FATAL_ERROR "Either LLAMA_USE_EXTERNAL or LLAMA_BUILD must be enabled")
endif()
