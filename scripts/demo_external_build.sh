#!/bin/bash

# Demo script showing how to use llama-cpp-python with external llama.cpp

set -e  # Exit on any error

echo "=== llama-cpp-python External Build Demo ==="
echo

# Check if llama.cpp is installed system-wide
if command -v llama-main >/dev/null 2>&1; then
    echo "✓ llama.cpp appears to be installed system-wide"
    LLAMA_VERSION=$(llama-main --version 2>/dev/null || echo "unknown")
    echo "  Version: $LLAMA_VERSION"
elif pkg-config --exists llama 2>/dev/null; then
    echo "✓ llama.cpp found via pkg-config"
    LLAMA_VERSION=$(pkg-config --modversion llama)
    echo "  Version: $LLAMA_VERSION"
elif [ -f /usr/local/lib/libllama.so ] || [ -f /usr/local/lib/libllama.dylib ]; then
    echo "✓ llama.cpp library found in /usr/local/lib"
elif [ -f /opt/homebrew/lib/libllama.so ] || [ -f /opt/homebrew/lib/libllama.dylib ]; then
    echo "✓ llama.cpp library found in /opt/homebrew/lib"
else
    echo "⚠ llama.cpp not found in standard locations"
    echo "  You may need to specify LLAMA_CPP_LIB_PATH during build"
    echo
    echo "To build llama.cpp externally:"
    echo "  git clone https://github.com/ggerganov/llama.cpp.git"
    echo "  cd llama.cpp"
    echo "  mkdir build && cd build"
    echo "  cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release"
    echo "  make -j\$(nproc)"
    echo "  sudo make install  # Optional"
    echo
fi

echo
echo "Current llama-cpp-python configuration:"

# Check current build configuration
if python -c "import llama_cpp; print('✓ llama_cpp imports successfully')" 2>/dev/null; then
    python -c "
import llama_cpp.llama_cpp as llama_cpp_mod
import pathlib
print(f'  Library base path: {llama_cpp_mod._base_path}')
if llama_cpp_mod._override_base_path:
    print(f'  Override path: {llama_cpp_mod._override_base_path}')
else:
    print('  No override path set')
"
else
    echo "✗ llama_cpp failed to import - may need to rebuild"
fi

echo
echo "Build commands for different scenarios:"
echo
echo "1. Default (uses external llama.cpp):"
echo "   pip install llama-cpp-python"
echo
echo "2. With custom llama.cpp path:"
echo "   CMAKE_ARGS=\"-DLLAMA_CPP_LIB_PATH=/custom/path\" pip install llama-cpp-python"
echo
echo "3. Legacy internal build:"
echo "   CMAKE_ARGS=\"-DLLAMA_BUILD=ON -DLLAMA_USE_EXTERNAL=OFF\" pip install llama-cpp-python"
echo
echo "4. Development build with external libraries:"
echo "   LLAMA_LIB_PATH=/path/to/llama.cpp/build make build.external"
echo
echo "5. Runtime library path override:"
echo "   LLAMA_CPP_LIB_PATH=/runtime/path python your_script.py"
echo

echo "=== End Demo ==="