# Using External llama.cpp

This guide explains how to configure llama-cpp-python to use an externally built version of llama.cpp instead of the bundled version.

## Benefits of External llama.cpp

- **Smaller package size**: No need to include llama.cpp source code in the Python package
- **Flexibility**: Use your own optimized build of llama.cpp
- **System integration**: Share llama.cpp installation across multiple applications
- **Custom builds**: Use llama.cpp with custom patches or configurations

## Default Behavior

By default, llama-cpp-python now expects to use an externally built llama.cpp library. The build system will search for llama.cpp in standard system locations.

## Building llama.cpp Externally

### Prerequisites

- CMake 3.21 or higher
- C++ compiler with C++17 support
- Git

### Step 1: Clone and Build llama.cpp

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with shared libraries
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Install to system location (optional)
sudo make install
```

### Step 2: Build with Specific Backends

For CUDA support:

```bash
cmake .. -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
```

For Metal support (macOS):

```bash
cmake .. -DBUILD_SHARED_LIBS=ON -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
```

For OpenBLAS support:

```bash
cmake .. -DBUILD_SHARED_LIBS=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DCMAKE_BUILD_TYPE=Release
```

### Step 3: Install llama.cpp (Optional)

To install to a standard system location:

```bash
sudo make install
```

Or install to a custom location:

```bash
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/llama.cpp -DBUILD_SHARED_LIBS=ON
make install
```

## Building llama-cpp-python

### Method 1: System Installation

If you installed llama.cpp to a standard location:

```bash
pip install llama-cpp-python
```

### Method 2: Custom Installation Path

If llama.cpp is installed in a custom location:

```bash
export LLAMA_LIB_PATH=/opt/llama.cpp/lib
make build.external
```

Or using cmake directly:

```bash
CMAKE_ARGS="-DLLAMA_CPP_LIB_PATH=/opt/llama.cpp/lib" pip install llama-cpp-python
```

### Method 3: Development Build

For development with external llama.cpp:

```bash
export LLAMA_LIB_PATH=/path/to/llama.cpp/build
make build.external
```

## Environment Variables

### Build Time

- `LLAMA_LIB_PATH`: Path to directory containing llama.cpp libraries
- `CMAKE_ARGS`: Additional CMake arguments

### Runtime

- `LLAMA_CPP_LIB_PATH`: Override the library search path at runtime

## Legacy Internal Build

If you need to use the old internal build method (not recommended):

```bash
make build.internal
```

Or with CMake args:

```bash
CMAKE_ARGS="-DLLAMA_BUILD=ON -DLLAMA_USE_EXTERNAL=OFF" pip install llama-cpp-python
```

## Troubleshooting

### Library Not Found

If you get an error about llama.cpp not being found:

1. Ensure llama.cpp is built with shared libraries (`-DBUILD_SHARED_LIBS=ON`)
2. Verify the library path is correct
3. Check that all required dependencies are installed
4. Use `LLAMA_CPP_LIB_PATH` environment variable to specify the library location

### Runtime Library Loading

If the library loads during build but fails at runtime:

1. Ensure the library is in your system's library path
2. Use `LLAMA_CPP_LIB_PATH` environment variable
3. Check that all backend dependencies are available

### Backend-Specific Issues

For CUDA:

- Ensure CUDA runtime is installed
- Verify CUDA libraries are in the library path

For Metal (macOS):

- Ensure you're on macOS with Metal support
- Metal libraries should be automatically found

### Version Compatibility

Ensure your external llama.cpp version is compatible with this version of llama-cpp-python. Major API changes in llama.cpp may require updates to the Python bindings.

## Performance Considerations

- Use `CMAKE_BUILD_TYPE=Release` for optimal performance
- Consider enabling CPU-specific optimizations during llama.cpp build
- For CUDA builds, specify appropriate compute capabilities

## Example Workflows

### Docker with External llama.cpp

```dockerfile
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y cmake build-essential python3 python3-pip git

# Build llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp
WORKDIR /opt/llama.cpp
RUN mkdir build && cd build && \
    cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install

# Install llama-cpp-python
RUN pip3 install llama-cpp-python

WORKDIR /app
```

### CI/CD Pipeline

```yaml
- name: Build llama.cpp
  run: |
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    mkdir build && cd build
    cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    sudo make install

- name: Build llama-cpp-python
  run: |
    pip install llama-cpp-python
```
