# Summary of Changes: External llama.cpp Support

This document summarizes the changes made to configure llama-cpp-python to use externally built llama.cpp libraries by default, instead of building from vendored sources.

## Files Modified

### Build System Changes

1. **`CMakeLists.txt`**
   - Changed default options: `LLAMA_BUILD=OFF`, `LLAMA_USE_EXTERNAL=ON`
   - Added new options: `LLAMA_CPP_LIB_PATH`, `LLAMA_CPP_INCLUDE_PATH`
   - Added logic to find and use external llama.cpp libraries
   - Added `llama_cpp_python_install_external_library()` function
   - Preserved internal build logic for legacy use (when `LLAMA_BUILD=ON`)

2. **`pyproject.toml`**
   - Changed sdist configuration from include to exclude vendor sources by default
   - Added comments explaining how to use internal build if needed

3. **`Makefile`**
   - Updated default `build` target to use external libraries
   - Added `build.external` target with custom path support
   - Added `build.internal` targets for legacy internal builds
   - Added debug variants for internal builds
   - Updated `.PHONY` declarations

### Documentation Changes

4. **`docs/external-llama-cpp.md`** (new file)
   - Comprehensive guide for using external llama.cpp
   - Instructions for building llama.cpp with different backends
   - Build and installation examples
   - Troubleshooting guide
   - Environment variable documentation
   - Docker and CI/CD examples

5. **`README.md`**
   - Updated installation section to prioritize external llama.cpp
   - Added section explaining external vs internal build methods
   - Updated development instructions
   - Fixed markdown formatting issues

6. **`.github/copilot-instructions.md`**
   - Updated to reflect external llama.cpp as the preferred approach
   - Modified build instructions and workflows
   - Updated validation sequences

### Testing and Validation

7. **`tests/test_external_config.py`** (new file)
   - Tests for external library loading mechanism
   - Configuration validation tests
   - Environment variable override tests

8. **`scripts/demo_external_build.sh`** (new file)
   - Demonstration script showing external build configuration
   - Detects existing llama.cpp installations
   - Shows build commands for different scenarios

## Key Changes Summary

### Default Behavior Change
- **Before**: Builds llama.cpp from vendored sources by default
- **After**: Uses externally built llama.cpp by default

### Build Process Changes
- **External Build** (new default):
  - Expects llama.cpp to be pre-installed
  - Searches standard library paths
  - Allows custom paths via `LLAMA_CPP_LIB_PATH`
  - Copies external libraries to package

- **Internal Build** (legacy):
  - Requires explicit opt-in: `CMAKE_ARGS="-DLLAMA_BUILD=ON -DLLAMA_USE_EXTERNAL=OFF"`
  - Builds from vendored sources in `vendor/llama.cpp`
  - Still supports all backend configurations

### Installation Methods
1. **Default** (external): `pip install llama-cpp-python`
2. **With custom path**: `CMAKE_ARGS="-DLLAMA_CPP_LIB_PATH=/path" pip install llama-cpp-python`
3. **Legacy internal**: `CMAKE_ARGS="-DLLAMA_BUILD=ON -DLLAMA_USE_EXTERNAL=OFF" pip install llama-cpp-python`

### Environment Variables
- **Build time**:
  - `LLAMA_LIB_PATH`: Path for Makefile targets
  - `CMAKE_ARGS`: CMake configuration arguments
- **Runtime**:
  - `LLAMA_CPP_LIB_PATH`: Override library search path

## Benefits of External Approach

1. **Smaller package size**: No vendored source code in distributions
2. **Flexibility**: Users can optimize llama.cpp for their specific needs
3. **System integration**: Share llama.cpp across multiple applications
4. **Faster builds**: Skip compilation if llama.cpp is already built
5. **Custom patches**: Use modified llama.cpp versions

## Migration Guide

### For End Users
- **No action required** if llama.cpp is installed system-wide
- **Custom installations**: Set `CMAKE_ARGS="-DLLAMA_CPP_LIB_PATH=/path"` during install
- **Legacy builds**: Add `CMAKE_ARGS="-DLLAMA_BUILD=ON -DLLAMA_USE_EXTERNAL=OFF"`

### For Developers
- **Default development**: Install llama.cpp externally, then `pip install -e .`
- **Legacy development**: Use `make build.internal` instead of `make build`
- **Custom paths**: Use `LLAMA_LIB_PATH=/path make build.external`

### For CI/CD
- **Before**: `pip install llama-cpp-python`
- **After**: Build llama.cpp first, then `pip install llama-cpp-python`

## Backward Compatibility

- All existing functionality is preserved
- Legacy internal build still available via explicit configuration
- All backends (CUDA, Metal, OpenBLAS, etc.) still supported
- Python API unchanged
- Server functionality unchanged

## Testing Status

- ✅ Package imports successfully
- ✅ External configuration tests pass
- ✅ Demo script validates configuration
- ✅ Build system supports both modes
- ⚠️ Full integration tests require actual llama.cpp installation

## Future Considerations

1. Update CI/CD pipelines to use external builds
2. Create pre-built wheels with common llama.cpp configurations
3. Add pkg-config support for better library detection
4. Consider removing vendor directory in future major version
5. Add automatic llama.cpp version compatibility checking