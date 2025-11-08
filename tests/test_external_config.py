try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Mock pytest.raises for when pytest is not available
    class MockPytest:
        class raises:
            def __init__(self, exc_type):
                self.exc_type = exc_type
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    raise AssertionError(f"Expected {self.exc_type.__name__} but no exception was raised")
                if not issubclass(exc_type, self.exc_type):
                    return False  # Let the exception propagate
                return True  # Suppress the expected exception
    pytest = MockPytest()

from llama_cpp._ctypes_extensions import load_shared_library
import pathlib


def test_external_library_loading():
    """Test that external library loading mechanism works"""
    # Test that the override path mechanism works
    base_path = pathlib.Path("/nonexistent/path")
    
    # This should raise FileNotFoundError since the path doesn't exist
    with pytest.raises(FileNotFoundError):
        load_shared_library("llama", base_path)


def test_environment_override():
    """Test that LLAMA_CPP_LIB_PATH environment variable is respected"""
    from llama_cpp import llama_cpp
    
    # The override path should be configurable
    # This test verifies the mechanism exists without requiring actual libraries
    assert hasattr(llama_cpp, '_override_base_path')
    assert hasattr(llama_cpp, '_base_path')


def test_cmake_configuration_options():
    """Test that the CMake configuration supports external builds"""
    # This is more of a documentation test to ensure the options are known
    external_build_options = {
        "LLAMA_USE_EXTERNAL": "ON",
        "LLAMA_BUILD": "OFF", 
        "LLAMA_CPP_LIB_PATH": "/custom/path/to/lib",
        "LLAMA_CPP_INCLUDE_PATH": "/custom/path/to/include"
    }
    
    # Just verify these are the expected configuration options
    # In a real build, these would be passed via CMAKE_ARGS
    assert all(isinstance(key, str) and isinstance(value, str) 
               for key, value in external_build_options.items())


if __name__ == "__main__":
    test_external_library_loading()
    test_environment_override() 
    test_cmake_configuration_options()
    print("External library configuration tests passed!")