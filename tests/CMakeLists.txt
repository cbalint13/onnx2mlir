add_test(
    NAME all_python_tests
    COMMAND ${PYTEST_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/python
)
set_tests_properties(all_python_tests PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python")
