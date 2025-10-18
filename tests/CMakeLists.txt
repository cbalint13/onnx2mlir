add_custom_target(python_tests
  COMMAND ${CMAKE_COMMAND} -E env
    "PYTHONPATH=${CMAKE_BINARY_DIR}/python:${CMAKE_SOURCE_DIR}/python"
    "PYTHONDONTWRITEBYTECODE=ON"
    ${PYTEST_EXECUTABLE} -v -s -p no:cacheprovider
    --junit-xml=${CMAKE_BINARY_DIR}/tests/python/pytest_report.xml
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/python
  COMMENT "Running python tests"
)

add_custom_target(test
  DEPENDS
    python_tests
  COMMENT "Finish with tests"
)
