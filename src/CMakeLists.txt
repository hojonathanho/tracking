include_directories(
  SYSTEM
  ${Boost_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
)

if (NOT APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
endif()

add_executable(test_npeigen
  test_npeigen.cpp
  npeigen.cpp
)
target_link_libraries(test_npeigen ${Boost_PYTHON_LIBRARY} ${PYTHON_LIBRARIES})


include_directories(${PYTHON_NUMPY_INCLUDE_DIR})
boost_python_module(trackingpy python_interface.cpp mass_system.cpp npeigen.cpp)
target_link_libraries(trackingpy ${Boost_SYSTEM_LIBRARY})
