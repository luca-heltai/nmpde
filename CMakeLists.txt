cmake_minimum_required(VERSION 3.25)

project(nmpde LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

option(NMPDE_BUILD_LABS "Build the course laboratory executables" ON)
option(NMPDE_BUILD_DEALII_LABS
       "Build the deal.II-based laboratory executables"
       ON)
option(NMPDE_BUILD_TESTS "Build the GoogleTest suite and asset generators" ON)

if(NMPDE_BUILD_DEALII_LABS OR NMPDE_BUILD_TESTS)
  find_package(deal.II 9.7.1 REQUIRED
    HINTS ${deal.II_DIR} $ENV{DEAL_II_DIR}
  )
  deal_ii_initialize_cached_variables()
endif()

set(NMPDE_DEALII_LAB_TARGETS
  lab-02
  lab-03
  lab-04
  lab-05
  lab-05b
  lab-05c
  lab-05d
  lab-05e
  lab-06
  lab-07
  lab-08
  lab-09
)

if(NMPDE_BUILD_LABS)
  add_executable(lab-01 "labs/lab-01/lab-01.cc")
  target_compile_features(lab-01 PRIVATE cxx_std_17)
endif()

if(NMPDE_BUILD_LABS AND NMPDE_BUILD_DEALII_LABS)
  foreach(lab IN LISTS NMPDE_DEALII_LAB_TARGETS)
    add_executable(${lab} "labs/${lab}/${lab}.cc")
    deal_ii_setup_target(${lab})
  endforeach()
endif()

if(NMPDE_BUILD_TESTS)
  include(CTest)
  include(FetchContent)

  find_package(GTest CONFIG QUIET)
  if(NOT GTest_FOUND)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    )
    FetchContent_MakeAvailable(googletest)
  endif()

  add_executable(nmpde_tests tests/course_tests.cc)
  target_compile_definitions(nmpde_tests PRIVATE
    NMPDE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    NMPDE_ASSET_DIR="${CMAKE_SOURCE_DIR}/notes/assets/generated"
  )
  target_link_libraries(nmpde_tests GTest::gtest_main)
  deal_ii_setup_target(nmpde_tests)

  include(GoogleTest)
  gtest_discover_tests(nmpde_tests DISCOVERY_MODE PRE_TEST)
endif()
