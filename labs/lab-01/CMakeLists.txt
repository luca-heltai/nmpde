# Set the name of the target executable
set(TARGET "lab-01")

# Set the source file for the target
set(TARGET_SRC "lab-01.cc")

# Minimum required version of CMake
cmake_minimum_required(VERSION 3.13.4)

# Find the deal.II library, version 9.3.0 or newer, using various hints for the location
find_package(deal.II 9.3.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

# If deal.II is not found, display a fatal error message and stop configuration
if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
  )
endif()

# Initialize cached deal.II variables
deal_ii_initialize_cached_variables()

# Define the project with the target name
project(${TARGET})

# Automatically configure target properties and dependencies using deal.II autopilot
deal_ii_invoke_autopilot()
