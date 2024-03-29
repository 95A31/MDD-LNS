# - Config file for the libminizinc package
# It defines the following variables
#  libminizinc_INCLUDE_DIRS - include directories for libminizinc
#  libminizinc_LIBRARIES    - libraries to link against
#  libminizinc_EXECUTABLE   - the bar executable

# Compute paths
get_filename_component(libminizinc_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
set(libminizinc_INCLUDE_DIRS "${libminizinc_CMAKE_DIR}/../../../include")

# Our library dependencies
list(APPEND CMAKE_MODULE_PATH ${libminizinc_CMAKE_DIR})
include(CMakeFindDependencyMacro)


# Our library targets (contains definitions for IMPORTED targets)
include("${libminizinc_CMAKE_DIR}/libminizincTargets.cmake")

