#----------------------------------------------------------------
# Generated CMake target import file for configuration "RELEASE".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "mzn" for configuration "RELEASE"
set_property(TARGET mzn APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(mzn PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C;CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmzn.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS mzn )
list(APPEND _IMPORT_CHECK_FILES_FOR_mzn "${_IMPORT_PREFIX}/lib/libmzn.a" )

# Import target "minizinc" for configuration "RELEASE"
set_property(TARGET minizinc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(minizinc PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/minizinc"
  )

list(APPEND _IMPORT_CHECK_TARGETS minizinc )
list(APPEND _IMPORT_CHECK_FILES_FOR_minizinc "${_IMPORT_PREFIX}/bin/minizinc" )

# Import target "mzn2doc" for configuration "RELEASE"
set_property(TARGET mzn2doc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(mzn2doc PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/mzn2doc"
  )

list(APPEND _IMPORT_CHECK_TARGETS mzn2doc )
list(APPEND _IMPORT_CHECK_FILES_FOR_mzn2doc "${_IMPORT_PREFIX}/bin/mzn2doc" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
