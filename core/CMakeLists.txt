# ==========================================================================
# SFPP Playground Core Library (header-only)
# ==========================================================================

include(GNUInstallDirs)

# Header-only target
add_library(sfpp_playground INTERFACE)
add_library(SFPP_Playground::core ALIAS sfpp_playground)

# Public include directories for consumers (build and install time)
target_include_directories(
  sfpp_playground INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Propagate language standard to dependents
target_compile_features(sfpp_playground INTERFACE cxx_std_20)

# Optional: propagate warnings interface to consumers
if(TARGET SFPP_Playground::warnings)
  target_link_libraries(sfpp_playground INTERFACE SFPP_Playground::warnings)
endif()

# propagate kokkos dependency
target_link_libraries(sfpp_playground INTERFACE Kokkos::kokkos)

# Provide a friendly export name
set_target_properties(sfpp_playground PROPERTIES EXPORT_NAME core)

# Install the interface target into the export set declared at the root
install(TARGETS sfpp_playground EXPORT SFPP_PlaygroundTargets)

# Install public headers (all .hpp files under this directory)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.hpp")
