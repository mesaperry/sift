cmake_minimum_required( VERSION 3.0 )

# Create Project
project( main )
add_executable( main benchmark.cpp rootsift.cpp )

# Find Package
find_package( OpenCV REQUIRED )
if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Library Directories
  link_directories( ${OpenCV_LIB_DIR} )

  # Additional Dependencies
  target_link_libraries( main ${OpenCV_LIBS} )
endif()
