# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.26.3/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.26.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/karenli/Desktop/Things/ORB

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/karenli/Desktop/Things/ORB/build

# Include any dependencies generated for this target.
include CMakeFiles/ORB.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ORB.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ORB.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ORB.dir/flags.make

CMakeFiles/ORB.dir/ORB.cpp.o: CMakeFiles/ORB.dir/flags.make
CMakeFiles/ORB.dir/ORB.cpp.o: /Users/karenli/Desktop/Things/ORB/ORB.cpp
CMakeFiles/ORB.dir/ORB.cpp.o: CMakeFiles/ORB.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/karenli/Desktop/Things/ORB/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ORB.dir/ORB.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ORB.dir/ORB.cpp.o -MF CMakeFiles/ORB.dir/ORB.cpp.o.d -o CMakeFiles/ORB.dir/ORB.cpp.o -c /Users/karenli/Desktop/Things/ORB/ORB.cpp

CMakeFiles/ORB.dir/ORB.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ORB.dir/ORB.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/karenli/Desktop/Things/ORB/ORB.cpp > CMakeFiles/ORB.dir/ORB.cpp.i

CMakeFiles/ORB.dir/ORB.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ORB.dir/ORB.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/karenli/Desktop/Things/ORB/ORB.cpp -o CMakeFiles/ORB.dir/ORB.cpp.s

# Object files for target ORB
ORB_OBJECTS = \
"CMakeFiles/ORB.dir/ORB.cpp.o"

# External object files for target ORB
ORB_EXTERNAL_OBJECTS =

ORB: CMakeFiles/ORB.dir/ORB.cpp.o
ORB: CMakeFiles/ORB.dir/build.make
ORB: /usr/local/lib/libopencv_gapi.4.7.0.dylib
ORB: /usr/local/lib/libopencv_highgui.4.7.0.dylib
ORB: /usr/local/lib/libopencv_ml.4.7.0.dylib
ORB: /usr/local/lib/libopencv_objdetect.4.7.0.dylib
ORB: /usr/local/lib/libopencv_photo.4.7.0.dylib
ORB: /usr/local/lib/libopencv_stitching.4.7.0.dylib
ORB: /usr/local/lib/libopencv_video.4.7.0.dylib
ORB: /usr/local/lib/libopencv_videoio.4.7.0.dylib
ORB: /usr/local/lib/libopencv_imgcodecs.4.7.0.dylib
ORB: /usr/local/lib/libopencv_dnn.4.7.0.dylib
ORB: /usr/local/lib/libopencv_calib3d.4.7.0.dylib
ORB: /usr/local/lib/libopencv_features2d.4.7.0.dylib
ORB: /usr/local/lib/libopencv_flann.4.7.0.dylib
ORB: /usr/local/lib/libopencv_imgproc.4.7.0.dylib
ORB: /usr/local/lib/libopencv_core.4.7.0.dylib
ORB: CMakeFiles/ORB.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/karenli/Desktop/Things/ORB/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ORB"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ORB.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ORB.dir/build: ORB
.PHONY : CMakeFiles/ORB.dir/build

CMakeFiles/ORB.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ORB.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ORB.dir/clean

CMakeFiles/ORB.dir/depend:
	cd /Users/karenli/Desktop/Things/ORB/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/karenli/Desktop/Things/ORB /Users/karenli/Desktop/Things/ORB /Users/karenli/Desktop/Things/ORB/build /Users/karenli/Desktop/Things/ORB/build /Users/karenli/Desktop/Things/ORB/build/CMakeFiles/ORB.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ORB.dir/depend

