# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jiawen/dbms/viper

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jiawen/dbms/viper/build

# Include any dependencies generated for this target.
include benchmark/CMakeFiles/latency_bw_bm.dir/depend.make

# Include the progress variables for this target.
include benchmark/CMakeFiles/latency_bw_bm.dir/progress.make

# Include the compile flags for this target's objects.
include benchmark/CMakeFiles/latency_bw_bm.dir/flags.make

benchmark/CMakeFiles/latency_bw_bm.dir/latency_bw_bm.cpp.o: benchmark/CMakeFiles/latency_bw_bm.dir/flags.make
benchmark/CMakeFiles/latency_bw_bm.dir/latency_bw_bm.cpp.o: ../benchmark/latency_bw_bm.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object benchmark/CMakeFiles/latency_bw_bm.dir/latency_bw_bm.cpp.o"
	cd /home/jiawen/dbms/viper/build/benchmark && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/latency_bw_bm.dir/latency_bw_bm.cpp.o -c /home/jiawen/dbms/viper/benchmark/latency_bw_bm.cpp

benchmark/CMakeFiles/latency_bw_bm.dir/latency_bw_bm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/latency_bw_bm.dir/latency_bw_bm.cpp.i"
	cd /home/jiawen/dbms/viper/build/benchmark && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiawen/dbms/viper/benchmark/latency_bw_bm.cpp > CMakeFiles/latency_bw_bm.dir/latency_bw_bm.cpp.i

benchmark/CMakeFiles/latency_bw_bm.dir/latency_bw_bm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/latency_bw_bm.dir/latency_bw_bm.cpp.s"
	cd /home/jiawen/dbms/viper/build/benchmark && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiawen/dbms/viper/benchmark/latency_bw_bm.cpp -o CMakeFiles/latency_bw_bm.dir/latency_bw_bm.cpp.s

benchmark/CMakeFiles/latency_bw_bm.dir/benchmark.cpp.o: benchmark/CMakeFiles/latency_bw_bm.dir/flags.make
benchmark/CMakeFiles/latency_bw_bm.dir/benchmark.cpp.o: ../benchmark/benchmark.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object benchmark/CMakeFiles/latency_bw_bm.dir/benchmark.cpp.o"
	cd /home/jiawen/dbms/viper/build/benchmark && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/latency_bw_bm.dir/benchmark.cpp.o -c /home/jiawen/dbms/viper/benchmark/benchmark.cpp

benchmark/CMakeFiles/latency_bw_bm.dir/benchmark.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/latency_bw_bm.dir/benchmark.cpp.i"
	cd /home/jiawen/dbms/viper/build/benchmark && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiawen/dbms/viper/benchmark/benchmark.cpp > CMakeFiles/latency_bw_bm.dir/benchmark.cpp.i

benchmark/CMakeFiles/latency_bw_bm.dir/benchmark.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/latency_bw_bm.dir/benchmark.cpp.s"
	cd /home/jiawen/dbms/viper/build/benchmark && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiawen/dbms/viper/benchmark/benchmark.cpp -o CMakeFiles/latency_bw_bm.dir/benchmark.cpp.s

benchmark/CMakeFiles/latency_bw_bm.dir/fixtures/common_fixture.cpp.o: benchmark/CMakeFiles/latency_bw_bm.dir/flags.make
benchmark/CMakeFiles/latency_bw_bm.dir/fixtures/common_fixture.cpp.o: ../benchmark/fixtures/common_fixture.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object benchmark/CMakeFiles/latency_bw_bm.dir/fixtures/common_fixture.cpp.o"
	cd /home/jiawen/dbms/viper/build/benchmark && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/latency_bw_bm.dir/fixtures/common_fixture.cpp.o -c /home/jiawen/dbms/viper/benchmark/fixtures/common_fixture.cpp

benchmark/CMakeFiles/latency_bw_bm.dir/fixtures/common_fixture.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/latency_bw_bm.dir/fixtures/common_fixture.cpp.i"
	cd /home/jiawen/dbms/viper/build/benchmark && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiawen/dbms/viper/benchmark/fixtures/common_fixture.cpp > CMakeFiles/latency_bw_bm.dir/fixtures/common_fixture.cpp.i

benchmark/CMakeFiles/latency_bw_bm.dir/fixtures/common_fixture.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/latency_bw_bm.dir/fixtures/common_fixture.cpp.s"
	cd /home/jiawen/dbms/viper/build/benchmark && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiawen/dbms/viper/benchmark/fixtures/common_fixture.cpp -o CMakeFiles/latency_bw_bm.dir/fixtures/common_fixture.cpp.s

# Object files for target latency_bw_bm
latency_bw_bm_OBJECTS = \
"CMakeFiles/latency_bw_bm.dir/latency_bw_bm.cpp.o" \
"CMakeFiles/latency_bw_bm.dir/benchmark.cpp.o" \
"CMakeFiles/latency_bw_bm.dir/fixtures/common_fixture.cpp.o"

# External object files for target latency_bw_bm
latency_bw_bm_EXTERNAL_OBJECTS =

benchmark/latency_bw_bm: benchmark/CMakeFiles/latency_bw_bm.dir/latency_bw_bm.cpp.o
benchmark/latency_bw_bm: benchmark/CMakeFiles/latency_bw_bm.dir/benchmark.cpp.o
benchmark/latency_bw_bm: benchmark/CMakeFiles/latency_bw_bm.dir/fixtures/common_fixture.cpp.o
benchmark/latency_bw_bm: benchmark/CMakeFiles/latency_bw_bm.dir/build.make
benchmark/latency_bw_bm: _deps/google_benchmark-build/src/libbenchmark.a
benchmark/latency_bw_bm: _deps/hdr_histogram-build/src/libhdr_histogram_static.a
benchmark/latency_bw_bm: /usr/lib/x86_64-linux-gnu/librt.so
benchmark/latency_bw_bm: /usr/lib/x86_64-linux-gnu/libz.so
benchmark/latency_bw_bm: benchmark/CMakeFiles/latency_bw_bm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable latency_bw_bm"
	cd /home/jiawen/dbms/viper/build/benchmark && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/latency_bw_bm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
benchmark/CMakeFiles/latency_bw_bm.dir/build: benchmark/latency_bw_bm

.PHONY : benchmark/CMakeFiles/latency_bw_bm.dir/build

benchmark/CMakeFiles/latency_bw_bm.dir/clean:
	cd /home/jiawen/dbms/viper/build/benchmark && $(CMAKE_COMMAND) -P CMakeFiles/latency_bw_bm.dir/cmake_clean.cmake
.PHONY : benchmark/CMakeFiles/latency_bw_bm.dir/clean

benchmark/CMakeFiles/latency_bw_bm.dir/depend:
	cd /home/jiawen/dbms/viper/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jiawen/dbms/viper /home/jiawen/dbms/viper/benchmark /home/jiawen/dbms/viper/build /home/jiawen/dbms/viper/build/benchmark /home/jiawen/dbms/viper/build/benchmark/CMakeFiles/latency_bw_bm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : benchmark/CMakeFiles/latency_bw_bm.dir/depend

