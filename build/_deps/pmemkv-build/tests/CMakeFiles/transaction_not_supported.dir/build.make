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
include _deps/pmemkv-build/tests/CMakeFiles/transaction_not_supported.dir/depend.make

# Include the progress variables for this target.
include _deps/pmemkv-build/tests/CMakeFiles/transaction_not_supported.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/pmemkv-build/tests/CMakeFiles/transaction_not_supported.dir/flags.make

_deps/pmemkv-build/tests/CMakeFiles/transaction_not_supported.dir/engine_scenarios/transaction/not_supported.cc.o: _deps/pmemkv-build/tests/CMakeFiles/transaction_not_supported.dir/flags.make
_deps/pmemkv-build/tests/CMakeFiles/transaction_not_supported.dir/engine_scenarios/transaction/not_supported.cc.o: _deps/pmemkv-src/tests/engine_scenarios/transaction/not_supported.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object _deps/pmemkv-build/tests/CMakeFiles/transaction_not_supported.dir/engine_scenarios/transaction/not_supported.cc.o"
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/transaction_not_supported.dir/engine_scenarios/transaction/not_supported.cc.o -c /home/jiawen/dbms/viper/build/_deps/pmemkv-src/tests/engine_scenarios/transaction/not_supported.cc

_deps/pmemkv-build/tests/CMakeFiles/transaction_not_supported.dir/engine_scenarios/transaction/not_supported.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/transaction_not_supported.dir/engine_scenarios/transaction/not_supported.cc.i"
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiawen/dbms/viper/build/_deps/pmemkv-src/tests/engine_scenarios/transaction/not_supported.cc > CMakeFiles/transaction_not_supported.dir/engine_scenarios/transaction/not_supported.cc.i

_deps/pmemkv-build/tests/CMakeFiles/transaction_not_supported.dir/engine_scenarios/transaction/not_supported.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/transaction_not_supported.dir/engine_scenarios/transaction/not_supported.cc.s"
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiawen/dbms/viper/build/_deps/pmemkv-src/tests/engine_scenarios/transaction/not_supported.cc -o CMakeFiles/transaction_not_supported.dir/engine_scenarios/transaction/not_supported.cc.s

# Object files for target transaction_not_supported
transaction_not_supported_OBJECTS = \
"CMakeFiles/transaction_not_supported.dir/engine_scenarios/transaction/not_supported.cc.o"

# External object files for target transaction_not_supported
transaction_not_supported_EXTERNAL_OBJECTS =

_deps/pmemkv-build/tests/transaction_not_supported: _deps/pmemkv-build/tests/CMakeFiles/transaction_not_supported.dir/engine_scenarios/transaction/not_supported.cc.o
_deps/pmemkv-build/tests/transaction_not_supported: _deps/pmemkv-build/tests/CMakeFiles/transaction_not_supported.dir/build.make
_deps/pmemkv-build/tests/transaction_not_supported: _deps/pmemkv-build/tests/libtest_backtrace.a
_deps/pmemkv-build/tests/transaction_not_supported: _deps/pmemkv-build/libpmemkv_json_config.so.1
_deps/pmemkv-build/tests/transaction_not_supported: _deps/pmemkv-build/libpmemkv.so.1
_deps/pmemkv-build/tests/transaction_not_supported: _deps/pmemkv-build/tests/CMakeFiles/transaction_not_supported.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable transaction_not_supported"
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/transaction_not_supported.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
_deps/pmemkv-build/tests/CMakeFiles/transaction_not_supported.dir/build: _deps/pmemkv-build/tests/transaction_not_supported

.PHONY : _deps/pmemkv-build/tests/CMakeFiles/transaction_not_supported.dir/build

_deps/pmemkv-build/tests/CMakeFiles/transaction_not_supported.dir/clean:
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build/tests && $(CMAKE_COMMAND) -P CMakeFiles/transaction_not_supported.dir/cmake_clean.cmake
.PHONY : _deps/pmemkv-build/tests/CMakeFiles/transaction_not_supported.dir/clean

_deps/pmemkv-build/tests/CMakeFiles/transaction_not_supported.dir/depend:
	cd /home/jiawen/dbms/viper/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jiawen/dbms/viper /home/jiawen/dbms/viper/build/_deps/pmemkv-src/tests /home/jiawen/dbms/viper/build /home/jiawen/dbms/viper/build/_deps/pmemkv-build/tests /home/jiawen/dbms/viper/build/_deps/pmemkv-build/tests/CMakeFiles/transaction_not_supported.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : _deps/pmemkv-build/tests/CMakeFiles/transaction_not_supported.dir/depend

