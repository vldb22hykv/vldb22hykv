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
include _deps/pmemkv-build/examples/CMakeFiles/example-pmemkv_config_c.dir/depend.make

# Include the progress variables for this target.
include _deps/pmemkv-build/examples/CMakeFiles/example-pmemkv_config_c.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/pmemkv-build/examples/CMakeFiles/example-pmemkv_config_c.dir/flags.make

_deps/pmemkv-build/examples/CMakeFiles/example-pmemkv_config_c.dir/pmemkv_config_c/pmemkv_config.c.o: _deps/pmemkv-build/examples/CMakeFiles/example-pmemkv_config_c.dir/flags.make
_deps/pmemkv-build/examples/CMakeFiles/example-pmemkv_config_c.dir/pmemkv_config_c/pmemkv_config.c.o: _deps/pmemkv-src/examples/pmemkv_config_c/pmemkv_config.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object _deps/pmemkv-build/examples/CMakeFiles/example-pmemkv_config_c.dir/pmemkv_config_c/pmemkv_config.c.o"
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build/examples && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/example-pmemkv_config_c.dir/pmemkv_config_c/pmemkv_config.c.o   -c /home/jiawen/dbms/viper/build/_deps/pmemkv-src/examples/pmemkv_config_c/pmemkv_config.c

_deps/pmemkv-build/examples/CMakeFiles/example-pmemkv_config_c.dir/pmemkv_config_c/pmemkv_config.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/example-pmemkv_config_c.dir/pmemkv_config_c/pmemkv_config.c.i"
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build/examples && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/jiawen/dbms/viper/build/_deps/pmemkv-src/examples/pmemkv_config_c/pmemkv_config.c > CMakeFiles/example-pmemkv_config_c.dir/pmemkv_config_c/pmemkv_config.c.i

_deps/pmemkv-build/examples/CMakeFiles/example-pmemkv_config_c.dir/pmemkv_config_c/pmemkv_config.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/example-pmemkv_config_c.dir/pmemkv_config_c/pmemkv_config.c.s"
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build/examples && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/jiawen/dbms/viper/build/_deps/pmemkv-src/examples/pmemkv_config_c/pmemkv_config.c -o CMakeFiles/example-pmemkv_config_c.dir/pmemkv_config_c/pmemkv_config.c.s

# Object files for target example-pmemkv_config_c
example__pmemkv_config_c_OBJECTS = \
"CMakeFiles/example-pmemkv_config_c.dir/pmemkv_config_c/pmemkv_config.c.o"

# External object files for target example-pmemkv_config_c
example__pmemkv_config_c_EXTERNAL_OBJECTS =

_deps/pmemkv-build/examples/example-pmemkv_config_c: _deps/pmemkv-build/examples/CMakeFiles/example-pmemkv_config_c.dir/pmemkv_config_c/pmemkv_config.c.o
_deps/pmemkv-build/examples/example-pmemkv_config_c: _deps/pmemkv-build/examples/CMakeFiles/example-pmemkv_config_c.dir/build.make
_deps/pmemkv-build/examples/example-pmemkv_config_c: _deps/pmemkv-build/libpmemkv_json_config.so.1
_deps/pmemkv-build/examples/example-pmemkv_config_c: _deps/pmemkv-build/libpmemkv.so.1
_deps/pmemkv-build/examples/example-pmemkv_config_c: _deps/pmemkv-build/examples/CMakeFiles/example-pmemkv_config_c.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable example-pmemkv_config_c"
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example-pmemkv_config_c.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
_deps/pmemkv-build/examples/CMakeFiles/example-pmemkv_config_c.dir/build: _deps/pmemkv-build/examples/example-pmemkv_config_c

.PHONY : _deps/pmemkv-build/examples/CMakeFiles/example-pmemkv_config_c.dir/build

_deps/pmemkv-build/examples/CMakeFiles/example-pmemkv_config_c.dir/clean:
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build/examples && $(CMAKE_COMMAND) -P CMakeFiles/example-pmemkv_config_c.dir/cmake_clean.cmake
.PHONY : _deps/pmemkv-build/examples/CMakeFiles/example-pmemkv_config_c.dir/clean

_deps/pmemkv-build/examples/CMakeFiles/example-pmemkv_config_c.dir/depend:
	cd /home/jiawen/dbms/viper/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jiawen/dbms/viper /home/jiawen/dbms/viper/build/_deps/pmemkv-src/examples /home/jiawen/dbms/viper/build /home/jiawen/dbms/viper/build/_deps/pmemkv-build/examples /home/jiawen/dbms/viper/build/_deps/pmemkv-build/examples/CMakeFiles/example-pmemkv_config_c.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : _deps/pmemkv-build/examples/CMakeFiles/example-pmemkv_config_c.dir/depend

