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
include _deps/pmemkv-build/tests/CMakeFiles/test_backtrace.dir/depend.make

# Include the progress variables for this target.
include _deps/pmemkv-build/tests/CMakeFiles/test_backtrace.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/pmemkv-build/tests/CMakeFiles/test_backtrace.dir/flags.make

_deps/pmemkv-build/tests/CMakeFiles/test_backtrace.dir/common/test_backtrace.c.o: _deps/pmemkv-build/tests/CMakeFiles/test_backtrace.dir/flags.make
_deps/pmemkv-build/tests/CMakeFiles/test_backtrace.dir/common/test_backtrace.c.o: _deps/pmemkv-src/tests/common/test_backtrace.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object _deps/pmemkv-build/tests/CMakeFiles/test_backtrace.dir/common/test_backtrace.c.o"
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/test_backtrace.dir/common/test_backtrace.c.o   -c /home/jiawen/dbms/viper/build/_deps/pmemkv-src/tests/common/test_backtrace.c

_deps/pmemkv-build/tests/CMakeFiles/test_backtrace.dir/common/test_backtrace.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/test_backtrace.dir/common/test_backtrace.c.i"
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/jiawen/dbms/viper/build/_deps/pmemkv-src/tests/common/test_backtrace.c > CMakeFiles/test_backtrace.dir/common/test_backtrace.c.i

_deps/pmemkv-build/tests/CMakeFiles/test_backtrace.dir/common/test_backtrace.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/test_backtrace.dir/common/test_backtrace.c.s"
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/jiawen/dbms/viper/build/_deps/pmemkv-src/tests/common/test_backtrace.c -o CMakeFiles/test_backtrace.dir/common/test_backtrace.c.s

# Object files for target test_backtrace
test_backtrace_OBJECTS = \
"CMakeFiles/test_backtrace.dir/common/test_backtrace.c.o"

# External object files for target test_backtrace
test_backtrace_EXTERNAL_OBJECTS =

_deps/pmemkv-build/tests/libtest_backtrace.a: _deps/pmemkv-build/tests/CMakeFiles/test_backtrace.dir/common/test_backtrace.c.o
_deps/pmemkv-build/tests/libtest_backtrace.a: _deps/pmemkv-build/tests/CMakeFiles/test_backtrace.dir/build.make
_deps/pmemkv-build/tests/libtest_backtrace.a: _deps/pmemkv-build/tests/CMakeFiles/test_backtrace.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C static library libtest_backtrace.a"
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build/tests && $(CMAKE_COMMAND) -P CMakeFiles/test_backtrace.dir/cmake_clean_target.cmake
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_backtrace.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
_deps/pmemkv-build/tests/CMakeFiles/test_backtrace.dir/build: _deps/pmemkv-build/tests/libtest_backtrace.a

.PHONY : _deps/pmemkv-build/tests/CMakeFiles/test_backtrace.dir/build

_deps/pmemkv-build/tests/CMakeFiles/test_backtrace.dir/clean:
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build/tests && $(CMAKE_COMMAND) -P CMakeFiles/test_backtrace.dir/cmake_clean.cmake
.PHONY : _deps/pmemkv-build/tests/CMakeFiles/test_backtrace.dir/clean

_deps/pmemkv-build/tests/CMakeFiles/test_backtrace.dir/depend:
	cd /home/jiawen/dbms/viper/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jiawen/dbms/viper /home/jiawen/dbms/viper/build/_deps/pmemkv-src/tests /home/jiawen/dbms/viper/build /home/jiawen/dbms/viper/build/_deps/pmemkv-build/tests /home/jiawen/dbms/viper/build/_deps/pmemkv-build/tests/CMakeFiles/test_backtrace.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : _deps/pmemkv-build/tests/CMakeFiles/test_backtrace.dir/depend
