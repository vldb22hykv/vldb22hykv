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

# Utility rule file for cppstyle.

# Include the progress variables for this target.
include _deps/pmemkv-build/CMakeFiles/cppstyle.dir/progress.make

cppstyle: _deps/pmemkv-build/CMakeFiles/cppstyle.dir/build.make

.PHONY : cppstyle

# Rule to build all files generated by this target.
_deps/pmemkv-build/CMakeFiles/cppstyle.dir/build: cppstyle

.PHONY : _deps/pmemkv-build/CMakeFiles/cppstyle.dir/build

_deps/pmemkv-build/CMakeFiles/cppstyle.dir/clean:
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build && $(CMAKE_COMMAND) -P CMakeFiles/cppstyle.dir/cmake_clean.cmake
.PHONY : _deps/pmemkv-build/CMakeFiles/cppstyle.dir/clean

_deps/pmemkv-build/CMakeFiles/cppstyle.dir/depend:
	cd /home/jiawen/dbms/viper/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jiawen/dbms/viper /home/jiawen/dbms/viper/build/_deps/pmemkv-src /home/jiawen/dbms/viper/build /home/jiawen/dbms/viper/build/_deps/pmemkv-build /home/jiawen/dbms/viper/build/_deps/pmemkv-build/CMakeFiles/cppstyle.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : _deps/pmemkv-build/CMakeFiles/cppstyle.dir/depend

