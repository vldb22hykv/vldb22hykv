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
include _deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/depend.make

# Include the progress variables for this target.
include _deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/flags.make

_deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/function_replacement.cpp.o: _deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/flags.make
_deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/function_replacement.cpp.o: _deps/onetbb-src/src/tbbmalloc_proxy/function_replacement.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object _deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/function_replacement.cpp.o"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc_proxy && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tbbmalloc_proxy.dir/function_replacement.cpp.o -c /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbbmalloc_proxy/function_replacement.cpp

_deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/function_replacement.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tbbmalloc_proxy.dir/function_replacement.cpp.i"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc_proxy && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbbmalloc_proxy/function_replacement.cpp > CMakeFiles/tbbmalloc_proxy.dir/function_replacement.cpp.i

_deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/function_replacement.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tbbmalloc_proxy.dir/function_replacement.cpp.s"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc_proxy && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbbmalloc_proxy/function_replacement.cpp -o CMakeFiles/tbbmalloc_proxy.dir/function_replacement.cpp.s

_deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/proxy.cpp.o: _deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/flags.make
_deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/proxy.cpp.o: _deps/onetbb-src/src/tbbmalloc_proxy/proxy.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object _deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/proxy.cpp.o"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc_proxy && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tbbmalloc_proxy.dir/proxy.cpp.o -c /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbbmalloc_proxy/proxy.cpp

_deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/proxy.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tbbmalloc_proxy.dir/proxy.cpp.i"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc_proxy && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbbmalloc_proxy/proxy.cpp > CMakeFiles/tbbmalloc_proxy.dir/proxy.cpp.i

_deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/proxy.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tbbmalloc_proxy.dir/proxy.cpp.s"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc_proxy && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbbmalloc_proxy/proxy.cpp -o CMakeFiles/tbbmalloc_proxy.dir/proxy.cpp.s

# Object files for target tbbmalloc_proxy
tbbmalloc_proxy_OBJECTS = \
"CMakeFiles/tbbmalloc_proxy.dir/function_replacement.cpp.o" \
"CMakeFiles/tbbmalloc_proxy.dir/proxy.cpp.o"

# External object files for target tbbmalloc_proxy
tbbmalloc_proxy_EXTERNAL_OBJECTS =

gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc_proxy.so.2.1: _deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/function_replacement.cpp.o
gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc_proxy.so.2.1: _deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/proxy.cpp.o
gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc_proxy.so.2.1: _deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/build.make
gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc_proxy.so.2.1: gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc.so.2.1
gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc_proxy.so.2.1: _deps/onetbb-src/src/tbbmalloc_proxy/def/lin64-proxy.def
gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc_proxy.so.2.1: _deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library ../../../../gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc_proxy.so"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc_proxy && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tbbmalloc_proxy.dir/link.txt --verbose=$(VERBOSE)
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc_proxy && $(CMAKE_COMMAND) -E cmake_symlink_library ../../../../gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc_proxy.so.2.1 ../../../../gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc_proxy.so.2 ../../../../gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc_proxy.so

gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc_proxy.so.2: gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc_proxy.so.2.1
	@$(CMAKE_COMMAND) -E touch_nocreate gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc_proxy.so.2

gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc_proxy.so: gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc_proxy.so.2.1
	@$(CMAKE_COMMAND) -E touch_nocreate gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc_proxy.so

# Rule to build all files generated by this target.
_deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/build: gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc_proxy.so

.PHONY : _deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/build

_deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/clean:
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc_proxy && $(CMAKE_COMMAND) -P CMakeFiles/tbbmalloc_proxy.dir/cmake_clean.cmake
.PHONY : _deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/clean

_deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/depend:
	cd /home/jiawen/dbms/viper/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jiawen/dbms/viper /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbbmalloc_proxy /home/jiawen/dbms/viper/build /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc_proxy /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : _deps/onetbb-build/src/tbbmalloc_proxy/CMakeFiles/tbbmalloc_proxy.dir/depend
