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
include _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/depend.make

# Include the progress variables for this target.
include _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/flags.make

_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/backend.cpp.o: _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/flags.make
_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/backend.cpp.o: _deps/onetbb-src/src/tbbmalloc/backend.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/backend.cpp.o"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tbbmalloc.dir/backend.cpp.o -c /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbbmalloc/backend.cpp

_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/backend.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tbbmalloc.dir/backend.cpp.i"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbbmalloc/backend.cpp > CMakeFiles/tbbmalloc.dir/backend.cpp.i

_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/backend.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tbbmalloc.dir/backend.cpp.s"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbbmalloc/backend.cpp -o CMakeFiles/tbbmalloc.dir/backend.cpp.s

_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/backref.cpp.o: _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/flags.make
_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/backref.cpp.o: _deps/onetbb-src/src/tbbmalloc/backref.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/backref.cpp.o"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tbbmalloc.dir/backref.cpp.o -c /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbbmalloc/backref.cpp

_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/backref.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tbbmalloc.dir/backref.cpp.i"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbbmalloc/backref.cpp > CMakeFiles/tbbmalloc.dir/backref.cpp.i

_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/backref.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tbbmalloc.dir/backref.cpp.s"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbbmalloc/backref.cpp -o CMakeFiles/tbbmalloc.dir/backref.cpp.s

_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/frontend.cpp.o: _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/flags.make
_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/frontend.cpp.o: _deps/onetbb-src/src/tbbmalloc/frontend.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/frontend.cpp.o"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tbbmalloc.dir/frontend.cpp.o -c /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbbmalloc/frontend.cpp

_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/frontend.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tbbmalloc.dir/frontend.cpp.i"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbbmalloc/frontend.cpp > CMakeFiles/tbbmalloc.dir/frontend.cpp.i

_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/frontend.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tbbmalloc.dir/frontend.cpp.s"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbbmalloc/frontend.cpp -o CMakeFiles/tbbmalloc.dir/frontend.cpp.s

_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/large_objects.cpp.o: _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/flags.make
_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/large_objects.cpp.o: _deps/onetbb-src/src/tbbmalloc/large_objects.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/large_objects.cpp.o"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tbbmalloc.dir/large_objects.cpp.o -c /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbbmalloc/large_objects.cpp

_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/large_objects.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tbbmalloc.dir/large_objects.cpp.i"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbbmalloc/large_objects.cpp > CMakeFiles/tbbmalloc.dir/large_objects.cpp.i

_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/large_objects.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tbbmalloc.dir/large_objects.cpp.s"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbbmalloc/large_objects.cpp -o CMakeFiles/tbbmalloc.dir/large_objects.cpp.s

_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/tbbmalloc.cpp.o: _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/flags.make
_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/tbbmalloc.cpp.o: _deps/onetbb-src/src/tbbmalloc/tbbmalloc.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/tbbmalloc.cpp.o"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tbbmalloc.dir/tbbmalloc.cpp.o -c /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbbmalloc/tbbmalloc.cpp

_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/tbbmalloc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tbbmalloc.dir/tbbmalloc.cpp.i"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbbmalloc/tbbmalloc.cpp > CMakeFiles/tbbmalloc.dir/tbbmalloc.cpp.i

_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/tbbmalloc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tbbmalloc.dir/tbbmalloc.cpp.s"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbbmalloc/tbbmalloc.cpp -o CMakeFiles/tbbmalloc.dir/tbbmalloc.cpp.s

_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/__/tbb/itt_notify.cpp.o: _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/flags.make
_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/__/tbb/itt_notify.cpp.o: _deps/onetbb-src/src/tbb/itt_notify.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/__/tbb/itt_notify.cpp.o"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tbbmalloc.dir/__/tbb/itt_notify.cpp.o -c /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbb/itt_notify.cpp

_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/__/tbb/itt_notify.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tbbmalloc.dir/__/tbb/itt_notify.cpp.i"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbb/itt_notify.cpp > CMakeFiles/tbbmalloc.dir/__/tbb/itt_notify.cpp.i

_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/__/tbb/itt_notify.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tbbmalloc.dir/__/tbb/itt_notify.cpp.s"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbb/itt_notify.cpp -o CMakeFiles/tbbmalloc.dir/__/tbb/itt_notify.cpp.s

# Object files for target tbbmalloc
tbbmalloc_OBJECTS = \
"CMakeFiles/tbbmalloc.dir/backend.cpp.o" \
"CMakeFiles/tbbmalloc.dir/backref.cpp.o" \
"CMakeFiles/tbbmalloc.dir/frontend.cpp.o" \
"CMakeFiles/tbbmalloc.dir/large_objects.cpp.o" \
"CMakeFiles/tbbmalloc.dir/tbbmalloc.cpp.o" \
"CMakeFiles/tbbmalloc.dir/__/tbb/itt_notify.cpp.o"

# External object files for target tbbmalloc
tbbmalloc_EXTERNAL_OBJECTS =

gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc.so.2.1: _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/backend.cpp.o
gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc.so.2.1: _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/backref.cpp.o
gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc.so.2.1: _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/frontend.cpp.o
gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc.so.2.1: _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/large_objects.cpp.o
gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc.so.2.1: _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/tbbmalloc.cpp.o
gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc.so.2.1: _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/__/tbb/itt_notify.cpp.o
gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc.so.2.1: _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/build.make
gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc.so.2.1: _deps/onetbb-src/src/tbbmalloc/def/lin64-tbbmalloc.def
gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc.so.2.1: _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX shared library ../../../../gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc.so"
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tbbmalloc.dir/link.txt --verbose=$(VERBOSE)
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc && $(CMAKE_COMMAND) -E cmake_symlink_library ../../../../gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc.so.2.1 ../../../../gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc.so.2 ../../../../gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc.so

gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc.so.2: gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc.so.2.1
	@$(CMAKE_COMMAND) -E touch_nocreate gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc.so.2

gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc.so: gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc.so.2.1
	@$(CMAKE_COMMAND) -E touch_nocreate gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc.so

# Rule to build all files generated by this target.
_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/build: gnu_9.3_cxx17_64_relwithdebinfo/libtbbmalloc.so

.PHONY : _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/build

_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/clean:
	cd /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc && $(CMAKE_COMMAND) -P CMakeFiles/tbbmalloc.dir/cmake_clean.cmake
.PHONY : _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/clean

_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/depend:
	cd /home/jiawen/dbms/viper/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jiawen/dbms/viper /home/jiawen/dbms/viper/build/_deps/onetbb-src/src/tbbmalloc /home/jiawen/dbms/viper/build /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc /home/jiawen/dbms/viper/build/_deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : _deps/onetbb-build/src/tbbmalloc/CMakeFiles/tbbmalloc.dir/depend

