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
CMAKE_SOURCE_DIR = /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild

# Utility rule file for pmemkv-populate.

# Include the progress variables for this target.
include CMakeFiles/pmemkv-populate.dir/progress.make

CMakeFiles/pmemkv-populate: CMakeFiles/pmemkv-populate-complete


CMakeFiles/pmemkv-populate-complete: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-install
CMakeFiles/pmemkv-populate-complete: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-mkdir
CMakeFiles/pmemkv-populate-complete: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-download
CMakeFiles/pmemkv-populate-complete: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-update
CMakeFiles/pmemkv-populate-complete: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-patch
CMakeFiles/pmemkv-populate-complete: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-configure
CMakeFiles/pmemkv-populate-complete: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-build
CMakeFiles/pmemkv-populate-complete: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-install
CMakeFiles/pmemkv-populate-complete: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-test
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Completed 'pmemkv-populate'"
	/usr/bin/cmake -E make_directory /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/CMakeFiles
	/usr/bin/cmake -E touch /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/CMakeFiles/pmemkv-populate-complete
	/usr/bin/cmake -E touch /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-done

pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-install: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-build
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "No install step for 'pmemkv-populate'"
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build && /usr/bin/cmake -E echo_append
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build && /usr/bin/cmake -E touch /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-install

pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Creating directories for 'pmemkv-populate'"
	/usr/bin/cmake -E make_directory /home/jiawen/dbms/viper/build/_deps/pmemkv-src
	/usr/bin/cmake -E make_directory /home/jiawen/dbms/viper/build/_deps/pmemkv-build
	/usr/bin/cmake -E make_directory /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/pmemkv-populate-prefix
	/usr/bin/cmake -E make_directory /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/pmemkv-populate-prefix/tmp
	/usr/bin/cmake -E make_directory /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/pmemkv-populate-prefix/src/pmemkv-populate-stamp
	/usr/bin/cmake -E make_directory /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/pmemkv-populate-prefix/src
	/usr/bin/cmake -E make_directory /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/pmemkv-populate-prefix/src/pmemkv-populate-stamp
	/usr/bin/cmake -E touch /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-mkdir

pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-download: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-gitinfo.txt
pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-download: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Performing download step (git clone) for 'pmemkv-populate'"
	cd /home/jiawen/dbms/viper/build/_deps && /usr/bin/cmake -P /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/pmemkv-populate-prefix/tmp/pmemkv-populate-gitclone.cmake
	cd /home/jiawen/dbms/viper/build/_deps && /usr/bin/cmake -E touch /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-download

pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-update: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Performing update step for 'pmemkv-populate'"
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-src && /usr/bin/cmake -P /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/pmemkv-populate-prefix/tmp/pmemkv-populate-gitupdate.cmake

pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-patch: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "No patch step for 'pmemkv-populate'"
	/usr/bin/cmake -E echo_append
	/usr/bin/cmake -E touch /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-patch

pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-configure: pmemkv-populate-prefix/tmp/pmemkv-populate-cfgcmd.txt
pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-configure: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-update
pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-configure: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "No configure step for 'pmemkv-populate'"
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build && /usr/bin/cmake -E echo_append
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build && /usr/bin/cmake -E touch /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-configure

pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-build: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "No build step for 'pmemkv-populate'"
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build && /usr/bin/cmake -E echo_append
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build && /usr/bin/cmake -E touch /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-build

pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-test: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-install
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "No test step for 'pmemkv-populate'"
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build && /usr/bin/cmake -E echo_append
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-build && /usr/bin/cmake -E touch /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-test

pmemkv-populate: CMakeFiles/pmemkv-populate
pmemkv-populate: CMakeFiles/pmemkv-populate-complete
pmemkv-populate: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-install
pmemkv-populate: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-mkdir
pmemkv-populate: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-download
pmemkv-populate: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-update
pmemkv-populate: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-patch
pmemkv-populate: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-configure
pmemkv-populate: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-build
pmemkv-populate: pmemkv-populate-prefix/src/pmemkv-populate-stamp/pmemkv-populate-test
pmemkv-populate: CMakeFiles/pmemkv-populate.dir/build.make

.PHONY : pmemkv-populate

# Rule to build all files generated by this target.
CMakeFiles/pmemkv-populate.dir/build: pmemkv-populate

.PHONY : CMakeFiles/pmemkv-populate.dir/build

CMakeFiles/pmemkv-populate.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pmemkv-populate.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pmemkv-populate.dir/clean

CMakeFiles/pmemkv-populate.dir/depend:
	cd /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild /home/jiawen/dbms/viper/build/_deps/pmemkv-subbuild/CMakeFiles/pmemkv-populate.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pmemkv-populate.dir/depend

