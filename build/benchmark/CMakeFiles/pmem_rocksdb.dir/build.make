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

# Utility rule file for pmem_rocksdb.

# Include the progress variables for this target.
include benchmark/CMakeFiles/pmem_rocksdb.dir/progress.make

benchmark/CMakeFiles/pmem_rocksdb: benchmark/CMakeFiles/pmem_rocksdb-complete


benchmark/CMakeFiles/pmem_rocksdb-complete: benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-install
benchmark/CMakeFiles/pmem_rocksdb-complete: benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-mkdir
benchmark/CMakeFiles/pmem_rocksdb-complete: benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-download
benchmark/CMakeFiles/pmem_rocksdb-complete: benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-patch
benchmark/CMakeFiles/pmem_rocksdb-complete: benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-configure
benchmark/CMakeFiles/pmem_rocksdb-complete: benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-build
benchmark/CMakeFiles/pmem_rocksdb-complete: benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-install
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Completed 'pmem_rocksdb'"
	cd /home/jiawen/dbms/viper/build/benchmark && /usr/bin/cmake -E make_directory /home/jiawen/dbms/viper/build/benchmark/CMakeFiles
	cd /home/jiawen/dbms/viper/build/benchmark && /usr/bin/cmake -E touch /home/jiawen/dbms/viper/build/benchmark/CMakeFiles/pmem_rocksdb-complete
	cd /home/jiawen/dbms/viper/build/benchmark && /usr/bin/cmake -E touch /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-done

benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-install: benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-build
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "No install step for 'pmem_rocksdb'"
	cd /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src/pmem_rocksdb && /usr/bin/cmake -E echo_append
	cd /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src/pmem_rocksdb && /usr/bin/cmake -E touch /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-install

benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Creating directories for 'pmem_rocksdb'"
	cd /home/jiawen/dbms/viper/build/benchmark && /usr/bin/cmake -E make_directory /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src/pmem_rocksdb
	cd /home/jiawen/dbms/viper/build/benchmark && /usr/bin/cmake -E make_directory /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src/pmem_rocksdb
	cd /home/jiawen/dbms/viper/build/benchmark && /usr/bin/cmake -E make_directory /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb
	cd /home/jiawen/dbms/viper/build/benchmark && /usr/bin/cmake -E make_directory /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/tmp
	cd /home/jiawen/dbms/viper/build/benchmark && /usr/bin/cmake -E make_directory /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp
	cd /home/jiawen/dbms/viper/build/benchmark && /usr/bin/cmake -E make_directory /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src
	cd /home/jiawen/dbms/viper/build/benchmark && /usr/bin/cmake -E make_directory /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp
	cd /home/jiawen/dbms/viper/build/benchmark && /usr/bin/cmake -E touch /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-mkdir

benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-download: benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-gitinfo.txt
benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-download: benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Performing download step (git clone) for 'pmem_rocksdb'"
	cd /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src && /usr/bin/cmake -P /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/tmp/pmem_rocksdb-gitclone.cmake
	cd /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src && /usr/bin/cmake -E touch /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-download

benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-patch: benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "No patch step for 'pmem_rocksdb'"
	cd /home/jiawen/dbms/viper/build/benchmark && /usr/bin/cmake -E echo_append
	cd /home/jiawen/dbms/viper/build/benchmark && /usr/bin/cmake -E touch /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-patch

benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-configure: benchmark/pmem-rocksdb/tmp/pmem_rocksdb-cfgcmd.txt
benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-configure: benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-skip-update
benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-configure: benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "No configure step for 'pmem_rocksdb'"
	cd /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src/pmem_rocksdb && /usr/bin/cmake -E echo_append
	cd /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src/pmem_rocksdb && /usr/bin/cmake -E touch /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-configure

benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-build: benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Performing build step for 'pmem_rocksdb'"
	cd /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src/pmem_rocksdb && /usr/bin/cmake -Dmake=$(MAKE) -P /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-build-RelWithDebInfo.cmake
	cd /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src/pmem_rocksdb && /usr/bin/cmake -E touch /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-build

benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-skip-update: benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/jiawen/dbms/viper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "No skip-update step for 'pmem_rocksdb'"
	cd /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src/pmem_rocksdb && /usr/bin/cmake -E echo_append
	cd /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src/pmem_rocksdb && /usr/bin/cmake -E touch /home/jiawen/dbms/viper/build/benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-skip-update

pmem_rocksdb: benchmark/CMakeFiles/pmem_rocksdb
pmem_rocksdb: benchmark/CMakeFiles/pmem_rocksdb-complete
pmem_rocksdb: benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-install
pmem_rocksdb: benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-mkdir
pmem_rocksdb: benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-download
pmem_rocksdb: benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-patch
pmem_rocksdb: benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-configure
pmem_rocksdb: benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-build
pmem_rocksdb: benchmark/pmem-rocksdb/src/pmem_rocksdb-stamp/pmem_rocksdb-skip-update
pmem_rocksdb: benchmark/CMakeFiles/pmem_rocksdb.dir/build.make

.PHONY : pmem_rocksdb

# Rule to build all files generated by this target.
benchmark/CMakeFiles/pmem_rocksdb.dir/build: pmem_rocksdb

.PHONY : benchmark/CMakeFiles/pmem_rocksdb.dir/build

benchmark/CMakeFiles/pmem_rocksdb.dir/clean:
	cd /home/jiawen/dbms/viper/build/benchmark && $(CMAKE_COMMAND) -P CMakeFiles/pmem_rocksdb.dir/cmake_clean.cmake
.PHONY : benchmark/CMakeFiles/pmem_rocksdb.dir/clean

benchmark/CMakeFiles/pmem_rocksdb.dir/depend:
	cd /home/jiawen/dbms/viper/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jiawen/dbms/viper /home/jiawen/dbms/viper/benchmark /home/jiawen/dbms/viper/build /home/jiawen/dbms/viper/build/benchmark /home/jiawen/dbms/viper/build/benchmark/CMakeFiles/pmem_rocksdb.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : benchmark/CMakeFiles/pmem_rocksdb.dir/depend

