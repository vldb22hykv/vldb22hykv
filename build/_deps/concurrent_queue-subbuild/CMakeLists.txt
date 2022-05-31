# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.16.3)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(concurrent_queue-populate NONE)

include(ExternalProject)
ExternalProject_Add(concurrent_queue-populate
                     "UPDATE_DISCONNECTED" "False" "GIT_REPOSITORY" "https://github.com/cameron314/concurrentqueue.git" "GIT_TAG" "1.0.2"
                    SOURCE_DIR          "/home/jiawen/dbms/viper/build/_deps/concurrent_queue-src"
                    BINARY_DIR          "/home/jiawen/dbms/viper/build/_deps/concurrent_queue-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)
