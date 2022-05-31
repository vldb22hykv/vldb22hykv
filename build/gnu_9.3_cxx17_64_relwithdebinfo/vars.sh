#!/bin/sh
#
# Copyright (c) 2005-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export TBBROOT=/home/jiawen/dbms/viper/build/_deps/onetbb-src

LD_LIBRARY_PATH="/home/jiawen/dbms/viper/build/gnu_9.3_cxx17_64_relwithdebinfo:${LD_LIBRARY_PATH}"; export LD_LIBRARY_PATH
LIBRARY_PATH="/home/jiawen/dbms/viper/build/gnu_9.3_cxx17_64_relwithdebinfo:${LIBRARY_PATH}"; export LIBRARY_PATH
CPATH="${TBBROOT}/include:${CPATH}"; export CPATH


