#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

set -e
set -u
set -o pipefail

CMAKE_VERSION="3.31.10"
CMAKE_SHA256="cf06fadfd6d41fa8e1ade5099e54976d1d844fd1487ab99942341f91b13d3e29"

# parse argument if provided
CMAKE_VERSION=${1:-$CMAKE_VERSION}
CMAKE_SHA256=${2:-$CMAKE_SHA256}

v=$(echo $CMAKE_VERSION | sed 's/\(.*\)\..*/\1/g')
echo "Installing cmake $CMAKE_VERSION ($v)"
wget https://cmake.org/files/v${v}/cmake-${CMAKE_VERSION}.tar.gz
echo "$CMAKE_SHA256" cmake-${CMAKE_VERSION}.tar.gz | sha256sum -c
tar xvf cmake-${CMAKE_VERSION}.tar.gz
pushd cmake-${CMAKE_VERSION}
  ./bootstrap
  make -j$(nproc)
  make install
popd
rm -rf cmake-${CMAKE_VERSION} cmake-${CMAKE_VERSION}.tar.gz
