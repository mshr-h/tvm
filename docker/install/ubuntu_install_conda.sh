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

MINICONDA_FILENAME="Miniconda3-py313_25.11.1-1-Linux-x86_64.sh"
MINICONDA_SHA256="e0b10e050e8928e2eb9aad2c522ee3b5d31d30048b8a9997663a8a460d538cef"

# parse argument if provided
MINICONDA_FILENAME=${1:-$MINICONDA_FILENAME}
MINICONDA_SHA256=${2:-$MINICONDA_SHA256}

MINICONDA_INSTALL_PATH="/opt/conda"

cd /tmp && wget -q https://repo.anaconda.com/miniconda/${MINICONDA_FILENAME}
echo "$MINICONDA_SHA256" ${MINICONDA_FILENAME} | sha256sum -c
chmod +x ${MINICONDA_FILENAME}
/tmp/${MINICONDA_FILENAME} -b -p ${MINICONDA_INSTALL_PATH}
rm /tmp/${MINICONDA_FILENAME}
${MINICONDA_INSTALL_PATH}/bin/conda upgrade --all
${MINICONDA_INSTALL_PATH}/bin/conda clean -ya
${MINICONDA_INSTALL_PATH}/bin/conda install conda-build conda-verify
chmod -R a+w ${MINICONDA_INSTALL_PATH}
