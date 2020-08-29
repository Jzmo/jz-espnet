#!/bin/bash
set -euo pipefail

if [ -z "${PS1:-}" ]; then
    PS1=__dummy__
fi
CONDA_URL=https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

if [ $# -eq 0 ] || [ $# -gt 4 ]; then
    echo "Usage: $0 <output> [conda-env-name] [python-version>]"
    exit 1;
elif [ $# -eq 3 ]; then
    output_dir="$1"
    name="$2"
    PYTHON_VERSION="$3"
elif [ $# -eq 2 ]; then
    output_dir="$1"
    name="$2"
    PYTHON_VERSION=""
elif [ $# -eq 1 ]; then
    output_dir="$1"
    name=""
    PYTHON_VERSION=""
fi

PYTHON_VERSION=3.7

if [ ! -e "${output_dir}/etc/profile.d/conda.sh" ]; then
    if [ ! -e miniconda.sh ]; then
        wget --tries=3 "${CONDA_URL}" -O miniconda.sh
    fi

    bash miniconda.sh -b -p "${output_dir}"
fi

source "${output_dir}/etc/profile.d/conda.sh"
conda deactivate

# If the env already exists, skip recreation
if [ -n "${name}" ] && ! conda activate ${name}; then
    conda create -yn "${name}"
fi
conda activate ${name}

if [ -n "${PYTHON_VERSION}" ]; then
    conda install -y conda "python=${PYTHON_VERSION}"
else
    conda install -y conda
fi

conda install -y pip setuptools

cat << EOF > activate_python.sh
if [ -z "\${PS1:-}" ]; then
    PS1=__dummy__
fi
. $(cd ${output_dir}; pwd)/etc/profile.d/conda.sh && conda deactivate && conda activate ${name}
EOF
