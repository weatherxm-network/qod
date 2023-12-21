#!/bin/sh

set -e

# activate our virtual environment here
. /opt/pysetup/.venv/bin/activate

# You can put other setup logic here

# Run the selected driver
python -m "$@"
