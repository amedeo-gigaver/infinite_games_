#!/bin/bash
set -euo pipefail

# Make sure these environment variables are set before running the script
: "${MNEMONIC_COLDKEY:?Must set MNEMONIC_COLDKEY}"
: "${MNEMONIC_HOTKEY:?Must set MNEMONIC_HOTKEY}"

# -- Regenerate coldkey - do not remove the empty lines between the EOF!!!
btcli w regen_coldkey --mnemonic "${MNEMONIC_COLDKEY}" <<EOF

validator
EOF

# -- Regenerate hotkey
btcli w regen_hotkey --mnemonic "${MNEMONIC_HOTKEY}" <<EOF

validator

EOF

echo "Coldkey and hotkey regenerated successfully."
