set -uo pipefail
env JAX_PLATFORMS=cpu $PY -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}'); sys.exit(0 if b=='gpu' else 42)"
rc=$?
echo cpu_preflight_rc=$rc
[ $rc -eq 42 ]
