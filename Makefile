.PHONY: test compile

test:
	python -m pytest -q tests/test_voxel_utils.py tests/test_renderer.py tests/test_tensor_env_smoke.py

compile:
	python -m compileall -q geoscout scripts
