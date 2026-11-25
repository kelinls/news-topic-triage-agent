"""Compatibility entry point for the offline tests."""

from runpy import run_module


if __name__ == "__main__":
    run_module("tests.test_experiment", run_name="__main__", alter_sys=True)
