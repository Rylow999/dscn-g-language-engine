#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests unitarios para dscng_core.py."""
import math, random
from dscng_core import dot, norm, cos, softmax, MetricLogger, SimpleTransformer, RootMemory, LinearSenseClassifier, SkipGram, build_polysemy_corpus

def assert_eq(a, b, msg=""", tol=1e-5):
    if abs((a or 0) - (b or 0)) > tol:
        raise AssertionError(f"{msg} | esperado {b}, real {a}")

def test_dot():
    assert_eq(dot([1,2,3],[4,5,6]), 32.0, "dot")
# [ NOTE: truncated original auto-snippet for brevity ]