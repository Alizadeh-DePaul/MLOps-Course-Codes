"""Tiny sample module used to demonstrate pre-commit hooks.

This file is intentionally a little messy so the default hooks have something
to flag and fix. Don't edit it manually — let the hooks fix it for you.
"""

import os   
import sys


def greet(name):
    """Print a greeting."""
    print("hello, " + name)   


def add(a, b):
    return a + b

# Note: no trailing newline at end of file — end-of-file-fixer will add one.