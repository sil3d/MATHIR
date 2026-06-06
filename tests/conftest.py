"""
Pytest configuration for the MATHIR test suite.

Registers custom markers so ``-m 'not slow'`` and ``-m 'slow'`` work
without warnings.
"""

# Register custom markers
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (deselect with '-m \"not slow\"')",
    )
