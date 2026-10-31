import pytest

from fastrsi.conformance import Suite
from fastrsi.engines.toy import Toy


@pytest.fixture(scope="module")
def record(tmp_path_factory):
    return Suite(Toy, tmp_path_factory.mktemp("c")).run()


@pytest.mark.parametrize("case", Suite.CASES)
def test_case(record, case):
    assert record["cases"][case]["passed"], record["cases"][case].get("error")


def test_long_workdir(tmp_path):
    """Socket paths have a low length limit; a deep workdir must still work."""
    deep = tmp_path / ("d" * 60) / ("e" * 60)
    record = Suite(Toy, deep).run(only={"c1_noop"})
    assert record["passed"], record
