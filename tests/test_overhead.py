from fastrsi import overhead
from fastrsi.engines.toy import Toy


def test_overhead_record(tmp_path):
    record = overhead.run(Toy, tmp_path / "o", rounds=1, repeats=3, warmup=1, cycles=3)
    assert record["stock_vs_agent"]["exercise_s"]["stock"]["n"] == 3
    session = record["session"]
    assert session["install_s"]["n"] == 3
    assert session["detect_hang_s"] >= 2.9
    assert session["witness_count_s"]["n"] == 5
