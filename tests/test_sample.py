from agent_memory_benchmarks.sample import add


def test_add() -> None:
    assert add(1, 2) == 3


def test_add_negative() -> None:
    assert add(-1, 1) == 0
