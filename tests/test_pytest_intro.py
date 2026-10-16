import pytest


def add(a, b):
    return a + b


def test_add():
    assert add(2, 3) == 5


@pytest.mark.parametrize("a, b, expected", [(1, 1, 2), (0, 5, 5), (-2, 2, 0)])
def test_add_many(a, b, expected):
    assert add(a, b) == expected


def test_error_is_raised():
    with pytest.raises(ZeroDivisionError):
        1 / 0


def test_temp_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hi")
    assert f.read_text() == "hi"