import pytest

from fastrsi.edits import Edit, FileEdit, Refused, classify, classify_file

BASE = """X = 1


def f(a):
    return a + 1


@decorate
def g(a):
    return a


class C:
    limit = 2

    def m(self):
        return 1

    @staticmethod
    def s():
        return 1
"""


def changed(old, new):
    return classify_file(BASE, BASE.replace(old, new))


def refused(old, new):
    with pytest.raises(Refused) as caught:
        changed(old, new)
    return caught.value.code


def test_body_changes():
    assert changed("return a + 1", "return a + 2") == ["f"]
    assert classify_file(BASE, BASE.replace("def m(self):\n        return 1",
                                            "def m(self):\n        return 2")) == ["C.m"]
    assert changed("def s():\n        return 1", "def s():\n        return 2") == ["C.s"]


def test_comment_only_changes_nothing():
    assert changed("return a + 1", "return a + 1  # note") == []


@pytest.mark.parametrize("old,new,code", [
    ("def f(a):", "def f(a, b=0):", "signature"),
    ("X = 1", "X = 2", "module-level"),
    ("limit = 2", "limit = 3", "module-level"),
    ("def f(a):", "def h():\n    return 0\n\n\ndef f(a):", "new-definition"),
    ("    return a\n", "    return a + 1\n", "decorated"),
    ("return a + 1", "return (", "syntax"),
])
def test_refusals(old, new, code):
    assert refused(old, new) == code


def test_removed_definition():
    with pytest.raises(Refused) as caught:
        classify_file(BASE, BASE.replace("def f(a):\n    return a + 1\n", ""))
    assert caught.value.code == "removed-definition"


def test_no_change_is_refused():
    with pytest.raises(Refused) as caught:
        classify(Edit((FileEdit("a.py", BASE, BASE),)))
    assert caught.value.code == "no-change"


def test_native_file_has_no_route():
    with pytest.raises(Refused) as caught:
        classify(Edit((FileEdit("a.c", "x", "y"),)))
    assert caught.value.code == "not-python"
