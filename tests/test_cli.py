import json
import subprocess
import sys


def run(cwd, *args):
    done = subprocess.run([sys.executable, "-m", "fastrsi.cli", *args], cwd=cwd,
                          capture_output=True, text=True, timeout=60)
    return done.returncode, done.stdout, done.stderr


def test_demo(tmp_path):
    code, out, err = run(tmp_path, "demo")
    assert code == 0, err
    assert "output     changed" in out
    assert "output restored" in out
    assert "refused" in out


def test_lifecycle(tmp_path):
    patch = tmp_path / "fix.patch"
    patch.write_text("--- a/toymodel.py\n+++ b/toymodel.py\n@@ -12,1 +12,1 @@\n"
                     "-    return x * 2 + 1\n+    return x * 3 + 1\n")
    bad = tmp_path / "bad.patch"
    bad.write_text("--- a/toymodel.py\n+++ b/toymodel.py\n@@ -11,1 +11,1 @@\n"
                   "-def forward(x):\n+def forward(x, y=0):\n")
    assert run(tmp_path, "start", "toy")[0] == 0
    try:
        before = json.loads(run(tmp_path, "exercise")[1])["digest"]
        assert run(tmp_path, "install", str(bad))[0] == 2
        code, out, err = run(tmp_path, "install", str(patch))
        assert code == 0, out + err
        assert json.loads(run(tmp_path, "exercise")[1])["digest"] != before
        status = json.loads(run(tmp_path, "status")[1])
        assert status["state"] == "edited" and status["proven"]
        code, out, err = run(tmp_path, "fresh", str(patch))
        assert code == 0 and json.loads(out)["same"], out + err
        assert run(tmp_path, "undo")[0] == 0
        assert json.loads(run(tmp_path, "status")[1])["state"] == "base"
    finally:
        run(tmp_path, "stop")
