"""Regression tests for long whitespace-free words in the word DP.

The word-internal segmentation DP used to be a memoized recursion over
grapheme positions, so a single word deeper than the interpreter's
recursion limit (default 1000 frames) crashed planning with a
``RecursionError`` and the revision endpoint answered HTTP 500 without
persisting anything.  The DP is now iterative (back-to-front over the
same memo table); these tests pin the boundary from the incident report:
3163 consecutive ``a`` on a 3163-cell line with an empty rule set must
plan, persist (HTTP 201) and finalize.
"""
import json
import os
import pathlib
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from app.planner import plan
from app.server import build_server, get_store

# Deeper than any recursion limit this test suite may run under; 3163 is
# the exact length from the incident report.
LONG = max(3163, sys.getrecursionlimit() * 3)


def all_cells(result):
    out = []
    for pg in result["pages"]:
        for ln in pg["lines"]:
            out.extend(ln["cells"])
    return out


class LongWordPlanTests(unittest.TestCase):
    def test_long_word_no_rules_plans_ok(self):
        # The reported case: 3163 consecutive 'a', cells_per_line=3163,
        # empty rules.  Recursion depth 3163 > default limit 1000.
        self.assertGreater(LONG, sys.getrecursionlimit())
        r = plan({"body": "a" * LONG,
                  "paper": {"cells_per_line": LONG, "lines_per_page": 25},
                  "rules": {}})
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["page_count"], 1)
        line = r["pages"][0]["lines"][0]["cells"]
        self.assertEqual(len(line), LONG)
        # grade-1 'a' is dot 1; grapheme mapping stays exact per cell
        self.assertTrue(all(c["dots"] == "1" for c in line))
        for i, c in enumerate(line):
            self.assertEqual((c["grapheme_start"], c["grapheme_end"]),
                             (i, i + 1))
            self.assertEqual((c["char_start"], c["char_end"]), (i, i + 1))
            self.assertEqual(c["col"], i + 1)

    def test_long_word_with_rules_plans_ok(self):
        # Rule matching must keep working at recursion-limit depths: every
        # 'ing' contracts to one cell (dots 346).  A word is one atom, so
        # the line must fit the contracted width (n_rep cells).
        n_rep = sys.getrecursionlimit()  # 3 * n_rep graphemes total
        r = plan({"body": "ing" * n_rep,
                  "paper": {"cells_per_line": n_rep, "lines_per_page": 25},
                  "rules": {"rules": [
                      {"id": "ing", "pattern": "ing", "cells": ["346"]}]}})
        self.assertEqual(r["status"], "ok")
        cells = all_cells(r)
        self.assertEqual(len(cells), n_rep)
        self.assertTrue(all(c["rule_id"] == "ing" for c in cells))

    def test_long_digit_run_plans_ok(self):
        # One number sign, then one cell per digit.
        n = LONG
        r = plan({"body": "7" * n,
                  "paper": {"cells_per_line": n + 1, "lines_per_page": 2}})
        self.assertEqual(r["status"], "ok")
        cells = all_cells(r)
        self.assertEqual(len(cells), n + 1)  # number sign + n digits
        self.assertEqual(cells[0]["unicode"], "⠼")
        self.assertEqual(
            sum(1 for c in cells if c["unicode"] == "⠼"), 1)


class LongWordEndpointTests(unittest.TestCase):
    """End-to-end: the revision endpoint must persist and allow finalize."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        import app.server as server
        from app import config as cfg
        cfg.DATA_DIR = pathlib.Path(self.dir)
        cfg.DB_PATH = os.path.join(self.dir, "t.db")
        server._STORE = None
        self.store = get_store()
        self.server = build_server(port=0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _post(self, path, obj):
        data = json.dumps(obj).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_revision_created_and_finalizable(self):
        payload = {"body": "a" * LONG,
                   "paper": {"cells_per_line": LONG, "lines_per_page": 25},
                   "rules": {}}
        status, rep = self._post("/documents/doc-long/revisions", payload)
        self.assertEqual(status, 201)
        self.assertEqual(rep["revision_no"], 1)
        self.assertEqual(rep["status"], "ok")
        self.assertTrue(rep["finalizable"])
        self.assertEqual(rep["layout"]["page_count"], 1)

        # persisted: the revision survives a fresh read
        rev = self.store.get_revision("doc-long", 1)
        self.assertEqual(rev["status"], "ok")
        self.assertEqual(rev["request"]["body"], "a" * LONG)

        # finalizes to immutable artifacts
        status, fin = self._post("/documents/doc-long/revisions/1/finalize",
                                 {})
        self.assertEqual(status, 201)
        self.assertTrue(fin["finalized"])
        info = self.store.get_finalization("doc-long", 1)
        self.assertEqual(info["summary_sha256"], fin["summary_sha256"])


if __name__ == "__main__":
    unittest.main()
