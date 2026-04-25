import sys
sys.path.insert(0, "prevaluation_env")
from server.guards import GuardSuite

suite = GuardSuite()
truth = {"true_fix_keywords": ["null", "check"]}

# Test 1: clean action, no guards should fire
action_clean = {
    "classification": "bug",
    "bug_line": 42,
    "team": "webdev",
    "suggested_fix": "Add a null check before dereferencing the pointer",
}
reward, results = suite.evaluate(action_clean, truth, 0.85, elapsed_ms=350.0)
assert all(not r.triggered for r in results), [r for r in results if r.triggered]
assert reward == 0.85
print("Test 1 passed: clean action, reward=", reward)

# Test 2: keyword stuffing (3 words, 2 keywords → density=0.67)
action_stuff = {
    "classification": "bug",
    "bug_line": 1,
    "team": "aiml",
    "suggested_fix": "null check pointer",
}
reward2, results2 = suite.evaluate(action_stuff, truth, 0.9, elapsed_ms=350.0)
kw = next(r for r in results2 if r.guard_name == "KeywordStuffingDetector")
print(f"Test 2 KW stuffing: triggered={kw.triggered}, penalty={kw.penalty:.2f}, reward={reward2:.4f}")

# Test 3: fix too short
action_short = {
    "classification": "feature",
    "bug_line": None,
    "team": None,
    "suggested_fix": "fix it",
}
reward3, results3 = suite.evaluate(action_short, truth, 0.7, elapsed_ms=350.0)
fq = next(r for r in results3 if r.guard_name == "FixQualityValidator")
assert fq.triggered, fq.reason
print(f"Test 3 short fix: triggered={fq.triggered}, penalty={fq.penalty}, reason={fq.reason!r}")

# Test 4: repetition penalty (same fingerprint 6 times, fires at 4th)
action_rep = {
    "classification": "duplicate",
    "bug_line": 10,
    "team": "devops",
    "suggested_fix": "Use the existing handler instead of creating a new one",
}
for _ in range(5):
    suite.evaluate(action_rep, truth, 0.6, elapsed_ms=300.0)
reward4, results4 = suite.evaluate(action_rep, truth, 0.6, elapsed_ms=300.0)
rep = next(r for r in results4 if r.guard_name == "RepetitionDetector")
print(f"Test 4 repetition: triggered={rep.triggered}, penalty={rep.penalty:.2f}")

# Test 5: timing guard (50ms < 200ms threshold)
reward5, results5 = suite.evaluate(action_clean, truth, 0.8, elapsed_ms=50.0)
tg = next(r for r in results5 if r.guard_name == "TimingGuard")
assert tg.triggered, tg.reason
print(f"Test 5 timing: triggered={tg.triggered}, penalty={tg.penalty}, reason={tg.reason!r}")

# Test 6: all-uppercase fix
action_caps = {
    "classification": "bug",
    "bug_line": 5,
    "team": "webdev",
    "suggested_fix": "NULL CHECK POINTER FIX",
}
reward6, results6 = suite.evaluate(action_caps, {}, 0.75, elapsed_ms=300.0)
fq6 = next(r for r in results6 if r.guard_name == "FixQualityValidator")
print(f"Test 6 all-caps: triggered={fq6.triggered}, reason={fq6.reason!r}")

# Test 7: empty fix → no penalty from FixQualityValidator or KeywordStuffingDetector
action_nofix = {"classification": "bug", "bug_line": 3, "team": "webdev", "suggested_fix": ""}
reward7, results7 = suite.evaluate(action_nofix, truth, 0.5, elapsed_ms=300.0)
fq7 = next(r for r in results7 if r.guard_name == "FixQualityValidator")
kw7 = next(r for r in results7 if r.guard_name == "KeywordStuffingDetector")
assert not fq7.triggered and not kw7.triggered
print(f"Test 7 empty fix: fq_triggered={fq7.triggered}, kw_triggered={kw7.triggered} (both False, correct)")

# Test 8: summary
summary = suite.get_summary()
log = suite.get_audit_log(5)
print(f"Test 8 summary: episodes={summary['total_episodes']}, penalty_rate={summary['penalty_rate']}")
print(f"Test 9 audit log: {len(log)} entries returned")
print()
print("All tests passed.")
