from norax.context.window import RollingWindow


def test_compact_solved_turns_archives_old_tool_traces_losslessly(tmp_path):
    w = RollingWindow(sleep_dir=tmp_path, protect_tail_turns=1)
    t1 = w.start_turn()
    w.add_user("old task", turn_id=t1)
    w.add_tool_call(call_id="c1", content='{"path":"big"}', turn_id=t1, name="read")
    w.add_tool_result(call_id="c1", content="x" * 4000, turn_id=t1)
    w.add_assistant("done with verified outcome", turn_id=t1)
    t2 = w.start_turn()
    w.add_user("current task", turn_id=t2)

    assert w.compact_solved_turns() == 2
    assert [(f.kind, f.turn_id) for f in w.body] == [
        ("user", t1),
        ("assistant", t1),
        ("user", t2),
    ]
    spills = list(tmp_path.glob("spill-*.jsonl"))
    assert len(spills) == 1
    raw = spills[0].read_text()
    assert "tool_call" in raw
    assert "tool_result" in raw
    assert "x" * 100 in raw


def test_load_sets_next_turn_id(tmp_path):
    p = tmp_path / "w.json"
    w = RollingWindow(sleep_dir=tmp_path)
    t = w.start_turn()
    w.add_user("hello", turn_id=t)
    w.save(p)
    loaded = RollingWindow.load(p)
    assert loaded.start_turn() == t + 1


def test_eviction_keeps_tool_call_result_pairs_atomic(tmp_path):
    w = RollingWindow(sleep_dir=tmp_path, budget_tokens=200, protect_tail_turns=0)
    t = w.start_turn()
    w.add_user("task", turn_id=t)
    w.add_tool_call(call_id="c1", content='{"path":"x"}', turn_id=t, name="read")
    w.add_tool_result(call_id="c1", content="y" * 500, turn_id=t)
    w.add_assistant("done", turn_id=t)
    groups = w._atomic_groups(w.body)
    pair_group = next(g for g in groups if any(f.kind == "tool_call" for f in g))
    kinds = [f.kind for f in pair_group]
    assert kinds == ["tool_call", "tool_result"]
