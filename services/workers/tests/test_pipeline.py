from memoryscapes_workers.pipeline import DAG, topological_order


def test_every_stage_runs_after_its_dependencies():
    order = topological_order()
    assert set(order) == set(DAG)
    for name, (_, deps) in DAG.items():
        assert all(order.index(d) < order.index(name) for d in deps)


def test_audio_branch_does_not_wait_for_splat_training():
    assert "splat" not in DAG["localize"][1]
