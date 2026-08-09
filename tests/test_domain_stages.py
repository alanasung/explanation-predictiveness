from simulate.stages import STAGES
def test_stage_names():
    assert list(STAGES)==["domains","reference","simulator","effects","welfare"]
def test_order_domains_first():
    assert list(STAGES)[0]=="domains"
def test_callable():
    assert all(callable(v) for v in STAGES.values())
def test_no_notimplemented():
    import inspect, simulate.stages as s
    assert "NotImplementedError" not in inspect.getsource(s)
