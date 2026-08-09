from simulate.simulate import domains, reference, simulator, effects, welfare
def test_modules_importable():
    assert domains and reference and simulator and effects and welfare
def test_stealth_definition_documented():
    assert "[[CUE:" in domains.CUE_TEMPLATE
