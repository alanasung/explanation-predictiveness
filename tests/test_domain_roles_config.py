from pathlib import Path
def test_pilot_yaml_has_roles_and_domain_stages():
    text = (Path(__file__).resolve().parents[1]/"configs"/"experiment"/"pilot.yaml").read_text()
    assert "roles:" in text
    assert "domains" in text and "reference" in text
    assert "welfare" in text
def test_domains_before_reference_in_yaml():
    text = (Path(__file__).resolve().parents[1]/"configs"/"experiment"/"pilot.yaml").read_text()
    stages = text.split("stages:")[1].split("\n")[0]
    assert stages.index("domains") < stages.index("reference")
