import pytest
import yaml

from pGermlinePoly.io import validate_config

# -------- Input Test Data -------- #


# -------- 1. Testing Configs --------- #
good_config = """
ind: JH214UJ
age: 50
sex: M
germline:
  - JH214UJbulk
clones:
  - JH214UJG11
  - JH214UJG2
annotations:
  - MLEAF
"""

no_anno_config = """
ind: JH214UJ
age: 50
sex: M
germline:
  - JH214UJbulk
clones:
  - JH214UJG11
  - JH214UJG2
annotations: []
"""

bad_sex_config = """
ind: JH214UJ
age: 50
sex: X
germline:
  - JH214UJbulk
clones:
  - JH214UJG11
  - JH214UJG2
annotations:
  - MLEAF
"""

no_sex_config = """
ind: JH214UJ
age: 50
germline:
  - JH214UJbulk
clones:
  - JH214UJG11
  - JH214UJG2
annotations:
  - MLEAF
"""


def test_valid_config_strings(tmp_path):
    """Testing valid configurations."""
    d = tmp_path / "valid"
    d.mkdir()
    p1 = d / "hello.yaml"
    p1.write_text(good_config)
    validate_config(p1)
    p2 = d / "hello2.yaml"
    p2.write_text(no_anno_config)
    validate_config(p2)


def test_bad_config_strings(tmp_path):
    """Testing bad configs in various cases."""
    d = tmp_path / "invalid"
    d.mkdir()
    for c_str in [bad_sex_config, no_sex_config]:
        p1 = d / "hello.yaml"
        p1.write_text(c_str)
        with pytest.raises(Exception):
            validate_config(p1)


# --------- 2. Testing VCFs ---------- #
