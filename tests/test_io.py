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


def test_valid_config_strings(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p1 = d / "hello.yaml"
    p1.write_text(good_config)
    validate_config(p1)
    p2 = d / "hello2.yaml"
    p2.write_text(no_anno_config)
    validate_config(p2)


# --------- 2. Testing VCFs ---------- #
