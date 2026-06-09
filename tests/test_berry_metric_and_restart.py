import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    path = scripts / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_constant_eigenvectors_have_zero_berry_and_metric():
    surfaces = load_script("wannier_berry_metric_surfaces")
    evecs = np.zeros((2, 2, 2, 2), dtype=complex)
    evecs[:, :, :, :] = np.eye(2)[None, None, :, :]

    berry, metric = surfaces.compute_berry_metric(evecs, 0.1, 0.1)

    assert np.allclose(berry, 0.0)
    assert np.allclose(metric, 0.0)


def test_restart_input_enables_epw_restart_and_a2f():
    restart = load_script("create_epw_restart_narval")
    base = """&inputepw
  prefix = 'pwscf',
  epbread = .false.,
  phonselfen = .false.,
  nqf1 = 1, nqf2 = 1, nqf3 = 1,
/
"""

    text = restart.render_restart_input(base, nqf=(4, 4, 4), nkf=None)

    assert "epbread = .true." in text
    assert "epwread = .true." in text
    assert "phonselfen = .true." in text
    assert "a2f = .true." in text
    assert "nqf1 = 4" in text
    assert "nqf2 = 4" in text
    assert "nqf3 = 4" in text


def test_restart_submit_uses_custom_input_name():
    restart = load_script("create_epw_restart_narval")

    text = restart.render_submit("K3C60", ntasks=4, time_limit="01:00:00", account="def-test", restart_name="epw_epc_restart.in")

    assert "EPW_INPUT=\"${EPW_INPUT:-epw_epc_restart.in}\"" in text
