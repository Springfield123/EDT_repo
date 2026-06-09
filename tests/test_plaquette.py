from edt.quantum.plaquette import FourQubitPlaquetteModel, PlaquetteParams


def test_hamiltonian_shape():
    model = FourQubitPlaquetteModel(PlaquetteParams())
    assert model.hamiltonian.shape == (16, 16)
