import pathlib

import pytest
from antibody_mcts.pdb import Antibody


@pytest.fixture
def data():
    return pathlib.Path(__file__).parent / "data"

@pytest.fixture
def glucagon_pdb_file(data):
    return data / "1gcn.pdb"

@pytest.fixture
def cr3022_pdb_file(data):
    return data / "relaxed-6w41.pdb"

@pytest.fixture
def cr3022(cr3022_pdb_file):
    return Antibody(pdb=cr3022_pdb_file)

@pytest.fixture
def faspr_executable():
    return pathlib.Path("/Users/wba/repos/FASPR/FASPR")
