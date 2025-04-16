import pathlib

import pytest

from antibody_mcts.mcts import MCTS
from antibody_mcts.pdb import Antibody

@pytest.fixture
def data():
    return pathlib.Path(__file__).parent / "data"

@pytest.fixture
def cr3022_pdb_file(data):
    return data / "relaxed-6w41.pdb"

@pytest.fixture
def faspr_executable():
    return pathlib.Path("/Users/wba/repos/FASPR/FASPR")

@pytest.fixture
def mcts_factory(data, faspr_executable):
    return lambda: MCTS(iterations=1, depth=1, mutations_dir=data, faspr_executable=faspr_executable)

def test_repeated_dumps_diff(mcts_factory, cr3022_pdb_file):
    # Update state
    mcts = mcts_factory()
    mcts.fname_to_state[cr3022_pdb_file.name] = Antibody(pdb=cr3022_pdb_file)
    
    assert mcts.dumps_diff() == [{"filename": cr3022_pdb_file.name, "total_score": 0, "visits": 0}]
    assert mcts.dumps_diff() == []

def test_dumps_loads_diff(mcts_factory, cr3022_pdb_file):
    "mcts1.dumps -> mcts2.loads -> same -> mcts2.dumps -> mcts1.loads -> same"
    mcts1 = mcts_factory()
    mcts2 = mcts_factory()

    ab = Antibody(pdb=cr3022_pdb_file)
    mcts1.fname_to_state[cr3022_pdb_file.name] = ab
    mcts1.bktree.add(ab)

    assert mcts2.fname_to_state == {} != mcts1.fname_to_state
    dump1 = mcts1.dumps_diff()
    mcts2.loads_diff(dump1)
    assert mcts1.fname_to_state == mcts2.fname_to_state != {}
    assert len(list(mcts1.bktree))
    assert sorted(mcts1.bktree) == sorted(mcts2.bktree)

def test_use_same_antibody_instance(mcts_factory, cr3022_pdb_file):
    "Test we're not creating new Antibody instances if they already exist"
    mcts1 = mcts_factory()
    mcts2 = mcts_factory()

    ab1 = Antibody(pdb=cr3022_pdb_file, total_score=1e-10, visits=1)
    mcts1.fname_to_state[cr3022_pdb_file.name] = ab1
    mcts1.bktree.add(ab1)
    ab2 = Antibody(pdb=cr3022_pdb_file)
    mcts2.fname_to_state[cr3022_pdb_file.name] = ab2
    mcts2.bktree.add(ab2)

    dump1 = mcts1.dumps_diff()
    mcts2.loads_diff(dump1)
    cr3022 = mcts2.fname_to_state[cr3022_pdb_file.name]
    assert cr3022 is ab2
    assert cr3022.total_score == 1e-10
    assert cr3022.visits == 1
  