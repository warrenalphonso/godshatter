from collections import defaultdict

import pytest
from antibody_mcts.distributed import LocalMessageTransport, LocalPDBStore, MCTSCoordinator, MCTSWorker, Topic
from antibody_mcts.mcts import MCTS


@pytest.fixture
def transport():
    return LocalMessageTransport()

@pytest.fixture
def remote_dir(tmp_path):
    p = tmp_path / "remote"
    p.mkdir()
    return p

@pytest.fixture
def pdb_store(remote_dir):
    return LocalPDBStore(base_dir=remote_dir)

@pytest.fixture
def coordinator(transport, pdb_store):
    return MCTSCoordinator(transport=transport, pdb_store=pdb_store)

@pytest.fixture
def starting_pdb_fname(cr3022_pdb_file):
    return cr3022_pdb_file.name

@pytest.fixture
def local_dir(tmp_path):
    p = tmp_path / "local"
    p.mkdir()
    return p

@pytest.fixture
def mcts_factory(local_dir, faspr_executable):
    return lambda: MCTS(depth=1, faspr_executable=faspr_executable, mutations_dir=local_dir)

def test_distributed_run(transport, pdb_store, mcts_factory, coordinator, starting_pdb_fname, cr3022_pdb_file):
    worker_ready = []
    job = []
    job_complete = []
    diff = []
    transport.subscribe(Topic.WORKER_READY, "test", lambda message: worker_ready.append(message.payload.copy()))
    transport.subscribe(Topic.NEW_JOB, "test", lambda message: job.append(message.payload.copy()))
    transport.subscribe(Topic.JOB_COMPLETE, "test", lambda message: job_complete.append(message.payload.copy()))
    transport.subscribe(Topic.DIFF, "test", lambda message: diff.append(message.payload.copy()))

    assert len(list(pdb_store.base_dir.iterdir())) == 0

    coordinator.start()
    worker0 = MCTSWorker("worker0", transport, pdb_store, mcts_factory)
    worker0.start()
    worker1 = MCTSWorker("worker1", transport, pdb_store, mcts_factory)
    worker1.start()
    assert worker_ready == [{"worker_id": "worker0"}, {"worker_id": "worker1"}]

    pdb_store.store_pdb(fname=starting_pdb_fname, content=cr3022_pdb_file.read_bytes())

    coordinator.run_distributed(starting_pdb_fname, total_iterations=6, iterations_per_job=2)
    assert job == [
        {"job_id": "job-0", "pdb_id": starting_pdb_fname, "iterations": 2, "target_worker": "worker0"},
        {"job_id": "job-1", "pdb_id": starting_pdb_fname, "iterations": 2, "target_worker": "worker1"},
        {"job_id": "job-2", "pdb_id": starting_pdb_fname, "iterations": 2, "target_worker": "worker0"},
    ]
    assert job_complete == [
        {"worker_id": "worker0", "job_id": "job-0"},
        {"worker_id": "worker1", "job_id": "job-1"},
        {"worker_id": "worker0", "job_id": "job-2"},
    ]

    for d in diff:
        assert "source" in d
        assert "diffs" in d

        vs = list(d["diffs"].values())
        assert len(vs) == 3 # update two new mutations and parent
        for v in vs:
            assert "total_score" in v
            assert "visits" in v

        for fname in list(d["diffs"].keys()):
            assert (pdb_store.base_dir / fname).exists()

    worker0.stop()
    worker1.stop()
    coordinator.stop()

    # Check that workers correctly merged diffs
    merged_diff = defaultdict(lambda: {"total_score": 0, "visits": 0})
    for d in diff:
        for fname, d_ in d["diffs"].items():
            merged_diff[fname]["total_score"] += d_["total_score"]
            merged_diff[fname]["visits"] += d_["visits"]

    # Convert each worker's state dict to same structure
    for worker in [worker0, worker1]:
        worker_diff = {}
        for fname, ab in worker.mcts.fname_to_state.items():
            worker_diff[fname] = {"total_score": ab.total_score, "visits": ab.visits}

        assert sorted(worker_diff.keys()) == sorted(merged_diff.keys())
        for fname in worker_diff:
            assert worker_diff[fname]["visits"] == merged_diff[fname]["visits"]
            assert worker_diff[fname]["total_score"] == pytest.approx(merged_diff[fname]["total_score"])
