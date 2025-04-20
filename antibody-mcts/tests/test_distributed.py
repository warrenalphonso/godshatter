import pytest
from antibody_mcts.distributed import LocalMessageTransport, LocalPDBStore, MCTSCoordinator, MCTSWorker, Message, Topic


@pytest.fixture
def transport():
    return LocalMessageTransport()

@pytest.fixture
def pdb_store(data):
    return LocalPDBStore(base_dir=data)

@pytest.fixture
def coordinator(transport, pdb_store):
    return MCTSCoordinator(transport=transport, pdb_store=pdb_store)

class MockMCTS:
    def __init__(self):
        self.runs = []
        self.fname_to_state = {}

    def run(self, pdb):
        """Record run"""
        self.runs.append(pdb)

    def dumps_diff(self):
        """Return fake diffs"""
        return {"test.pdb": {"total_score": -10, "visits": 1}}

    def loads_diff(self, diffs):
        """Record diffs"""
        pass

@pytest.fixture
def mcts_factory():
    return lambda: MockMCTS()

def test_distributed_run(transport, pdb_store, mcts_factory, coordinator):
    worker_ready = []
    job = []
    job_complete = []
    diff = []
    transport.subscribe(Topic.WORKER_READY, "test", lambda message: worker_ready.append(message.payload.copy()))
    transport.subscribe(Topic.NEW_JOB, "test", lambda message: job.append(message.payload.copy()))
    transport.subscribe(Topic.JOB_COMPLETE, "test", lambda message: job_complete.append(message.payload.copy()))
    transport.subscribe(Topic.DIFF, "test", lambda message: diff.append(message.payload.copy()))

    coordinator.start()
    worker0 = MCTSWorker("worker0", transport, pdb_store, mcts_factory)
    worker0.start()
    worker1 = MCTSWorker("worker1", transport, pdb_store, mcts_factory)
    worker1.start()
    assert worker_ready == [
        {"worker_id": "worker0"},
        {"worker_id": "worker1"},
    ]

    coordinator.run_distributed("test.pdb", total_iterations=10, iterations_per_job=2)
    assert job == [
        {"job_id": "job-0", "pdb_id": "test.pdb", "iterations": 2, "target_worker": "worker0"},
        {"job_id": "job-1", "pdb_id": "test.pdb", "iterations": 2, "target_worker": "worker1"},
        {"job_id": "job-2", "pdb_id": "test.pdb", "iterations": 2, "target_worker": "worker0"},
        {"job_id": "job-3", "pdb_id": "test.pdb", "iterations": 2, "target_worker": "worker1"},
        {"job_id": "job-4", "pdb_id": "test.pdb", "iterations": 2, "target_worker": "worker0"},
    ]
    assert job_complete == [
        {"worker_id": "worker0", "job_id": "job-0"},
        {"worker_id": "worker1", "job_id": "job-1"},
        {"worker_id": "worker0", "job_id": "job-2"},
        {"worker_id": "worker1", "job_id": "job-3"},
        {"worker_id": "worker0", "job_id": "job-4"},
    ]

    assert diff == [
        {"source": "worker0", "diffs": {"test.pdb": {"total_score": -10, "visits": 1}}},
        {"source": "worker1", "diffs": {"test.pdb": {"total_score": -10, "visits": 1}}},
        {"source": "worker0", "diffs": {"test.pdb": {"total_score": -10, "visits": 1}}},
        {"source": "worker1", "diffs": {"test.pdb": {"total_score": -10, "visits": 1}}},
        {"source": "worker0", "diffs": {"test.pdb": {"total_score": -10, "visits": 1}}},
    ]

    worker0.stop()
    worker1.stop()
    coordinator.stop()

    assert len(worker0.mcts.runs) + len(worker1.mcts.runs) == 10
    # Workers should finish immediately and so should alternate
    assert len(worker0.mcts.runs) == 6
    assert len(worker1.mcts.runs) == 4
