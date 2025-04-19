import abc
import dataclasses
import pathlib
from collections import defaultdict, deque
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from antibody_mcts.mcts import MCTS

@dataclasses.dataclass
class Message:
    payload: dict

class MessageTransport(abc.ABC):
    @abc.abstractmethod
    def send(self, topic: str, message: Message) -> None:
        "Send a message to a topic"

    @abc.abstractmethod
    def subscribe(self, topic: str, callback: Callable[[Message], None]) -> None:
        pass

    @abc.abstractmethod
    def unsubscribe(self, topic: str, callback: Callable[[Message], None]) -> None:
        pass

class PDBStore(abc.ABC):
    @abc.abstractmethod
    def get_pdb(self, fname: str) -> pathlib.Path:
        "Get a PDB file, downloading if necessary"

    @abc.abstractmethod
    def store_pdb(self, fname: str, pdb_data: bytes) -> pathlib.Path:
        pass

class LocalMessageTransport(MessageTransport):
    """In-memory message transport for same-process communication"""
    def __init__(self):
        self.subscribers = defaultdict(dict) # dict as ordered set

    def send(self, topic: str, message: dict[str, Any]) -> None:
        for callback in self.subscribers[topic].keys():
            callback(message)

    def subscribe(self, topic: str, callback: Callable[[dict[str, Any]], None]) -> None:
        self.subscribers[topic][callback] = None

    def unsubscribe(self, topic: str, callback: Callable[[dict[str, Any]], None]) -> None:
        self.subscribers[topic].pop(callback)

class LocalPDBStore(PDBStore):
    """Local filesystem PDB store"""
    def __init__(self, base_dir: pathlib.Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(exist_ok=True, parents=True)

    def get_pdb(self, fname: str) -> pathlib.Path:
        return self.base_dir / fname

    def store_pdb(self, fname: str, pdb_data: bytes) -> pathlib.Path:
        path = self.base_dir / fname
        path.write_bytes(pdb_data)
        return path

class MCTSWorker:
    """Worker that runs MCTS iterations"""
    def __init__(self, worker_id: str, message_transport: MessageTransport, pdb_store: PDBStore, mcts_factory: Callable[[], "MCTS"]):
        self.worker_id = worker_id
        self.transport = message_transport
        self.pdb_store = pdb_store
        self.mcts = mcts_factory()
        self.running = False

    def start(self) -> None:
        self.transport.subscribe("diff", self._handle_diff)
        self.transport.subscribe("job", self._handle_job)
        self.transport.subscribe("stop", self._handle_stop)
        self.transport.send("worker_ready", Message(payload={"worker_id": self.worker_id}))
        self.running = True

    def stop(self) -> None:
        self.running = False
        self.transport.unsubscribe("diff", self._handle_diff)
        self.transport.unsubscribe("job", self._handle_job)
        self.transport.unsubscribe("stop", self._handle_stop)

    def _handle_diff(self, message: Message) -> None:
        """Handle diffs from other workers"""
        if message.payload["source"] != self.worker_id:  # Avoid our own diffs
            self.mcts.loads_diff(message.payload["diffs"])

    def _handle_job(self, message: Message) -> None:
        """Handle a job assignment"""
        if message.payload["target_worker"] != self.worker_id: return
        pdb_path = self.pdb_store.get_pdb(message.payload["pdb_id"])
        iterations = message.payload["iterations"]
        for _ in range(iterations):
            self.mcts.run(pdb=pdb_path)
        diffs = self.mcts.dumps_diff()
        self.transport.send("diff", Message(payload={"source": self.worker_id, "diffs": diffs}))
        self.transport.send("job_complete", Message(payload={"worker_id": self.worker_id, "job_id": message.payload["job_id"]}))

    def _handle_stop(self, _message: Message) -> None:
        """Handle stop request"""
        self.stop()

class MCTSCoordinator:
    """Coordinates multiple workers"""
    def __init__(self, transport: MessageTransport, pdb_store: PDBStore):
        self.transport = transport
        self.pdb_store = pdb_store
        self.workers = []
        self.available_workers = deque()
        self.job_queue = deque()
        self.active_jobs = {}
        self.running = False

    def start(self) -> None:
        """Start the coordinator"""
        self.transport.subscribe("worker_ready", self._handle_worker_ready)
        self.transport.subscribe("job_complete", self._handle_job_complete)
        self.running = True

    def stop(self) -> None:
        """Stop the coordinator"""
        self.running = False
        self.transport.send("stop", Message(payload={}))
        self.transport.unsubscribe("worker_ready", self._handle_worker_ready)
        self.transport.unsubscribe("job_complete", self._handle_job_complete)

    def _handle_worker_ready(self, message: Message) -> None:
        """Handle worker ready message"""
        worker_id = message.payload["worker_id"]
        self.workers.append(worker_id)
        self.available_workers.append(worker_id)
        self._assign_pending_jobs()

    def _handle_job_complete(self, message: Message) -> None:
        """Handle job completion message"""
        worker_id = message.payload["worker_id"]
        job_id = message.payload["job_id"]
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
        self.available_workers.append(worker_id)
        self._assign_pending_jobs()

    def run_distributed(self, fname: str, total_iterations: int, iterations_per_job: int) -> list[tuple]:
        """Run distributed MCTS"""
        if not self.running: self.start()
        job_id = 0
        while total_iterations > 0:
            iterations_for_job = min(iterations_per_job, total_iterations)
            job = {"job_id": f"job-{job_id}", "pdb_id": fname, "iterations": iterations_for_job, "target_worker": None}
            self.job_queue.append(job)
            total_iterations -= iterations_for_job
            job_id += 1
        self._assign_pending_jobs()

    def _assign_pending_jobs(self) -> None:
        """Assign pending jobs to available workers"""
        while self.job_queue and self.available_workers:
            job = self.job_queue.popleft()
            worker_id = self.available_workers.popleft()
            job["target_worker"] = worker_id
            self.active_jobs[job["job_id"]] = job
            self.transport.send("job", Message(payload=job))
