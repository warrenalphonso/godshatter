import abc
import dataclasses
import enum
import logging
import pathlib
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from antibody_mcts.mcts import MCTS

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class Message:
    payload: dict

class Topic(str, enum.Enum):
    WORKER_READY = "WORKER_READY"
    NEW_JOB = "NEW_JOB"
    JOB_COMPLETE = "JOB_COMPLETE"
    DIFF = "DIFF"

class MessageTransport(abc.ABC):
    @abc.abstractmethod
    def send(self, topic: Topic, message: Message) -> None:
        "Send a message to a topic"

    @abc.abstractmethod
    def subscribe(self, topic: Topic, id: str, callback: Callable[[Message], None]) -> None:
        "Subscribe `callback` to `topic`."

    @abc.abstractmethod
    def unsubscribe(self, topic: Topic, id: str) -> None:
        "Unsubscribe callback associated with `id` from `topic`."

class PDBStore(abc.ABC):
    @abc.abstractmethod
    def get_pdb(self, fname: str) -> bytes: pass

    @abc.abstractmethod
    def store_pdb(self, fname: str, content: bytes) -> None: pass

class LocalMessageTransport(MessageTransport):
    "In-memory message transport for same-process communication"
    def __init__(self):
        self.subscribers: dict[str, dict[str, Callable[[Message], None]]] = defaultdict(dict)
    def send(self, topic: Topic, message: dict[str, Any]) -> None:
        for callback in self.subscribers[topic].values():
            callback(message)
    def subscribe(self, topic: Topic, id: str, callback: Callable[[dict[str, Any]], None]) -> None:
        self.subscribers[topic][id] = callback
    def unsubscribe(self, topic: Topic, id: str) -> None:
        self.subscribers[topic].pop(id)

class LocalPDBStore(PDBStore):
    "Local filesystem PDB store"
    def __init__(self, base_dir: pathlib.Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(exist_ok=True, parents=True)
    def get_pdb(self, fname: str) -> bytes:
        return (self.base_dir / fname).read_bytes()
    def store_pdb(self, fname: str, content: bytes) -> None:
        (self.base_dir / fname).write_bytes(content)

class MCTSWorker:
    def __init__(self, worker_id: str, message_transport: MessageTransport, pdb_store: PDBStore, mcts_factory: Callable[[], "MCTS"]):
        self.worker_id = worker_id
        self.transport = message_transport
        self.pdb_store = pdb_store
        self.mcts = mcts_factory()
        self.running = False

    def start(self) -> None:
        if self.running: return
        self.transport.subscribe(Topic.DIFF, self.worker_id, self._handle_diff)
        self.transport.subscribe(Topic.NEW_JOB, self.worker_id, self._handle_job)
        self.transport.send(Topic.WORKER_READY, Message(payload={"worker_id": self.worker_id}))
        logger.info("Sent worker ready")
        self.running = True

    def stop(self) -> None:
        if not self.running: return
        self.transport.unsubscribe(Topic.DIFF, self.worker_id)
        self.transport.unsubscribe(Topic.NEW_JOB, self.worker_id)
        self.running = False

    def _handle_diff(self, message: Message) -> None:
        "Handle diffs from other workers"
        logger.info("Received diff")
        if message.payload["source"] != self.worker_id:  # Avoid our own diffs
            logger.info("Received diff from other worker")
            diffs = message.payload["diffs"]
            for fname in diffs:
                p = self.mcts.mutations_dir / fname
                if not p.exists(): p.write_bytes(self.pdb_store.get_pdb(fname))
            self.mcts.loads_diff(diffs)

    def _handle_job(self, message: Message) -> None:
        logger.info("Received job")
        if message.payload["target_worker"] != self.worker_id: return
        logger.info("Received job for me")
        fname = message.payload["pdb_id"]
        pdb_path = self.mcts.mutations_dir / fname
        if not pdb_path.exists(): pdb_path.write_bytes(self.pdb_store.get_pdb(fname))
        iterations = message.payload["iterations"]
        for _ in range(iterations):
            self.mcts.run(pdb=pdb_path)
        logger.info("Finished simulating")
        diffs = self.mcts.dumps_diff()
        for fname in diffs:
            self.pdb_store.store_pdb(fname=fname, content=(self.mcts.mutations_dir / fname).read_bytes())
        self.transport.send(Topic.DIFF, Message(payload={"source": self.worker_id, "diffs": diffs}))
        self.transport.send(Topic.JOB_COMPLETE, Message(payload={"worker_id": self.worker_id, "job_id": message.payload["job_id"]}))
        logger.info("Sent diff and job complete")

class MCTSCoordinator:
    def __init__(self, transport: MessageTransport, pdb_store: PDBStore):
        self.transport = transport
        self.pdb_store = pdb_store
        self.workers = []
        self.available_workers = deque()
        self.job_queue = deque()
        self.active_jobs = {}
        self.running = False

    def start(self) -> None:
        if self.running: return
        self.transport.subscribe(Topic.WORKER_READY, "coordinator", self._handle_worker_ready)
        self.transport.subscribe(Topic.JOB_COMPLETE, "coordinator", self._handle_job_complete)
        self.running = True

    def stop(self) -> None:
        if not self.running: return
        self.transport.unsubscribe(Topic.WORKER_READY, "coordinator")
        self.transport.unsubscribe(Topic.JOB_COMPLETE, "coordinator")
        self.running = False

    def _handle_worker_ready(self, message: Message) -> None:
        logger.info("Received worker ready")
        worker_id = message.payload["worker_id"]
        self.workers.append(worker_id)
        self.available_workers.append(worker_id)
        self._assign_pending_jobs()

    def _handle_job_complete(self, message: Message) -> None:
        logger.info("Received job complete")
        worker_id = message.payload["worker_id"]
        job_id = message.payload["job_id"]
        del self.active_jobs[job_id]
        self.available_workers.append(worker_id)
        self._assign_pending_jobs()

    def run_distributed(self, fname: str, total_iterations: int, iterations_per_job: int) -> list[tuple]:
        job_id = 0
        while total_iterations > 0:
            iterations_for_job = min(iterations_per_job, total_iterations)
            job = {"job_id": f"job-{job_id}", "pdb_id": fname, "iterations": iterations_for_job, "target_worker": None}
            self.job_queue.append(job)
            total_iterations -= iterations_for_job
            job_id += 1
        self._assign_pending_jobs()

    def _assign_pending_jobs(self) -> None:
        while self.job_queue and self.available_workers:
            job = self.job_queue.popleft()
            worker_id = self.available_workers.popleft()
            job["target_worker"] = worker_id
            self.active_jobs[job["job_id"]] = job
            self.transport.send(Topic.NEW_JOB, Message(payload=job))
            logger.info("Sent new job")
