import functools
import json
import logging
import pathlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from antibody_mcts.distributed import Message, MessageTransport, PDBStore, Topic
from google.api_core.exceptions import NotFound
from google.cloud import pubsub_v1, storage
from google.cloud.pubsub_v1.subscriber.message import Message as PubSubOriginalMessage
from google.cloud.pubsub_v1.subscriber.scheduler import ThreadScheduler

logger = logging.getLogger(__name__)

class PubSubTransport(MessageTransport):
    def __init__(self, project_id: str, topic_prefix: str):
        self.project_id = project_id
        self.topic_prefix = topic_prefix
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()
        self._executor = ThreadPoolExecutor(max_workers=1) # single worker to process callbacks synchronously
        self.scheduler = ThreadScheduler(executor=self._executor)

        for topic in [Topic.WORKER_READY, Topic.NEW_JOB, Topic.JOB_COMPLETE, Topic.DIFF]:
            path = self._get_topic_path(topic)
            try:
                self.publisher.get_topic(topic=path)
            except NotFound:
                self.publisher.create_topic(name=path)

        self._callbacks = defaultdict(dict)
        self._subscriptions = defaultdict(dict)

    def send(self, topic: Topic, message: Message) -> None:
        topic_path = self._get_topic_path(topic)
        data = json.dumps(message.payload).encode("utf-8")
        self.publisher.publish(topic_path, data).result()

    def subscribe(self, topic: Topic, id: str, callback: Callable[[Message], None]) -> None:
        self._callbacks[topic][id] = callback
        topic_path, subscription_path = self._get_topic_path(topic), self._get_subscription_path(topic, id)
        self.subscriber.create_subscription(topic=topic_path, name=subscription_path)
        future = self.subscriber.subscribe(subscription=subscription_path, callback=self._callback_wrapper(callback), scheduler=self.scheduler)
        self._subscriptions[topic][id] = future

    def unsubscribe(self, topic: Topic, id: str) -> None:
        del self._callbacks[topic][id]
        future = self._subscriptions[topic].pop(id)
        future.cancel()
        future.result(timeout=1)
        self.subscriber.delete_subscription(subscription=self._get_subscription_path(topic=topic, id=id))

    def _get_topic_path(self, topic: Topic) -> str:
        return self.publisher.topic_path(self.project_id, f"{self.topic_prefix}{topic}")
    def _get_subscription_path(self, topic: Topic, id: str) -> str:
        return self.subscriber.subscription_path(self.project_id, f"{self.topic_prefix}{topic}-{id}")
    def _callback_wrapper(self, callback):
        @functools.wraps(callback)
        def wrapper(pubsub_message: PubSubOriginalMessage):
            try:
                payload = json.loads(pubsub_message.data.decode("utf-8"))
                message = Message(payload=payload)
                callback(message)
            except:
                logger.exception("Something went wrong")
                pubsub_message.nack()
            else:
                pubsub_message.ack()
        return wrapper

class GCSPDBStore(PDBStore):
    def __init__(self, project_id: str, bucket: str, local_dir: pathlib.Path):
        self.project_id = project_id
        self.local_dir = local_dir
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.client = storage.Client(project=self.project_id)
        self.bucket = self.client.bucket(bucket)
        try:
            self.client.get_bucket(bucket)
        except NotFound:
            self.bucket.create()
    def get_pdb(self, fname: str) -> Path:
        path = self.local_dir / fname
        if not path.exists():
            self.bucket.blob(fname).download_to_filename(path)
        return path
    def store_pdb(self, fname: str, pdb_file: pathlib.Path) -> None:
        self.bucket.blob(fname).upload_from_filename(pdb_file)
