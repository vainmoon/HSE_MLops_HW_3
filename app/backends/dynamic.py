import asyncio
import logging

logger = logging.getLogger(__name__)


class DynamicBatchingBackend:
    def __init__(self, inner_backend, max_batch_size: int, max_wait_ms: float):
        self._backend = inner_backend
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms / 1000.0
        self._queue: asyncio.Queue | None = None
        self._worker_task: asyncio.Task | None = None

    async def start(self) -> None:
        self._queue = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._worker())
        logger.info(
            "Dynamic batching started: max_batch_size=%d, max_wait_ms=%.1f",
            self._max_batch_size,
            self._max_wait_ms * 1000,
        )

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()

    async def encode(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put((text, future))
        return await future

    async def _worker(self) -> None:
        loop = asyncio.get_running_loop()
        while True:

            text, future = await self._queue.get()
            batch = [(text, future)]


            deadline = loop.time() + self._max_wait_ms
            while len(batch) < self._max_batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            texts = [t for t, _ in batch]
            futures = [f for _, f in batch]

            logger.info("Processing batch of %d requests", len(batch))

            try:
                embeddings = await loop.run_in_executor(
                    None, self._backend.encode_batch, texts
                )
                for fut, emb in zip(futures, embeddings):
                    if not fut.done():
                        fut.set_result(emb)
            except Exception as e:
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(e)
