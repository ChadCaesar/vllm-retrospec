# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A process-isolated CPU target with coordinator-owned file-backed KV."""

import atexit
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from collections import deque
from contextlib import suppress
from multiprocessing.connection import Client, Listener
from pathlib import Path

import torch


class _CPUForwardInterrupted(Exception):
    pass


def map_cpu_kv(path: str, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.from_file(
        path,
        shared=True,
        size=math.prod(shape),
        dtype=torch.bfloat16,
    ).view(shape)


def _cpu_target_main(
    connection, model_path: str, revision: str | None, threads: int, config_path=None
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        if __package__:
            from .cpu_model import attach_cache, grouped_decode_attention
        else:
            from cpu_model import attach_cache, grouped_decode_attention

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        ALL_ATTENTION_FUNCTIONS.register("sdpa", grouped_decode_attention)
        config = AutoConfig.from_pretrained(
            model_path, revision=revision, local_files_only=True
        )
        if config_path:
            config.update(json.loads(Path(config_path).read_text()))
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            revision=revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        ).eval()
        if any(parameter.device.type != "cpu" for parameter in model.parameters()):
            raise RuntimeError("CPU verifier loaded a parameter on a non-CPU device")
        connection.send(("ready",))
        cache = storage = journal = output = None
        session = epoch = 0
        token = position = limit = 0
        running = False
        pending_command = None
        eos_ids: set[int] = set()

        def check_control(_module, _inputs):
            nonlocal pending_command
            if connection.poll():
                pending_command = connection.recv()
                raise _CPUForwardInterrupted

        # Stop at a decoder-layer boundary instead of reading the remaining
        # model weights while GPU full verification gathers its historical KV.
        # No output is published for a partially executed CPU token.
        for layer in model.model.layers:
            layer.register_forward_pre_hook(check_control)

        with torch.inference_mode():
            while True:
                if pending_command is not None or connection.poll() or not running:
                    command = pending_command
                    pending_command = None
                    if command is None:
                        command = connection.recv()
                    operation = command[0]
                    if operation == "close":
                        return
                    if operation == "pause":
                        epoch = command[1]
                        running = False
                        connection.send(("paused", session, epoch))
                        continue
                    if operation == "release":
                        _, session, epoch = command
                        running = False
                        # All CPU kernels are synchronous. Reaching this point
                        # after a layer-boundary interruption fences every old
                        # KV access, including partially written token rows.
                        cache = storage = journal = output = None
                        connection.send(("released", session, epoch))
                        continue
                    if operation == "initialize":
                        _, session, epoch, paths, shape, position, token, limit, eos = (
                            command
                        )
                        storage = map_cpu_kv(paths[0], shape)
                        journal = map_cpu_kv(paths[1], shape)
                        cache = attach_cache(model.config, storage, position)
                        eos_ids = set(eos)
                    elif operation == "resume":
                        _, session, epoch, start, position, token, limit = command
                        if cache is None or cache.get_seq_length() < start:
                            raise RuntimeError(
                                "CPU verifier missed an authoritative KV prefix"
                            )
                        storage[..., start:position, :].copy_(
                            journal[..., start:position, :]
                        )
                        for index in range(len(cache.layers)):
                            cache.layers[index].length = position
                    else:
                        raise RuntimeError(f"Unknown CPU verifier command: {operation}")
                    running = position < limit
                    connection.send(("resumed", session, epoch, position))
                    # Drain newer control commands before starting another token.
                    continue
                ids = torch.tensor([[token]], dtype=torch.int64)
                started = time.monotonic()
                try:
                    output = model(
                        input_ids=ids,
                        past_key_values=cache,
                        use_cache=True,
                        cache_position=torch.tensor([position]),
                        logits_to_keep=1,
                    )
                except _CPUForwardInterrupted:
                    for index in range(len(cache.layers)):
                        cache.layers[index].length = position
                    connection.send(("interrupted", session, epoch, position))
                    continue
                logits = output.logits[0, -1]
                if not torch.isfinite(logits).all():
                    raise RuntimeError("CPU verifier produced non-finite logits")
                token = int(logits.argmax().item())
                position += 1
                connection.send(
                    (
                        "token",
                        session,
                        epoch,
                        position,
                        token,
                        time.monotonic() - started,
                    )
                )
                running = position < limit and token not in eos_ids
    except (EOFError, ConnectionError):
        return
    except BaseException:
        try:
            with suppress(EOFError, ConnectionError):
                connection.send(("error", traceback.format_exc()))
        finally:
            connection.close()


class RetroSpecCPUWorker:
    """Only TP rank zero owns this process; no CUDA tensors cross the pipe."""

    def __init__(
        self, model_path: str, revision: str | None, threads: int, config=None
    ):
        # vLLM GPU workers can themselves be daemon processes, which cannot
        # create multiprocessing children. A fresh interpreter also keeps all
        # CUDA initialization and platform globals out of the CPU process.
        self.socket_directory = tempfile.mkdtemp(prefix="retrospec-cpu-control-")
        address = str(Path(self.socket_directory) / "control")
        self.config_path = Path(self.socket_directory) / "config.json"
        if config is not None:
            self.config_path.write_text(config.to_json_string(use_diff=False))
        listener = Listener(address, family="AF_UNIX")
        listener._listener._socket.settimeout(1)
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                address,
                model_path,
                revision or "",
                str(threads),
                str(self.config_path) if config is not None else "",
            ],
            env=environment,
        )
        deadline = time.monotonic() + 180
        try:
            while True:
                try:
                    self.connection = listener.accept()
                    break
                except TimeoutError:
                    if self.process.poll() is not None:
                        raise RuntimeError(
                            "CPU verifier exited before connecting"
                        ) from None
                    if time.monotonic() > deadline:
                        self.process.terminate()
                        raise RuntimeError(
                            "CPU verifier did not connect within 180s"
                        ) from None
        finally:
            listener.close()
        self.ready = False
        self.messages = deque()
        self.started_at = time.monotonic()
        self.last_progress = self.started_at
        self.awaiting_progress = False
        self.last_command = None
        self.released = None
        self.closed = False
        self.close_requested = False
        atexit.register(self.close)

    def poll(self):
        while self.connection.poll():
            try:
                message = self.connection.recv()
            except EOFError as exc:
                raise RuntimeError(
                    "Independent CPU verifier exited unexpectedly"
                ) from exc
            if message[0] == "error":
                raise RuntimeError("Independent CPU verifier failed:\n" + message[1])
            if message[0] == "ready":
                self.ready = True
            else:
                self.messages.append(message)
            if message[0] == "released":
                self.released = message[1:3]
            self.last_progress = time.monotonic()
            if self.last_command is not None:
                operation, session, epoch, limit = self.last_command
                if message[0] == "paused" and operation == "pause":
                    self.awaiting_progress = message[2] != epoch
                elif message[0] == "released" and operation == "release":
                    self.awaiting_progress = message[1:3] != (session, epoch)
                elif (
                    message[0] == "token"
                    and message[1:3] == (session, epoch)
                    and message[3] >= limit
                ):
                    self.awaiting_progress = False
        if self.process.poll() is not None and not self.closed:
            raise RuntimeError(
                f"CPU verifier exited with code {self.process.returncode}"
            )
        if not self.ready and time.monotonic() - self.started_at > 180:
            raise RuntimeError("CPU verifier did not finish initialization within 180s")
        if self.awaiting_progress and time.monotonic() - self.last_progress > 300:
            raise RuntimeError(
                "CPU verifier made no progress for 300s: "
                f"pid={self.process.pid}, command={self.last_command}"
            )
        messages = list(self.messages)
        self.messages.clear()
        return messages

    def send(self, command):
        if self.process.poll() is not None:
            raise RuntimeError("Independent CPU verifier is not running")
        self.connection.send(command)
        if not self.awaiting_progress:
            self.last_progress = time.monotonic()
        if command[0] in ("initialize", "resume"):
            self.released = None
            self.last_command = (command[0], command[1], command[2], command[-2])
            if command[0] == "resume":
                self.last_command = (command[0], command[1], command[2], command[-1])
            self.awaiting_progress = True
        elif command[0] == "pause":
            self.last_command = ("pause", None, command[1], 0)
        elif command[0] == "release":
            self.released = None
            self.last_command = ("release", command[1], command[2], 0)
            self.awaiting_progress = True

    def wait_for_release(self, session: int, epoch: int, timeout: float = 300):
        """Drain old results and fence a previously submitted release."""
        deadline = time.monotonic() + timeout
        while True:
            self.poll()
            if self.released == (session, epoch):
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    "CPU verifier did not release KV within "
                    f"{timeout}s: pid={self.process.pid}, "
                    f"session={session}, epoch={epoch}"
                )
            self.connection.poll(min(0.1, remaining))

    def request_close(self):
        if self.closed or self.close_requested:
            return
        self.close_requested = True
        if self.process.poll() is None:
            with suppress(OSError, EOFError):
                self.connection.send(("close",))

    def close(self):
        if self.closed:
            return
        self.request_close()
        self.closed = True
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
        self.connection.close()
        Path(self.socket_directory, "control").unlink(missing_ok=True)
        self.config_path.unlink(missing_ok=True)
        Path(self.socket_directory).rmdir()


class RetroSpecCPUStorage:
    def __init__(self, shape: tuple[int, ...], directory: str | None = None):
        self.owner = directory is None
        self.directory = directory or tempfile.mkdtemp(prefix="retrospec-cpu-kv-")
        self.shape = shape
        self.paths = tuple(
            str(Path(self.directory) / name) for name in ("kv", "gpu-journal")
        )
        if self.owner:
            size = math.prod(shape) * torch.bfloat16.itemsize
            if (
                size
                > os.statvfs(self.directory).f_bavail
                * os.statvfs(self.directory).f_frsize
            ):
                raise RuntimeError("Insufficient temporary storage for CPU verifier KV")
            for path in self.paths:
                with open(path, "wb") as stream:
                    stream.truncate(size)
        self.kv = map_cpu_kv(self.paths[0], shape)
        self.journal = map_cpu_kv(self.paths[1], shape)

    def can_reuse(self, shape: tuple[int, ...]) -> bool:
        return (
            self.kv is not None
            and self.shape[:-2] == shape[:-2]
            and self.shape[-1] == shape[-1]
            and self.shape[-2] >= shape[-2]
        )

    def close(self):
        self.kv = self.journal = None
        if self.owner:
            for path in self.paths:
                Path(path).unlink(missing_ok=True)
            Path(self.directory).rmdir()


if __name__ == "__main__":
    connection = Client(sys.argv[1], family="AF_UNIX")
    _cpu_target_main(
        connection,
        sys.argv[2],
        sys.argv[3] or None,
        int(sys.argv[4]),
        sys.argv[5] or None,
    )
