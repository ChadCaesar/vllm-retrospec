# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field


@dataclass
class RetroSpecCPUWindow:
    """Absolute output positions and fixed candidate slots for one proposal."""

    base_position: int
    confirmed: int = 0
    sparse: int = 0
    version: int = 0
    tokens: list[int] = field(default_factory=list)

    @property
    def draft_end(self) -> int:
        return len(self.tokens)

    @property
    def sparse_pending(self) -> int:
        return max(0, self.draft_end - max(self.confirmed, self.sparse))

    @property
    def full_pending(self) -> int:
        return self.draft_end - self.confirmed

    def confirm(self, position: int, token: int) -> bool:
        index = position - self.base_position - 1
        if index < self.confirmed:
            if index < 0 or self.tokens[index] != token:
                raise RuntimeError("CPU verifier attempted to change committed output")
            return False
        if index != self.confirmed:
            raise RuntimeError("CPU verifier skipped a token in its confirmed prefix")
        changed = index >= self.draft_end or self.tokens[index] != token
        if changed:
            self.tokens[index:] = [token]
            self.sparse = min(self.sparse, index)
            self.version += 1
        self.confirmed += 1
        return changed

    def apply_sparse(self, version: int, start: int, tokens: list[int]) -> bool:
        if version != self.version:
            return False
        end = start + len(tokens)
        if start < self.confirmed:
            overlap = min(self.confirmed, end) - start
            if self.tokens[start : start + overlap] != tokens[:overlap]:
                return False
        if end < self.confirmed:
            return True
        self.tokens[start:] = tokens
        self.sparse = end
        return True
