# -*- coding: utf-8 -*-
"""本地单测：kv_io 的槽偏移/窄读/写回（不需要板子，用假 tensor）。

跑法：
    python -B -X utf8 board/test_kv_readback.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'patch'))

import kv_io   # noqa: E402

RESULTS = []


def check(cond, name):
    RESULTS.append((bool(cond), name))
    print(('PASS ' if cond else 'FAIL ') + name)


class _Type(object):
    def __init__(self, shape, dtype):
        self.shape = shape
        self.element_dtype = dtype


class FakeTensor(object):
    """按 [1,H,depth,head_dim] 连续布局存一份 fp16 缓冲，模拟 icraft read()。"""

    def __init__(self, logical, offset_bytes=0, dtype='@fp(16)'):
        self._data = logical.tobytes()
        self._offset = offset_bytes
        self._type = _Type(list(logical.shape), dtype)

    def tensorType(self):
        return self._type

    def dtype(self):
        return self._type

    def offset(self):
        return self._offset

    def read(self, buf, off, size):
        start = off - self._offset
        assert 0 <= start and start + size <= len(self._data), 'read out of range'
        buf[:] = np.frombuffer(self._data[start:start + size], dtype=np.uint8)


def main():
    heads, depth, head_dim = 8, 512, 128
    rng = np.random.RandomState(0)
    data = rng.randint(-1000, 1000, size=(1, heads, depth, head_dim)).astype(np.float16)

    offs = kv_io.slot_offsets(3, heads, depth, head_dim, 2)
    check(offs[0] == 3 * head_dim * 2 and offs[1] - offs[0] == depth * head_dim * 2,
          'kv_io：槽偏移按 head/depth 步长计算')
    check(kv_io.nbytes_per_slot(heads, head_dim, 2) == 2048, 'kv_io：每槽 2 KiB')

    for base in (0, 777):
        t = FakeTensor(data, offset_bytes=base)
        for pos in (0, 1, 7, 511):
            got = kv_io.read_slot(t, pos)
            check(np.array_equal(got, data[:, :, pos, :]),
                  'kv_io：pos=%d（offset=%d）窄读与整块一致' % (pos, base))

    cache = np.zeros((heads, depth, head_dim), dtype=np.float16)
    t = FakeTensor(data)
    kv_io.write_slot(cache, 9, kv_io.read_slot(t, 9))
    check(np.array_equal(cache[:, 9, :], data[0, :, 9, :]), 'kv_io：窄读写回缓存槽 9')

    passed = sum(1 for ok, _ in RESULTS if ok)
    print('test_kv_readback: %d/%d ok' % (passed, len(RESULTS)))
    return 0 if passed == len(RESULTS) else 1


if __name__ == '__main__':
    sys.exit(main())