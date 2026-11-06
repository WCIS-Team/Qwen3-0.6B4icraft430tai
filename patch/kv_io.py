# -*- coding: utf-8 -*-
"""Qwen3 KV 槽位窄回读（2026-11-07）。

图输出 `past_key_values.*` 形状 [1, H, depth, head_dim]（fp16），每步只有
**一个槽位** pos 是新的；整块回读是 1 MiB/张量 ×56 = 58 MiB/步。这里按字节
偏移只读该槽（H × head_dim × 2B = 2 KiB/张量），把每步回读压到 ~112 KiB。

`read(tensor, ...)` 的 offset 相对整块分配（与 `Tensor2Numpy` 里
`read(buf, offset=item.offset(), size=byte_size)` 同一语义），所以
绝对偏移 = tensor.offset() + 槽内相对偏移。
"""
import numpy as np

_DTYPES = {2: np.float16, 4: np.float32}


def element_size(dtype_text):
    """`@fp(16)`→2，`@fp(32)`→4；其余不支持（Qwen3 的 KV 只有 fp16）。"""
    text = str(dtype_text)
    if 'fp(16)' in text:
        return 2
    if 'fp(32)' in text:
        return 4
    raise ValueError('unsupported kv element dtype: %s' % text)


def slot_offsets(pos, n_heads, depth, head_dim, itemsize=2):
    """每个 head 在槽 pos 上的相对字节偏移（连续布局 [1,H,depth,head_dim]）。"""
    return [((h * depth + pos) * head_dim) * itemsize for h in range(n_heads)]


def _tensor_type(tensor):
    """运行期输出张量是 xrt.Tensor：用 dtype()；网络里的静态张量才有 tensorType()。"""
    getter = getattr(tensor, 'dtype', None)
    if callable(getter):
        return getter()
    return tensor.tensorType()


def read_slot(tensor, pos, element_dtype=None):
    """只读 tensor 的第 pos 槽，返回 [1, H, head_dim] 的 numpy 数组。"""
    ttype = _tensor_type(tensor)
    shape = [int(x) for x in ttype.shape]
    if len(shape) != 4:
        raise ValueError('expected 4-D kv tensor, got %r' % (shape,))
    _1, heads, depth, head_dim = shape
    if not 0 <= pos < depth:
        raise ValueError('pos %d out of range for depth %d' % (pos, depth))
    itemsize = element_size(element_dtype if element_dtype is not None
                            else ttype.element_dtype)
    chunk = head_dim * itemsize
    buf = np.zeros(heads * chunk, dtype=np.uint8)
    base = tensor.offset()
    for h, rel in enumerate(slot_offsets(pos, heads, depth, head_dim, itemsize)):
        tensor.read(buf[h * chunk:(h + 1) * chunk], base + rel, chunk)
    return np.frombuffer(buf.tobytes(), dtype=_DTYPES[itemsize]).reshape(1, heads, head_dim)


def write_slot(cache, pos, slot_value):
    """把窄读结果写进缓存槽；cache 为 [1,H,depth,head_dim] 或 [H,depth,head_dim]。"""
    value = slot_value[0]
    if cache.ndim == 4:
        cache[0, :, pos, :] = value
    elif cache.ndim == 3:
        cache[:, pos, :] = value
    else:
        raise ValueError('unsupported cache shape %r' % (cache.shape,))
    return cache


def nbytes_per_slot(heads, head_dim, itemsize=2):
    return heads * head_dim * itemsize