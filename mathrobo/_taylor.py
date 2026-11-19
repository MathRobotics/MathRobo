"""Products of factorial-normalized matrix coefficient arrays."""


def taylor_matmul(left, right, xp):
    """Multiply series with equal coefficient counts on axis -3, truncating to that count."""
    order = left.shape[-3]
    if order == 0:
        return left @ right
    return xp.stack([
        xp.sum(left[..., :k + 1, :, :] @ right[..., k::-1, :, :], axis=-3)
        for k in range(order)
    ], axis=-3)
