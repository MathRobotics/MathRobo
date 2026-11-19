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


def taylor_matmul_left_vjp(right, cotangent, xp):
    """Pull back through the left operand of a truncated series product."""
    order = right.shape[-3]
    right_t = right.swapaxes(-1, -2)
    return xp.stack([
        xp.sum(cotangent[..., k:, :, :] @ right_t[..., :order - k, :, :], axis=-3)
        for k in range(order)
    ], axis=-3)


def taylor_matmul_vjp(left, right, cotangent, xp):
    """Transpose the linearization of taylor_matmul under the coefficient pairing."""
    left_cotangent = taylor_matmul_left_vjp(right, cotangent, xp)
    right_cotangent_t = taylor_matmul_left_vjp(
        left.swapaxes(-1, -2), cotangent.swapaxes(-1, -2), xp,
    )
    return left_cotangent, right_cotangent_t.swapaxes(-1, -2)
