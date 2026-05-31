import numpy as np

import mathrobo as mr


def test_so3_batch_hat_exp_integ():
    vec = np.random.rand(2, 3, 3)
    vec[0, 0] = 0.0
    a = 0.37

    np.testing.assert_allclose(
        mr.SO3.hat(vec),
        np.stack([[mr.SO3.hat(vec[i, j]) for j in range(vec.shape[1])] for i in range(vec.shape[0])]),
    )
    np.testing.assert_allclose(
        mr.SO3.exp(vec, a),
        np.stack([[mr.SO3.exp(vec[i, j], a) for j in range(vec.shape[1])] for i in range(vec.shape[0])]),
    )
    np.testing.assert_allclose(
        mr.SO3.exp_integ(vec, a),
        np.stack([[mr.SO3.exp_integ(vec[i, j], a) for j in range(vec.shape[1])] for i in range(vec.shape[0])]),
    )


def test_se3_batch_exp_matmul_and_adjoint():
    vec = np.random.rand(4, 6)
    h = mr.SE3.set_mat(mr.SE3.exp(vec, 0.4))
    other = mr.SE3.set_mat(mr.SE3.exp(vec * 0.25, 0.7))
    points = np.random.rand(4, 3)

    np.testing.assert_allclose(
        h.mat(),
        np.stack([mr.SE3.exp(vec[i], 0.4) for i in range(vec.shape[0])]),
    )
    np.testing.assert_allclose(
        (h @ other).mat(),
        np.stack([
            (mr.SE3.set_mat(mr.SE3.exp(vec[i], 0.4)) @ mr.SE3.set_mat(mr.SE3.exp(vec[i] * 0.25, 0.7))).mat()
            for i in range(vec.shape[0])
        ]),
    )
    np.testing.assert_allclose(
        h @ points,
        np.stack([mr.SE3.set_mat(h.mat()[i]) @ points[i] for i in range(vec.shape[0])]),
    )
    np.testing.assert_allclose(
        h.mat_adj(),
        np.stack([mr.SE3.set_mat(h.mat()[i]).mat_adj() for i in range(vec.shape[0])]),
    )
    np.testing.assert_allclose(h.mat_adj() @ h.mat_inv_adj(), np.broadcast_to(np.eye(6), (4, 6, 6)), atol=1e-12)


def test_cmtm_se3_batch_mat_adj_inv_and_mul():
    vec = np.random.rand(3, 6)
    elem = mr.SE3.set_mat(mr.SE3.exp(np.random.rand(2, 6), 0.4))
    elem_vecs = np.random.rand(2, 2, 6)
    cmtm = mr.CMTM[mr.SE3](elem, elem_vecs)

    np.testing.assert_allclose(
        cmtm.mat(),
        np.stack([mr.CMTM[mr.SE3](mr.SE3.set_mat(elem.mat()[i]), elem_vecs[i]).mat() for i in range(2)]),
    )
    np.testing.assert_allclose(
        cmtm.mat_adj(),
        np.stack([mr.CMTM[mr.SE3](mr.SE3.set_mat(elem.mat()[i]), elem_vecs[i]).mat_adj() for i in range(2)]),
    )
    np.testing.assert_allclose(
        cmtm.mat() @ cmtm.mat_inv(),
        np.broadcast_to(np.eye(cmtm.size()), (2, cmtm.size(), cmtm.size())),
        rtol=1e-10,
        atol=1e-10,
    )

    rhs = mr.CMTM[mr.SE3](
        mr.SE3.set_mat(mr.SE3.exp(np.random.rand(2, 6), 0.3)),
        np.random.rand(2, 2, 6),
    )
    np.testing.assert_allclose(
        (cmtm @ rhs).mat(),
        cmtm.mat() @ rhs.mat(),
        rtol=1e-10,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        mr.CMTM.hat(mr.SE3, np.stack([vec, vec * 0.2])),
        np.stack([mr.CMTM.hat(mr.SE3, vec), mr.CMTM.hat(mr.SE3, vec * 0.2)]),
    )


def test_cmtm_se3_batch_order5_mul_inv_and_tangent():
    batch = 2
    order = 5
    left = mr.CMTM[mr.SE3](
        mr.SE3.set_mat(mr.SE3.exp(np.random.rand(batch, 6), 0.2)),
        np.random.rand(batch, order - 1, 6),
    )
    right = mr.CMTM[mr.SE3](
        mr.SE3.set_mat(mr.SE3.exp(np.random.rand(batch, 6), 0.3)),
        np.random.rand(batch, order - 1, 6),
    )

    np.testing.assert_allclose(
        (left @ right).mat(),
        left.mat() @ right.mat(),
        rtol=1e-10,
        atol=1e-10,
    )
    unbatched_left = mr.CMTM[mr.SE3](
        mr.SE3.set_mat(mr.SE3.exp(np.random.rand(6), 0.2)),
        np.random.rand(order - 1, 6),
    )
    np.testing.assert_allclose(
        (unbatched_left @ right).mat(),
        unbatched_left.mat() @ right.mat(),
        rtol=1e-10,
        atol=1e-10,
    )

    eye = np.broadcast_to(np.eye(left.size()), (batch, left.size(), left.size()))
    np.testing.assert_allclose(left.mat() @ left.mat_inv(), eye, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(left.mat() @ left.inv().mat(), eye, rtol=1e-10, atol=1e-10)

    tangent_eye = np.broadcast_to(np.eye(left.adj_size()), (batch, left.adj_size(), left.adj_size()))
    np.testing.assert_allclose(
        left.tangent_mat() @ left.tangent_mat_inv(),
        tangent_eye,
        rtol=1e-10,
        atol=1e-10,
    )


def test_cmtm_batch_hat_adj_contract():
    batch = 3
    order = 5
    vec1 = np.random.rand(batch, order, 6)
    vec2 = np.random.rand(batch, order, 6)

    hat_adj = mr.CMTM.hat_adj(mr.SE3, vec1)
    hat_commute_adj = mr.CMTM.hat_commute_adj(mr.SE3, vec2)

    assert hat_adj.shape == (batch, order * 6, order * 6)
    assert hat_commute_adj.shape == (batch, order * 6, order * 6)
    np.testing.assert_allclose(
        hat_adj @ vec2.reshape(batch, order * 6, 1),
        hat_commute_adj @ vec1.reshape(batch, order * 6, 1),
        rtol=1e-12,
        atol=1e-12,
    )


def test_cmtm_se3_batch_sub_and_variation():
    batch = 2
    order = 3
    left = mr.CMTM[mr.SE3](
        mr.SE3.set_mat(mr.SE3.exp(np.random.rand(batch, 6), 0.2)),
        np.random.rand(batch, order - 1, 6),
    )
    right = mr.CMTM[mr.SE3](
        mr.SE3.set_mat(mr.SE3.exp(np.random.rand(batch, 6), 0.3)),
        np.random.rand(batch, order - 1, 6),
    )

    np.testing.assert_allclose(
        mr.CMTM.sub_vec(left, right),
        np.stack([
            mr.CMTM.sub_vec(
                mr.CMTM[mr.SE3](mr.SE3.set_mat(left.elem_mat()[i]), left.vecs()[i]),
                mr.CMTM[mr.SE3](mr.SE3.set_mat(right.elem_mat()[i]), right.vecs()[i]),
            )
            for i in range(batch)
        ]),
    )
    np.testing.assert_allclose(
        mr.CMTM.sub_tan_vec(left, right),
        np.stack([
            mr.CMTM.sub_tan_vec(
                mr.CMTM[mr.SE3](mr.SE3.set_mat(left.elem_mat()[i]), left.vecs()[i]),
                mr.CMTM[mr.SE3](mr.SE3.set_mat(right.elem_mat()[i]), right.vecs()[i]),
            )
            for i in range(batch)
        ]),
    )
    np.testing.assert_allclose(
        mr.CMTM.sub_tan_vec(left, right, "fframe"),
        np.stack([
            mr.CMTM.sub_tan_vec(
                mr.CMTM[mr.SE3](mr.SE3.set_mat(left.elem_mat()[i]), left.vecs()[i]),
                mr.CMTM[mr.SE3](mr.SE3.set_mat(right.elem_mat()[i]), right.vecs()[i]),
                "fframe",
            )
            for i in range(batch)
        ]),
    )

    arb_vec = mr.cmvec.CMVector(np.random.rand(batch, order, 6))
    tan_var_vec = mr.cmvec.CMVector(np.random.rand(batch, order, 6))
    for frame in ["bframe", "fframe"]:
        np.testing.assert_allclose(
            left.mat_var_x_arb_vec(arb_vec, tan_var_vec, frame).cm_vec(),
            np.stack([
                mr.CMTM[mr.SE3](mr.SE3.set_mat(left.elem_mat()[i]), left.vecs()[i])
                .mat_var_x_arb_vec(
                    mr.cmvec.CMVector(arb_vec.vecs()[i]),
                    mr.cmvec.CMVector(tan_var_vec.vecs()[i]),
                    frame,
                )
                .cm_vec()
                for i in range(batch)
            ]),
            rtol=1e-14,
            atol=1e-14,
        )


def test_se3_batch_hat_commute():
    vec1 = np.random.rand(3, 6)
    vec2 = np.concatenate([np.random.rand(3, 3), np.zeros((3, 1))], axis=-1)

    np.testing.assert_allclose(
        mr.SE3.hat(vec1) @ vec2[..., None],
        mr.SE3.hat_commute(vec2) @ vec1[..., None],
    )
    np.testing.assert_allclose(
        mr.SE3.hat_commute(vec2),
        np.stack([mr.SE3.hat_commute(vec2[i]) for i in range(vec2.shape[0])]),
    )
    np.testing.assert_allclose(
        mr.SE3wrench.hat_commute(vec2),
        np.stack([mr.SE3wrench.hat_commute(vec2[i]) for i in range(vec2.shape[0])]),
    )


def test_cmvector_batch():
    vecs = np.random.rand(5, 3, 6)
    cm = mr.cmvec.CMVector(vecs)

    assert cm.vec().shape == (5, 18)
    assert cm.cm_vec().shape == (5, 18)
    np.testing.assert_allclose(cm.vecs(), vecs)
    np.testing.assert_allclose(mr.cmvec.CMVector.set_cmvecs(cm.cm_vecs()).vecs(), vecs)
