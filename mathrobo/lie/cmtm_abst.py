from typing import TypeVar, Generic, Union
import numpy as np
import jax.numpy as jnp
import math
import jax

from ..basic import cm_vec as cmvec

T = TypeVar('T')

class CMTM(Generic[T]):
    def __init__(self, elem_mat : T, elem_vecs : np.ndarray = None, LIB = 'numpy'): 
        '''
        Constructor
        '''
        if elem_vecs is None:
            elem_vecs = np.array([])
        self._mat = elem_mat
        self._vecs = elem_vecs
        self._cmvecs = None

        elem_cls = type(elem_mat)

        def _static_int(name: str):
            attr = getattr(elem_cls, name, None)
            if attr is None:
                return None
            try:
                val = attr() if callable(attr) else attr
            except TypeError:
                return None
            if isinstance(val, (int, np.integer)):
                return int(val)
            return None

        self._dof = _static_int('dof')
        self._mat_size = _static_int('mat_size')
        self._mat_adj_size = _static_int('mat_adj_size')

        if self._mat_size is None:
            self._mat_size = elem_mat.mat().shape[0]
        if self._mat_adj_size is None:
            self._mat_adj_size = elem_mat.mat_adj().shape[0]
        if self._dof is None:
            self._dof = self._mat_adj_size

        self._n = elem_vecs.shape[0] + 1
        self._size = self._mat_size * self._n
        self._adj_size = self._mat_adj_size * self._n
        self._lib = LIB

        # NumPy backend caches (small-order repeated calls benefit most).
        self._hat_series_cache_numpy = {}
        self._mat_blocks_cache_numpy = {}
        self._mat_adj_blocks_cache_numpy = {}
        self._mat_inv_blocks_cache_numpy = {}
        self._mat_inv_adj_blocks_cache_numpy = {}
        self._tangent_table_cache_numpy = {}
        self._tangent_cm_table_cache_numpy = {}
        self._mat_matrix_cache_numpy = {}
        self._mat_adj_matrix_cache_numpy = {}
        self._mat_inv_matrix_cache_numpy = {}
        self._mat_inv_adj_matrix_cache_numpy = {}
        self._tangent_matrix_cache_numpy = {}
        self._tangent_cm_matrix_cache_numpy = {}

    def _cmvecs_obj(self) -> cmvec.CMVector:
        if self._cmvecs is None:
            self._cmvecs = cmvec.CMVector(self._vecs)
        return self._cmvecs

    def size(self) -> int:
        return self._size
    
    def adj_size(self) -> int:
        return self._adj_size

    def __check_output_order(self, output_order : int):
        if output_order is None:
            output_order = self._n
        if output_order > self._n:
            raise TypeError("Output order should be less than or equal to the order of CMTM")
        if output_order < 0:
            output_order = self._n + output_order
        return output_order

    def _hat_series_numpy(self, output_order: int, adj: bool, dtype) -> np.ndarray:
        mat_size = self._mat_adj_size if adj else self._mat_size
        if output_order <= 1:
            return np.empty((0, mat_size, mat_size), dtype=dtype)

        dtype_str = np.dtype(dtype).str
        cache_key = (adj, output_order, dtype_str)
        cached = self._hat_series_cache_numpy.get(cache_key)
        if cached is not None:
            return cached

        cm_vecs = self._cmvecs_obj().cm_vecs()
        hats = np.empty((output_order - 1, mat_size, mat_size), dtype=dtype)
        hat_func = self._mat.hat_adj if adj else self._mat.hat
        for i in range(output_order - 1):
            hats[i] = hat_func(cm_vecs[i])

        self._hat_series_cache_numpy[cache_key] = hats
        return hats
        
    def __mat_elem(self, p : int):
        if self._lib == 'jax':
            # p: jnp.int32 scalar でも Python int でも OK
            dtype = self._vecs[0].dtype

            # 逆階乗列  [1/0!, 1/1!, ..., 1/(p-1)!]
            inv_fact = jnp.reciprocal(             # 1 / n!
                jnp.cumprod(jnp.arange(1, p, dtype=dtype), exclusive=True)
            )                                       # shape (p,)

            M0 = self._mat.mat(LIB="jax")
            Ms = jnp.zeros((p + 1, self._mat_size, self._mat_size), dtype).at[0].set(M0)

            def outer_body(k, Ms):
                def inner_body(i, acc):
                    Mk_prev = Ms[k - i - 1]
                    hat_i   = self._mat.hat(self._vecs[i] * inv_fact[i])
                    return acc + Mk_prev @ hat_i

                acc0 = jnp.zeros((self._mat_size, self._mat_size), dtype)
                tmp  = jax.lax.fori_loop(0, k, inner_body, acc0)
                return Ms.at[k].set(tmp / k)

            Ms = jax.lax.fori_loop(1, p + 1, outer_body, Ms)
            return Ms[p]
        elif self._lib == 'numpy':    
            if p == 0:
                return self._mat.mat()
            else:
                mat = np.zeros( (self._mat_size, self._mat_size) ) 
                for i in range(p):
                    mat = mat + self.__mat_elem(p-(i+1)) @ self._mat.hat(self._cmvecs_obj().cm_vecs()[i])

                return mat / p

    def _mat_blocks_numpy(self, output_order: int) -> np.ndarray:
        cached = self._mat_blocks_cache_numpy.get(output_order)
        if cached is not None:
            return cached

        mat0 = self._mat.mat()
        dtype = mat0.dtype
        mat_size = self._mat_size
        blocks = np.zeros((output_order, mat_size, mat_size), dtype=dtype)

        if output_order == 0:
            self._mat_blocks_cache_numpy[output_order] = blocks
            return blocks

        blocks[0] = mat0
        if output_order == 1:
            self._mat_blocks_cache_numpy[output_order] = blocks
            return blocks

        hats = self._hat_series_numpy(output_order, adj=False, dtype=dtype)

        for k in range(1, output_order):
            acc = np.zeros((mat_size, mat_size), dtype=dtype)
            for i in range(k):
                acc += blocks[k - i - 1] @ hats[i]
            blocks[k] = acc / k

        self._mat_blocks_cache_numpy[output_order] = blocks
        return blocks

    def _mat_adj_blocks_numpy(self, output_order: int) -> np.ndarray:
        cached = self._mat_adj_blocks_cache_numpy.get(output_order)
        if cached is not None:
            return cached

        mat0 = self._mat.mat_adj()
        dtype = mat0.dtype
        mat_size = self._mat_adj_size
        blocks = np.zeros((output_order, mat_size, mat_size), dtype=dtype)

        if output_order == 0:
            self._mat_adj_blocks_cache_numpy[output_order] = blocks
            return blocks

        blocks[0] = mat0
        if output_order == 1:
            self._mat_adj_blocks_cache_numpy[output_order] = blocks
            return blocks

        hats = self._hat_series_numpy(output_order, adj=True, dtype=dtype)

        for k in range(1, output_order):
            acc = np.zeros((mat_size, mat_size), dtype=dtype)
            for i in range(k):
                acc += blocks[k - i - 1] @ hats[i]
            blocks[k] = acc / k

        self._mat_adj_blocks_cache_numpy[output_order] = blocks
        return blocks

    def _mat_inv_blocks_numpy(self, output_order: int) -> np.ndarray:
        cached = self._mat_inv_blocks_cache_numpy.get(output_order)
        if cached is not None:
            return cached

        mat0 = self._mat.mat_inv()
        dtype = mat0.dtype
        mat_size = self._mat_size
        blocks = np.zeros((output_order, mat_size, mat_size), dtype=dtype)

        if output_order == 0:
            self._mat_inv_blocks_cache_numpy[output_order] = blocks
            return blocks

        blocks[0] = mat0
        if output_order == 1:
            self._mat_inv_blocks_cache_numpy[output_order] = blocks
            return blocks

        hats = self._hat_series_numpy(output_order, adj=False, dtype=dtype)

        for k in range(1, output_order):
            acc = np.zeros((mat_size, mat_size), dtype=dtype)
            for i in range(k):
                acc -= hats[i] @ blocks[k - i - 1]
            blocks[k] = acc / k

        self._mat_inv_blocks_cache_numpy[output_order] = blocks
        return blocks

    def _mat_inv_adj_blocks_numpy(self, output_order: int) -> np.ndarray:
        cached = self._mat_inv_adj_blocks_cache_numpy.get(output_order)
        if cached is not None:
            return cached

        mat0 = self._mat.mat_inv_adj()
        dtype = mat0.dtype
        mat_size = self._mat_adj_size
        blocks = np.zeros((output_order, mat_size, mat_size), dtype=dtype)

        if output_order == 0:
            self._mat_inv_adj_blocks_cache_numpy[output_order] = blocks
            return blocks

        blocks[0] = mat0
        if output_order == 1:
            self._mat_inv_adj_blocks_cache_numpy[output_order] = blocks
            return blocks

        hats = self._hat_series_numpy(output_order, adj=True, dtype=dtype)

        for k in range(1, output_order):
            acc = np.zeros((mat_size, mat_size), dtype=dtype)
            for i in range(k):
                acc -= hats[i] @ blocks[k - i - 1]
            blocks[k] = acc / k

        self._mat_inv_adj_blocks_cache_numpy[output_order] = blocks
        return blocks

    @staticmethod
    def _lower_toeplitz_numpy(blocks: np.ndarray) -> np.ndarray:
        output_order = blocks.shape[0]
        if output_order == 0:
            return np.zeros((0, 0), dtype=blocks.dtype)

        mat_size = blocks.shape[1]
        if output_order < 6:
            mat = np.zeros((mat_size * output_order, mat_size * output_order), dtype=blocks.dtype)
            for i in range(output_order):
                blk = blocks[i]
                for j in range(i, output_order):
                    mat[mat_size*j:mat_size*(j+1), mat_size*(j-i):mat_size*(j-i+1)] = blk
            return mat

        idx = np.arange(output_order)
        diff = idx[:, None] - idx[None, :]
        mask = diff >= 0
        toeplitz_blocks = blocks[np.clip(diff, 0, None)]
        toeplitz_blocks = np.where(mask[..., None, None], toeplitz_blocks, 0)
        return toeplitz_blocks.transpose(0, 2, 1, 3).reshape(mat_size * output_order, mat_size * output_order)

    @staticmethod
    def _lower_tri_blocks_numpy(blocks: np.ndarray, col_scales: np.ndarray = None) -> np.ndarray:
        output_order = blocks.shape[0]
        if output_order == 0:
            return np.zeros((0, 0), dtype=blocks.dtype)

        mat_size = blocks.shape[2]
        if output_order < 4:
            mat = np.zeros((mat_size * output_order, mat_size * output_order), dtype=blocks.dtype)
            for i in range(output_order):
                for j in range(i + 1):
                    blk = blocks[i, j]
                    if col_scales is not None:
                        blk = blk * col_scales[j]
                    mat[mat_size*i:mat_size*(i+1), mat_size*j:mat_size*(j+1)] = blk
            return mat

        tri_blocks = blocks
        if col_scales is not None:
            tri_blocks = tri_blocks * col_scales[np.newaxis, :, np.newaxis, np.newaxis]

        idx = np.arange(output_order)
        mask = idx[:, None] >= idx[None, :]
        tri_blocks = np.where(mask[..., None, None], tri_blocks, 0)
        return tri_blocks.transpose(0, 2, 1, 3).reshape(mat_size * output_order, mat_size * output_order)

    def _hat_adj_series_numpy(self, output_order: int, dtype) -> np.ndarray:
        return self._hat_series_numpy(output_order, adj=True, dtype=dtype)

    def _tangent_table_numpy(self, output_order: int, hats: np.ndarray = None, dtype=None) -> np.ndarray:
        use_cache = hats is None and dtype is None
        if use_cache:
            cached = self._tangent_table_cache_numpy.get(output_order)
            if cached is not None:
                return cached

        if dtype is None:
            dtype = self._mat.mat_adj().dtype
        mat_size = self._mat_adj_size
        tangent = np.zeros((output_order, output_order, mat_size, mat_size), dtype=dtype)

        if output_order == 0:
            return tangent

        eye = np.eye(mat_size, dtype=dtype)
        tangent[0, 0] = eye
        if output_order == 1:
            return tangent

        if hats is None:
            hats = self._hat_adj_series_numpy(output_order, dtype)

        for i in range(1, output_order):
            tangent[i, i] = eye / i
            for j in range(i):
                acc = np.zeros((mat_size, mat_size), dtype=dtype)
                for k in range(i - j):
                    acc -= hats[k] @ tangent[i - k - 1, j]
                tangent[i, j] = acc / i

        if use_cache:
            self._tangent_table_cache_numpy[output_order] = tangent
        return tangent

    def _tangent_cm_table_numpy(self, output_order: int, hats: np.ndarray = None, dtype=None) -> np.ndarray:
        use_cache = hats is None and dtype is None
        if use_cache:
            cached = self._tangent_cm_table_cache_numpy.get(output_order)
            if cached is not None:
                return cached

        if dtype is None:
            dtype = self._mat.mat_adj().dtype
        mat_size = self._mat_adj_size
        tangent_cm = np.zeros((output_order, output_order, mat_size, mat_size), dtype=dtype)

        if output_order == 0:
            return tangent_cm

        eye = np.eye(mat_size, dtype=dtype)
        tangent_cm[0, 0] = eye
        if output_order == 1:
            return tangent_cm

        if hats is None:
            hats = self._hat_adj_series_numpy(output_order, dtype)

        for i in range(1, output_order):
            tangent_cm[i, i] = eye / i
            for j in range(i):
                acc = np.zeros((mat_size, mat_size), dtype=dtype)
                for k in range(j, i):
                    acc -= hats[i - 1 - k] @ tangent_cm[k, j]
                tangent_cm[i, j] = acc / i

        if use_cache:
            self._tangent_cm_table_cache_numpy[output_order] = tangent_cm
        return tangent_cm

    def _tangent_tables_numpy(self, output_order: int) -> tuple[np.ndarray, np.ndarray]:
        dtype = self._mat.mat_adj().dtype
        hats = self._hat_adj_series_numpy(output_order, dtype)
        tangent = self._tangent_table_numpy(output_order, hats=hats, dtype=dtype)
        tangent_cm = self._tangent_cm_table_numpy(output_order, hats=hats, dtype=dtype)
        return tangent, tangent_cm

    @staticmethod
    def _set_from_mat_blocks(T, mat_blocks: np.ndarray, LIB: str = 'numpy') -> 'CMTM':
        n = mat_blocks.shape[0]
        size = mat_blocks.shape[1]
        dof = T.dof()

        # SE3-specialized fast path for the dominant CMTM multiplication use-case.
        if T.__name__ == 'SE3' and size == 4 and dof == 6:
            m = T.set_mat(mat_blocks[0], LIB=LIB)
            vs = np.zeros((n - 1, 6), dtype=mat_blocks.dtype)

            if n == 1:
                return CMTM(m, vs, LIB=LIB)

            m_inv_mat = m.mat_inv() if hasattr(m, "mat_inv") else m.inv().mat()

            fact = np.ones(n, dtype=mat_blocks.dtype)
            for i in range(1, n):
                fact[i] = fact[i - 1] * i
            inv_fact = 1.0 / fact

            def _hat6(v: np.ndarray) -> np.ndarray:
                wx, wy, wz, vx, vy, vz = v
                h = np.zeros((4, 4), dtype=mat_blocks.dtype)
                h[0, 1] = -wz
                h[0, 2] = wy
                h[1, 0] = wz
                h[1, 2] = -wx
                h[2, 0] = -wy
                h[2, 1] = wx
                h[0, 3] = vx
                h[1, 3] = vy
                h[2, 3] = vz
                return h

            hats = [None] * (n - 1)
            for i in range(n - 1):
                m_tmp = np.zeros((4, 4), dtype=mat_blocks.dtype)
                for j in range(i):
                    hat_j = hats[j]
                    if hat_j is None:
                        hat_j = _hat6(vs[j] * inv_fact[j])
                        hats[j] = hat_j
                    m_tmp += mat_blocks[i - j] @ hat_j

                delta = m_inv_mat @ (mat_blocks[i + 1] * (i + 1) - m_tmp)
                vec = np.empty(6, dtype=mat_blocks.dtype)
                vec[0] = 0.5 * (delta[2, 1] - delta[1, 2])
                vec[1] = 0.5 * (delta[0, 2] - delta[2, 0])
                vec[2] = 0.5 * (delta[1, 0] - delta[0, 1])
                vec[3] = delta[0, 3]
                vec[4] = delta[1, 3]
                vec[5] = delta[2, 3]
                vs[i] = vec * fact[i]
                hats[i] = _hat6(vs[i] * inv_fact[i])

            return CMTM(m, vs, LIB=LIB)

        m = T.set_mat(mat_blocks[0])
        vs = np.zeros((n - 1, dof), dtype=mat_blocks.dtype)

        if n == 1:
            return CMTM(m, vs, LIB=LIB)

        m_inv_obj = m.inv()
        m_inv_mat = m_inv_obj.mat() if hasattr(m_inv_obj, "mat") else m_inv_obj

        # Precompute i! and 1/i! tables once for the reconstruction loop.
        fact = np.ones(n, dtype=mat_blocks.dtype)
        for i in range(1, n):
            fact[i] = fact[i - 1] * i
        inv_fact = 1.0 / fact

        hats = [None] * (n - 1)
        for i in range(n - 1):
            m_tmp = np.zeros((size, size), dtype=mat_blocks.dtype)
            for j in range(i):
                hat_j = hats[j]
                if hat_j is None:
                    hat_j = T.hat(vs[j] * inv_fact[j])
                    hats[j] = hat_j
                m_tmp += mat_blocks[i - j] @ hat_j
            vs[i] = T.vee(m_inv_mat @ (mat_blocks[i + 1] * (i + 1) - m_tmp)) * fact[i]
            hats[i] = T.hat(vs[i] * inv_fact[i])

        return CMTM(m, vs, LIB=LIB)
        
    def mat(self, output_order = None):
        if self._lib == 'jax':
            P      = self.__check_output_order(output_order)
            msize  = self._mat_size
            dtype  = self._vecs[0].dtype

            # 逆階乗テーブル (static 長)
            seq      = jnp.arange(1, P, dtype=dtype)
            fact     = jnp.cumprod(seq)                      # または lax.cumprod
            inv_fact = jnp.concatenate([jnp.ones(1, dtype), 1.0 / fact])

            # Ms[k] を格納するバッファ
            Ms = jnp.zeros((P, msize, msize), dtype).at[0].set(self._mat.mat())

            def outer(k, Ms):                             # k = 1 .. P-1
                def inner(i, acc):
                    term = Ms[k - i - 1] @ self._mat.hat(self._vecs[i] * inv_fact[i], LIB=self._lib)
                    return acc + term
                acc0 = jnp.zeros((msize, msize), dtype)
                tmp  = jax.lax.fori_loop(0, k, inner, acc0)
                return Ms.at[k].set(tmp / k)

            Ms = jax.lax.fori_loop(1, P, outer, Ms)       # 全 M_k 完成

            # ---- ブロック行列化（以前と同じ） ----
            idx   = jnp.arange(P)
            diff  = idx[:, None] - idx[None, :]
            mask  = diff >= 0
            blocks     = Ms[jnp.clip(diff, 0)]            # (P,P,m,m)
            zero_block = jnp.zeros_like(Ms[0])
            blocks     = jnp.where(mask[..., None, None], blocks, zero_block)
            big_mat = blocks.transpose(0, 2, 1, 3).reshape(P*msize, P*msize)
            return big_mat  
        elif self._lib == 'numpy':
            output_order = self.__check_output_order(output_order)
            cached = self._mat_matrix_cache_numpy.get(output_order)
            if cached is not None:
                return cached.copy()
            tmp = self._mat_blocks_numpy(output_order)
            mat = self._lower_toeplitz_numpy(tmp)
            self._mat_matrix_cache_numpy[output_order] = mat.copy()
            return mat
    
    def __mat_adj_elem(self, p : int):
        if p == 0:
            return self._mat.mat_adj()
        else:
            mat = np.zeros( (self._mat_adj_size, self._mat_adj_size) ) 
            for i in range(p):
                mat = mat + self.__mat_adj_elem(p-(i+1)) @ self._mat.hat_adj(self._cmvecs_obj().cm_vecs()[i])

            return mat / p
        
    def mat_adj(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        cached = self._mat_adj_matrix_cache_numpy.get(output_order)
        if cached is not None:
            return cached.copy()
        tmp = self._mat_adj_blocks_numpy(output_order)
        mat = self._lower_toeplitz_numpy(tmp)
        self._mat_adj_matrix_cache_numpy[output_order] = mat.copy()
        return mat

    @staticmethod
    def set_mat(T, mat : np.ndarray, LIB = 'numpy'):
        size = T.eye().mat().shape[0]
        if mat.shape[0] % size != 0:
            raise TypeError("Matrix size is not same as element matrix")
        if size == 0:
            raise TypeError("Element matrix size is zero")
     
        n = int(mat.shape[0] / size)

        tmp = np.zeros((n, size, size), dtype=mat.dtype)
        for i in range(n):
            for j in range(n-i):
                tmp[i] += mat[(j+i)*size:(j+i+1)*size, j*size:(j+1)*size]
            tmp[i] = tmp[i] / (n-i)

        return CMTM._set_from_mat_blocks(T, tmp, LIB=LIB)
    
    @staticmethod
    def eye(T, output_order = 3):
        return CMTM(T.eye(), np.zeros((output_order-1,T.dof())))
    
    @staticmethod
    def rand(T, output_order = 3):
        return CMTM(T.rand(), np.random.rand(output_order-1,T.dof()))  
    
    def elem_mat(self):
        return self._mat.mat()
    
    def elem_vecs(self, i):
        if(self._n - 1 > i ):
            if(self._vecs.ndim == 1):
                return self._vecs
            else:
                return self._vecs[i]
            
    def vecs(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        return self._vecs[:output_order-1]
    
    def cmvecs(self):
        return self._cmvecs_obj()
    
    def vecs_flatten(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        return self._vecs[:output_order-1].flatten()
    
    def inv(self) -> 'CMTM':
        if self._n < 0 :
            inv_cmvec = -self.mat_adj(output_order=self._n-1) @ self._cmvecs_obj().cm_vec()
            v = cmvec.CMVector.set_cmvecs(inv_cmvec.reshape(self._n-1, self._dof))
            return CMTM(self._mat.inv(), v.vecs())
        else:
            return CMTM.set_mat(type(self._mat), self.mat_inv(), LIB=self._lib)

    def __mat_inv_elem(self, p : int):
        if p == 0:
            return self._mat.mat_inv()
        else:
            mat = np.zeros( (self._mat_size, self._mat_size) ) 
            for i in range(p):
                mat = mat - self._mat.hat(self._cmvecs_obj().cm_vecs()[i]) @ self.__mat_inv_elem(p-(i+1))
                
            return mat / p
    
    def mat_inv(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        cached = self._mat_inv_matrix_cache_numpy.get(output_order)
        if cached is not None:
            return cached.copy()
        tmp = self._mat_inv_blocks_numpy(output_order)
        mat = self._lower_toeplitz_numpy(tmp)
        self._mat_inv_matrix_cache_numpy[output_order] = mat.copy()
        return mat
    
    def __mat_inv_adj_elem(self, p : int):
        if p == 0:
            return self._mat.mat_inv_adj()
        else:
            mat = np.zeros( (self._mat_adj_size, self._mat_adj_size) ) 
            for i in range(p):
                mat = mat - self._mat.hat_adj(self._cmvecs_obj().cm_vecs()[i]) @ self.__mat_inv_adj_elem(p-(i+1))
                
            return mat / p
    
    def mat_inv_adj(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        cached = self._mat_inv_adj_matrix_cache_numpy.get(output_order)
        if cached is not None:
            return cached.copy()
        tmp = self._mat_inv_adj_blocks_numpy(output_order)
        mat = self._lower_toeplitz_numpy(tmp)
        self._mat_inv_adj_matrix_cache_numpy[output_order] = mat.copy()
        return mat
    
    @staticmethod
    def __hat_func(hat, vecs):
        n = vecs.shape[0]
        m = hat(vecs[0]).shape[0]
        mat = np.zeros((m*n,m*n))
        for i in range(n):
            tmp = hat(vecs[i])
            for j in range(n-i):
                mat[m*i+m*j:m*(i+1)+m*j,m*j:m*(j+1)] = tmp
        return mat
    
    @staticmethod
    def hat(T, vecs):
        return CMTM.__hat_func(T.hat, vecs)

    @staticmethod
    def hat_adj(T, vecs):
        return CMTM.__hat_func(T.hat_adj, vecs)
    
    @staticmethod
    def __hat_cm_func(hat, vecs : cmvec.CMVector):
        n = vecs._n
        m = vecs._dim
        mat = np.zeros((m*n,m*n))
        for i in range(n):
            tmp = hat(vecs.cm_vecs()[i])
            for j in range(n-i):
                mat[m*i+m*j:m*(i+1)+m*j,m*j:m*(j+1)] = tmp
        return mat
    
    @staticmethod
    def hat_cm(T, vecs : cmvec.CMVector):
        return CMTM.__hat_cm_func(T.hat, vecs)

    @staticmethod
    def hat_cm_adj(T, vecs : cmvec.CMVector):
        return CMTM.__hat_cm_func(T.hat_adj, vecs)

    @staticmethod
    def __vee_func(dof, size : int, vee, mat):
        '''
        dof : dof of lie group
        size : size of matrix
        vee : vee function
        '''
        if size == 0:
            raise TypeError("Element matrix size is zero")
        n = dof
        m = int(mat.shape[0] / size)
        vecs = np.zeros((m,n))
        for i in range(m):
            tmp = np.zeros(n)
            for j in range(i,m):
                tmp += vee( mat[j*size:(j+1)*size, (j-i)*size:(j-i+1)*size] )
            vecs[i] = tmp / (m-i)
        return vecs

    @staticmethod
    def vee(T, hat_mat):
        return CMTM.__vee_func(T.dof(), T.mat_size(), T.vee, hat_mat)
    
    @staticmethod
    def vee_adj(T, hat_mat):
        return CMTM.__vee_func(T.dof(), T.mat_adj_size(), T.vee_adj, hat_mat)
    
    @staticmethod
    def __vee_cm_func(dof, size : int, vee, mat) -> cmvec.CMVector:
        '''
        dof : dof of lie group
        size : size of matrix
        vee : vee function
        '''
        if size == 0:
            raise TypeError("Element matrix size is zero")
        n = dof
        m = int(mat.shape[0] / size)
        vecs = np.zeros((m,n))
        for i in range(m):
            tmp = np.zeros(n)
            for j in range(i,m):
                tmp += vee( mat[j*size:(j+1)*size, (j-i)*size:(j-i+1)*size] )
            vecs[i] = tmp / (m-i) * math.factorial(i)
        return cmvec.CMVector(vecs)

    @staticmethod
    def vee_cm(T, hat_mat):
        return CMTM.__vee_cm_func(T.dof(), T.mat_size(), T.vee, hat_mat)
    
    @staticmethod
    def vee_cm_adj(T, hat_mat):
        return CMTM.__vee_cm_func(T.dof(), T.mat_adj_size(), T.vee_adj, hat_mat)
    
    @staticmethod
    def hat_commute(T, vecs):
        return CMTM.__hat_func(T.hat_commute, vecs)
    
    @staticmethod
    def hat_commute_adj(T, vecs):
        return CMTM.__hat_func(T.hat_commute_adj, vecs)
    
    @staticmethod
    def hat_cm_commute(T, vecs : cmvec.CMVector):
        return CMTM.__hat_cm_func(T.hat_commute, vecs)
    
    @staticmethod
    def hat_cm_commute_adj(T, vecs : cmvec.CMVector):
        return CMTM.__hat_cm_func(T.hat_commute_adj, vecs)

    def __tangent_mat_elem(self, i : int, j : int):
        mat = np.zeros( (self._mat_adj_size, self._mat_adj_size) )
        if i == 0:
            return np.eye( self._mat_adj_size ) 
        if i == j:
            return np.eye( self._mat_adj_size ) / j
        else:
            for k in range(i-j):
                if k > 0:
                    mat = mat - self._mat.hat_adj(self._vecs[k]/math.factorial(k)) @ self.__tangent_mat_elem(i-k-1, j) 
                else:
                    mat = mat - self._mat.hat_adj(self._vecs[k]) @ self.__tangent_mat_elem(i-k-1, j)
            return mat / i

    def tangent_mat(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        cached = self._tangent_matrix_cache_numpy.get(output_order)
        if cached is not None:
            return cached.copy()
        tangent = self._tangent_table_numpy(output_order)
        scales = np.ones(output_order, dtype=tangent.dtype)
        for j in range(2, output_order):
            scales[j] = 1 / math.factorial(j - 1)
        mat = self._lower_tri_blocks_numpy(tangent, col_scales=scales)
        self._tangent_matrix_cache_numpy[output_order] = mat.copy()
        return mat

    def tangent_mat_inv(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        return np.linalg.inv(self.tangent_mat(output_order))

    def __tangent_mat_cm_elem(self, i : int, j : int):
        mat = np.zeros( (self._mat_adj_size, self._mat_adj_size) )
        if i == 0:
            return np.eye( self._mat_adj_size ) 
        if i == j:
            return np.eye( self._mat_adj_size ) / i
        else:
            for k in range(j,i):
                mat = mat - self._mat.hat_adj(self._cmvecs_obj().cm_vecs()[i-1-k]) @ self.__tangent_mat_elem(k, j) 
            return mat / i

    def tangent_mat_cm(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        cached = self._tangent_cm_matrix_cache_numpy.get(output_order)
        if cached is not None:
            return cached.copy()
        tangent_cm = self._tangent_cm_table_numpy(output_order)
        mat = self._lower_tri_blocks_numpy(tangent_cm)
        self._tangent_cm_matrix_cache_numpy[output_order] = mat.copy()
        return mat

    def tangent_mat_cm_inv(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        A = np.zeros((self._mat_adj_size * output_order, self._mat_adj_size * output_order))
        B = np.eye(self._mat_adj_size * output_order) 

        A[self._mat_adj_size:, :-self._mat_adj_size] = self.hat_cm_adj(type(self._mat), self._cmvecs_obj())

        values = np.repeat(np.arange(np.ceil(output_order-1).astype(int)) + 1, self._mat_adj_size)[:output_order * self._mat_adj_size]
        np.fill_diagonal(B[self._mat_adj_size:, self._mat_adj_size:], values)

        return A + B

    @staticmethod
    def sub_vec(lval, rval, frame = 'bframe') -> np.ndarray: 
        if lval._n != rval._n:
            raise TypeError("Left operand should be same order in right operand")
        if lval._dof != rval._dof:
            raise TypeError("Left operand should be same dof in right operand")

        dof = lval._mat._dof
        vec = np.zeros((lval._n * dof))
        vec[:dof] = lval._mat.sub_tan_vec(lval._mat, rval._mat, frame)

        for i in range(1,lval._n):
            vec[dof*i:dof*(i+1)] = rval._vecs[i-1] - lval._vecs[i-1]

        return vec

    @staticmethod
    def sub_tan_vec(lval, rval, frame = 'bframe') -> np.ndarray:
        '''
        Subtract two variant CMTM objects in tangent space.
        '''
        if lval._n != rval._n:
            raise TypeError("Left operand should be same order in right operand")
        if lval._dof != rval._dof:
            raise TypeError("Left operand should be same dof in right operand")

        if frame == 'bframe':
            vec = lval.vee(type(lval._mat), lval.mat_inv() @ (rval.mat() - lval.mat()))
        elif frame == 'fframe':
            vec = lval.vee(type(lval._mat), (rval.mat() - lval.mat()) @ lval.mat_inv())
        
        return vec.flatten()

    def __matmul__(self, rval):
        if isinstance(rval, CMTM):
            if self._n == rval._n:
                if self._n > 4:
                    if self._lib == 'numpy' and rval._lib == 'numpy':
                        l_blocks = self._mat_blocks_numpy(self._n)
                        r_blocks = rval._mat_blocks_numpy(rval._n)
                        dtype = np.result_type(l_blocks.dtype, r_blocks.dtype)
                        mat_size = self._mat_size
                        out_blocks = np.zeros((self._n, mat_size, mat_size), dtype=dtype)

                        for k in range(self._n):
                            acc = np.zeros((mat_size, mat_size), dtype=dtype)
                            for i in range(k + 1):
                                acc += l_blocks[i] @ r_blocks[k - i]
                            out_blocks[k] = acc

                        return CMTM._set_from_mat_blocks(type(self._mat), out_blocks, LIB=self._lib)
                    # fallback for non-numpy backends
                    return CMTM.set_mat(type(self._mat), self.mat() @ rval.mat(), LIB=self._lib)
                m = self._mat @ rval._mat
                v = np.zeros((self._n-1,self._mat.dof()))
                if self._n > 1:
                    v[0] = rval._mat.mat_inv_adj() @ self._vecs[0] + rval._vecs[0]
                if self._n > 2:
                    v[1] = rval._mat.mat_inv_adj() @ self._vecs[1] + self._mat.hat_adj(rval._mat.mat_inv_adj() @ self._vecs[0]) @ rval._vecs[0] + rval._vecs[1]
                if self._n > 3:
                    # v[2] = rval._mat.mat_inv_adj() @ self._vecs[2] \
                    #         - 2 * self._mat.hat_adj(rval._vecs[0]) @ rval._mat.mat_inv_adj() @ self._vecs[1] \
                    #         + (- self._mat.hat_adj(rval._vecs[1]) + self._mat.hat_adj(rval._vecs[0]) @ self._mat.hat_adj(rval._vecs[0]) ) @ rval._mat.mat_inv_adj() @ self._vecs[0] \
                    #         + rval._vecs[2]
                    v[2] = rval._mat.mat_inv_adj() @ self._vecs[2] \
                            + 2 * self._mat.hat_adj(rval._mat.mat_inv_adj() @ self._vecs[1]) @ rval._vecs[0] \
                            + self._mat.hat_adj( self._mat.hat_adj(rval._mat.mat_inv_adj() @ self._vecs[0]) @ rval._vecs[0]) @ rval._vecs[0] \
                            + self._mat.hat_adj(rval._mat.mat_inv_adj() @ self._vecs[0]) @ rval._vecs[1] \
                            + rval._vecs[2]
                return CMTM(m, v)
            else:
                raise TypeError("Right operand should be same size in left operand")
        elif isinstance(rval, cmvec.CMVector):
            return cmvec.CMVector.set_cmvecs((self.mat_adj() @ rval.cm_vec()).reshape(self._n, self._mat_adj_size))
        elif isinstance(rval, np.ndarray):
            return self.mat() @ rval
        else:
            raise TypeError("Right operand should be CMTM or numpy.ndarray")

    def __repr__(self):
        return f"CMTM(\n\telem_mat=\n{self._mat},\n\telem_vecs=\n{self._vecs},\n\tLIB='{self._lib}'\n)"
    
    @classmethod
    def change_elemclass(cls, a : 'CMTM', elem_cls) -> 'CMTM':
        b = cls.__new__(cls)
        b.__dict__ = a.__dict__.copy()
        b_elem_mat = elem_cls.change_class(a._mat)
        cls.__init__(b, b_elem_mat, a._vecs, a._lib)
        return b
    
    def mat_var_x_arb_vec(self, 
                            arb_vec : cmvec.CMVector,
                            tan_var_vec : cmvec.CMVector,
                            frame : str = 'bframe') -> cmvec.CMVector:
        '''
        delta X @ arb_vec = X @ hat(tan_var_vec) @ arb_vec = X @ hat_commute(arb_vec) @ tan_var_vec  (bframe)
        delta X @ arb_vec = hat(tan_var_vec) @ X @ arb_vec = hat_commute(X @ arb_vec) @ tan_var_vec  (fframe)
        '''
        cls = type(self)
        cls_elem = type(self._mat) 
        if frame == 'bframe':
            return cmvec.CMVector.set_cmvecs((self.mat_adj() @ cls.hat_cm_commute_adj(cls_elem, arb_vec) @ tan_var_vec.cm_vec()).reshape(self._n, self._mat_adj_size))
        elif frame == 'fframe':
            return cmvec.CMVector.set_cmvecs((cls.hat_cm_commute_adj(cls_elem, self @ arb_vec) @ tan_var_vec.cm_vec()).reshape(self._n, self._mat_adj_size))

    def mat_var_x_arb_vec_jacob(self, arb_vec : cmvec.CMVector,
                           frame : str = 'bframe') -> cmvec.CMVector:
        '''
        delta X @ arb_vec = X @ hat(tan_var_vec) @ arb_vec = X @ hat_commute(arb_vec) @ tan_var_vec  (bframe)
        delta X @ arb_vec = hat(tan_var_vec) @ X @ arb_vec = hat_commute(X @ arb_vec) @ tan_var_vec  (fframe)
        returns: 
        X @ hat_commute(arb_vec) (bframe)
        hat_commute(X @ arb_vec) (fframe)
        '''
        cls = type(self)
        cls_elem = type(self._mat)
        if frame == 'bframe':
            return self.mat_adj() @ cls.hat_cm_commute_adj(cls_elem, arb_vec)
        elif frame == 'fframe':
            return cls.hat_cm_commute_adj(cls_elem, self @ arb_vec)
        
    def mat_inv_var_x_arb_vec_jacob(self, arb_vec : cmvec.CMVector,
                           frame : str = 'bframe') -> cmvec.CMVector:
        '''
        delta X^-1 @ arb_vec 
            = - X^-1 @ delta X @ X^-1 @ arb_vec
            = - X^-1 @ X @ hat(tan_var_vec) @ X^-1 @ arb_vec
            = -hat_commute(X^-1 @ arb_vec) @ tan_var_vec  (bframe)
        returns: 
        X @ hat_commute(arb_vec) (bframe)
        hat_commute(X @ arb_vec) (fframe)
        '''
        cls = type(self)
        cls_elem = type(self._mat)

        if frame == 'bframe':
            return -cls.hat_cm_commute_adj(cls_elem, self.inv() @ arb_vec)
        elif frame == 'fframe':
            raise NotImplementedError("Not implemented for fframe")
