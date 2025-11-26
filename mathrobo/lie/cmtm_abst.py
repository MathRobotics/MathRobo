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
        self._cmvecs = cmvec.CMVector(elem_vecs)
        self._dof = elem_mat.mat_adj().shape[0]
        self._mat_size = elem_mat.mat().shape[0]
        self._mat_adj_size = elem_mat.mat_adj().shape[0]
        self._n = elem_vecs.shape[0] + 1
        self._size = self._mat_size * self._n
        self._adj_size = self._mat_adj_size * self._n
        self._lib = LIB

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
                    mat = mat + self.__mat_elem(p-(i+1)) @ self._mat.hat(self._cmvecs.cm_vecs()[i])

                return mat / p
        
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
            
            mat = np.eye(self._mat_size * output_order)

            tmp = np.zeros((output_order, self._mat_size, self._mat_size))
            for i in range(output_order):
                tmp[i] = self.__mat_elem(i)

            for i in range(output_order):
                for j in range(i, output_order):
                        mat[self._mat_size*j:self._mat_size*(j+1),self._mat_size*(j-i):self._mat_size*(j-i+1)] = tmp[i]

            return mat
    
    def __mat_adj_elem(self, p : int):
        if p == 0:
            return self._mat.mat_adj()
        else:
            mat = np.zeros( (self._mat_adj_size, self._mat_adj_size) ) 
            for i in range(p):
                mat = mat + self.__mat_adj_elem(p-(i+1)) @ self._mat.hat_adj(self._cmvecs.cm_vecs()[i])

            return mat / p
        
    def mat_adj(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        
        mat = np.eye(self._mat_adj_size * output_order)

        tmp = np.zeros((output_order, self._mat_adj_size, self._mat_adj_size))
        for i in range(output_order):
            tmp[i] = self.__mat_adj_elem(i)

        for i in range(output_order):
            for j in range(i, output_order):
                    mat[self._mat_adj_size*j:self._mat_adj_size*(j+1),self._mat_adj_size*(j-i):self._mat_adj_size*(j-i+1)] = tmp[i]

        return mat

    @staticmethod
    def set_mat(T, mat : np.ndarray, LIB = 'numpy'):
        size = T.eye().mat().shape[0]
        if mat.shape[0] % size != 0:
            raise TypeError("Matrix size is not same as element matrix")
        if size == 0:
            raise TypeError("Element matrix size is zero")
     
        n = int(mat.shape[0] / size)

        tmp = np.zeros((n, size, size))
        for i in range(n):
            for j in range(n-i):
                tmp[i] += mat[(j+i)*size:(j+i+1)*size, j*size:(j+1)*size]
            tmp[i] = tmp[i] / (n-i)
            
        m = T.set_mat(tmp[0])
        vs = np.zeros((n-1, T.dof()))

        for i in range(n-1):
            m_tmp = np.zeros((size, size))
            for j in range(i):
                m_tmp += tmp[i-j] @ T.hat(vs[j]/math.factorial(j))
            vs[i] = T.vee( m.inv() @  ( tmp[i+1] * (i+1) - m_tmp) ) * math.factorial(i)

        return CMTM(m, vs)
    
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
        return self._cmvecs
    
    def vecs_flatten(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        return self._vecs[:output_order-1].flatten()
    
    def inv(self):
        vecs = np.zeros_like(self._vecs)
        if self._n < 0:
            for i in range(self._n-1):
                vecs[i] = -self._mat.mat_adj() @ self._vecs[i]
            return CMTM(self._mat.inv(), vecs)
        else:
            # tentative implementation
            return CMTM.set_mat(type(self._mat), self.mat_inv())

    def __mat_inv_elem(self, p : int):
        if p == 0:
            return self._mat.mat_inv()
        else:
            mat = np.zeros( (self._mat_size, self._mat_size) ) 
            for i in range(p):
                mat = mat - self._mat.hat(self._cmvecs.cm_vecs()[i]) @ self.__mat_inv_elem(p-(i+1))
                
            return mat / p
    
    def mat_inv(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        
        mat = np.eye(self._mat_size * output_order)

        tmp = np.zeros((output_order, self._mat_size, self._mat_size))
        for i in range(output_order):
            tmp[i] = self.__mat_inv_elem(i)

        for i in range(output_order):
            for j in range(i, output_order):
                    mat[self._mat_size*j:self._mat_size*(j+1),self._mat_size*(j-i):self._mat_size*(j-i+1)] = tmp[i]

        return mat
    
    def __mat_inv_adj_elem(self, p : int):
        if p == 0:
            return self._mat.mat_inv_adj()
        else:
            mat = np.zeros( (self._mat_adj_size, self._mat_adj_size) ) 
            for i in range(p):
                mat = mat - self._mat.hat_adj(self._cmvecs.cm_vecs()[i]) @ self.__mat_inv_adj_elem(p-(i+1))
                
            return mat / p
    
    def mat_inv_adj(self, output_order = None):
        output_order = self.__check_output_order(output_order)

        mat = np.eye(self._mat_adj_size * output_order)

        tmp = np.zeros((output_order, self._mat_adj_size, self._mat_adj_size))
        for i in range(output_order):
            tmp[i] = self.__mat_inv_adj_elem(i)

        for i in range(output_order):
            for j in range(i, output_order):
                    mat[self._mat_adj_size*j:self._mat_adj_size*(j+1),self._mat_adj_size*(j-i):self._mat_adj_size*(j-i+1)] = tmp[i]

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
        mat = np.zeros((self._mat_adj_size * output_order, self._mat_adj_size * output_order))

        for i in range(output_order):
            for j in range(i+1):
                mat[self._mat_adj_size*i:self._mat_adj_size*(i+1),self._mat_adj_size*j:self._mat_adj_size*(j+1)] = self.__tangent_mat_elem(i, j)
                if j > 1:
                    mat[self._mat_adj_size*i:self._mat_adj_size*(i+1),self._mat_adj_size*j:self._mat_adj_size*(j+1)] *= 1/math.factorial(j-1)

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
                mat = mat - self._mat.hat_adj(self._cmvecs.cm_vecs()[i-1-k]) @ self.__tangent_mat_elem(k, j) 
            return mat / i

    def tangent_mat_cm(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        mat = np.zeros((self._mat_adj_size * output_order, self._mat_adj_size * output_order))

        for i in range(output_order):
            for j in range(i+1):
                mat[self._mat_adj_size*i:self._mat_adj_size*(i+1),self._mat_adj_size*j:self._mat_adj_size*(j+1)] = self.__tangent_mat_cm_elem(i, j)

        return mat

    def tangent_mat_cm_inv(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        A = np.zeros((self._mat_adj_size * output_order, self._mat_adj_size * output_order))
        B = np.eye(self._mat_adj_size * output_order) 

        A[self._mat_adj_size:, :-self._mat_adj_size] = self.hat_cm_adj(type(self._mat), self._cmvecs)

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
                    # tentative implementation
                    return CMTM.set_mat(type(self._mat), self.mat() @ rval.mat())
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
        return f"CMTM(\nelem_mat=\n{self._mat},\nelem_vecs=\n{self._vecs},\nLIB='{self._lib}')"
    
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