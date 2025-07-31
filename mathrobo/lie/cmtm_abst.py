from typing import TypeVar, Generic
from ..basic import *
import jax

T = TypeVar('T')

class CMTM(Generic[T]):
    def __init__(self, elem_mat, elem_vecs = np.array([]), LIB = 'numpy'): 
        '''
        Constructor
        '''
        self._mat = elem_mat
        self._vecs = elem_vecs
        self._dof = elem_mat.mat_adj().shape[0]
        self._mat_size = elem_mat.mat().shape[0]
        self._mat_adj_size = elem_mat.mat_adj().shape[0]
        self._n = elem_vecs.shape[0] + 1
        self._lib = LIB

    def __check_output_order(self, output_order : int):
        if output_order is None:
            output_order = self._n
        if output_order > self._n:
            raise TypeError("Output order should be less than or equal to the order of CMTM")
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
                    mat = mat + self.__mat_elem(p-(i+1)) @ self._mat.hat(self._vecs[i]/math.factorial(i))

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
            
            mat = identity(self._mat_size * output_order)

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
            mat = zeros( (self._mat_adj_size, self._mat_adj_size) ) 
            for i in range(p):
                mat = mat + self.__mat_adj_elem(p-(i+1)) @ self._mat.hat_adj(self._vecs[i]/math.factorial(i))
                
            return mat / p
        
    def mat_adj(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        
        mat = identity(self._mat_adj_size * output_order)

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
    
    def vecs_flatten(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        return self._vecs[:output_order-1].flatten()
    
    def ptan_vecs(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        v = np.zeros((output_order-1, self._dof))
        for i in range(output_order-1):
            v[i] = self._vecs[i]
            for j in range(i):
                v[i] += self._mat.hat_adj(self._vecs[j]) @ self._vecs[i-j-1]
        return v
    
    def ptan_vecs_flatten(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        return self.ptan_vecs(output_order).flatten()
        
    def tan_vecs(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        v = self.ptan_vecs(output_order)
        p = 1
        for i in range(output_order-1):
            v[i] = v[i] / p
            p = p * (i + 1)
        return v
    
    def tan_vecs_flatten(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        return self.tan_vecs(output_order).flatten()
            
    def inv(self):
        vecs = np.zeros_like(self._vecs)
        if self._n < 0:
            for i in range(self._n-1):
                vecs[i] = -self._mat.mat_adj() @ self._vecs[i]
            return CMTM(self._mat.inv(), vecs)
        else:
            # tentative implementation
            return CMTM.set_mat(type(self._mat), self.mat_inv())

    def __mat_inv_elem(self, p):
        if p == 0:
            return self._mat.mat_inv()
        else:
            mat = zeros( (self._mat_size, self._mat_size) ) 
            for i in range(p):
                mat = mat - self._mat.hat(self._vecs[i]/math.factorial(i)) @ self.__mat_inv_elem(p-(i+1))
                
            return mat / p
    
    def mat_inv(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        
        mat = identity(self._mat_size * output_order)

        tmp = np.zeros((output_order, self._mat_size, self._mat_size))
        for i in range(output_order):
            tmp[i] = self.__mat_inv_elem(i)

        for i in range(output_order):
            for j in range(i, output_order):
                    mat[self._mat_size*j:self._mat_size*(j+1),self._mat_size*(j-i):self._mat_size*(j-i+1)] = tmp[i]

        return mat
    
    def __mat_inv_adj_elem(self, p):
        if p == 0:
            return self._mat.mat_inv_adj()
        else:
            mat = zeros( (self._mat_adj_size, self._mat_adj_size) ) 
            for i in range(p):
                mat = mat - self._mat.hat_adj(self._vecs[i]/math.factorial(i)) @ self.__mat_inv_adj_elem(p-(i+1))
                
            return mat / p
    
    def mat_inv_adj(self, output_order = None):
        output_order = self.__check_output_order(output_order)

        mat = identity(self._mat_adj_size * output_order)

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
    def __vee_func(dof, size, vee, mat):
        '''
        dof : dof of lie group
        size : size of matrix
        vee : vee function
        '''
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

    def __ptan_map_elem(self, p : int):
        if p == 0:
            return identity( self._mat_adj_size ) 
        else:
            mat = zeros( (self._mat_adj_size, self._mat_adj_size) )
            for i in range(p):
                mat = mat - self.__ptan_map_elem(p-(i+1)) @ self._mat.hat_adj(self._vecs[i])
            return mat

    def ptan_map(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        mat = identity(self._mat_adj_size * output_order)
        
        tmp = np.zeros((output_order, self._mat_adj_size, self._mat_adj_size))
        for i in range(output_order):
            tmp[i] = self.__ptan_map_elem(i)

        for i in range(output_order):
            for j in range(i, output_order):
                    mat[self._mat_adj_size*j:self._mat_adj_size*(j+1),self._mat_adj_size*(j-i):self._mat_adj_size*(j-i+1)] = tmp[i]

        return mat
    
    def __ptan_map_inv_elem(self, p : int):
        if p == 0:
            return identity( self._mat_adj_size ) 
        else:
            mat = self._mat.hat_adj(self._vecs[p-1])
            return mat
    
    def ptan_map_inv(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        mat = identity(self._mat_adj_size * output_order)
        
        tmp = np.zeros((output_order, self._mat_adj_size, self._mat_adj_size))
        for i in range(output_order):
            tmp[i] = self.__ptan_map_inv_elem(i)

        for i in range(output_order):
            for j in range(i, output_order):
                    mat[self._mat_adj_size*j:self._mat_adj_size*(j+1),self._mat_adj_size*(j-i):self._mat_adj_size*(j-i+1)] = tmp[i]
                    
        return mat

    @staticmethod
    def ptan_to_tan(dof, output_order : int):
        '''
        Convert matrix the pseudo tangent vector to tangent vector.
        '''
        k = 1
        mat = np.zeros((dof*output_order, dof*output_order))
        for i in range(output_order):
            mat[i*dof:(i+1)*dof,i*dof:(i+1)*dof] = identity(dof) / k
            k = k * (i + 1)
        return mat

    @staticmethod
    def tan_to_ptan(dof, output_order : int):
        '''
        Convert the tangent vector to pseudo tangent vector.
        '''
        k = 1
        mat = np.zeros((dof*output_order, dof*output_order))
        for i in range(output_order):
            mat[i*dof:(i+1)*dof,i*dof:(i+1)*dof] = identity(dof) * k
            k = k * (i + 1)
        return mat

    def tan_map(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        return self.ptan_to_tan(self._mat_adj_size, output_order) @ self.ptan_map(output_order)

    def tan_map_inv(self, output_order = None):
        output_order = self.__check_output_order(output_order)
        return self.ptan_map_inv(output_order) @ self.tan_to_ptan(self._mat_adj_size, output_order)

    @staticmethod
    def sub_vec(lval, rval, type = 'bframe') -> np.ndarray: 
        if lval._n != rval._n:
            raise TypeError("Left operand should be same order in right operand")
        if lval._dof != rval._dof:
            raise TypeError("Left operand should be same dof in right operand")

        dof = lval._mat._dof
        vec = np.zeros((lval._n * dof))
        vec[:dof] = lval._mat.sub_tan_vec(lval._mat, rval._mat, type)

        for i in range(1,lval._n):
            vec[dof*i:dof*(i+1)] = rval._vecs[i-1] - lval._vecs[i-1]

        return vec
    
    @staticmethod
    def sub_ptan_vec(lval, rval, frame_type = 'bframe') -> np.ndarray: 
        '''
        Subtract the psuedu tangent vector of two CMTM objects.
        '''
        if lval._n != rval._n:
            raise TypeError("Left operand should be same order in right operand")
        if lval._dof != rval._dof:
            raise TypeError("Left operand should be same dof in right operand")

        dof = lval._mat._dof
        vec = np.zeros((lval._n * dof))
        vec[:dof] = lval._mat.sub_tan_vec(lval._mat, rval._mat, frame_type)
        for i in range(lval._n-1):
            vec[dof*(i+1):dof*(i+2)] = (rval._vecs[i] - lval._vecs[i])
            for j in range(i+1):
                vec[dof*(i+1):dof*(i+2)] += (lval._mat.hat_adj(vec[dof*j:dof*(j+1)]) @ lval._vecs[i-j])

        return vec
    
    @staticmethod
    def sub_tan_vec(lval, rval, frame_type = 'bframe') -> np.ndarray:
        return lval.ptan_to_tan(lval._mat.dof(), lval._n) @ CMTM.sub_ptan_vec(lval, rval, frame_type)

    @staticmethod
    def sub_tan_vec_var(lval, rval, frame_type = 'bframe') -> np.ndarray:
        '''
        Subtract two variant CMTM objects in tangent space.
        '''
        if lval._n != rval._n:
            raise TypeError("Left operand should be same order in right operand")
        if lval._dof != rval._dof:
            raise TypeError("Left operand should be same dof in right operand")

        if frame_type == 'bframe':
            vec = lval.vee(type(lval._mat), lval.mat_inv() @ (rval.mat() - lval.mat()))
        elif frame_type == 'fframe':
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
                TypeError("Right operand should be same size in left operand")
        elif isinstance(rval, np.ndarray):
            return self.mat() @ rval
        else:
            TypeError("Right operand should be CMTM or numpy.ndarray")

    def print(self):
        print("mat")
        print(self._mat.mat())
        print("vecs")
        print(self._vecs)
