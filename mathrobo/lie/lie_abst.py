

from typing import Union
from abc import ABC, abstractmethod

from ..basic import *

class LieAbstract(ABC):
    @abstractmethod
    def __init__(self, LIB : str = 'numpy'): 
        '''
        Constructor
        '''
        self._lib = LIB

    @property
    @abstractmethod
    def lib(self) -> str:
        '''
        Return the library used for the Lie group
        '''
        return self._lib

    @staticmethod
    @abstractmethod
    def dof() -> int:
        '''
        Return the degree of freedom of the Lie group
        '''
        pass

    @property
    @abstractmethod
    def mat_size(self) -> int:
        '''
        Return the size of the matrix representation of the Lie group
        '''
        pass

    @abstractmethod
    def mat(self) -> Union[np.ndarray, jnp.ndarray]:
        '''
        Return the matrix representation of the Lie group
        '''
        pass

    @abstractmethod
    def set_mat(self, mat : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> 'LieAbstract':
        '''
        Set the matrix representation of the Lie group
        '''
        pass
    
    @staticmethod
    @abstractmethod
    def hat(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        '''
        hat operator on the tanget space vector
        '''
        pass
    
    @staticmethod
    def hat_commute(vec  : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        '''
        hat commute operator on the tanget space vector
        hat(a) @ b = hat_commute(b) @ a 
        '''
        pass

    @staticmethod
    @abstractmethod
    def vee(vec_hat : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        '''
        a = vee(hat(a))
        '''
        pass  
    
    @staticmethod
    def exp(vec, a, LIB = 'numpy'):
        pass

    @staticmethod
    def exp_integ(vec, a, LIB = 'numpy'):
        pass
    
    @abstractmethod
    def inv(self):
        pass
    
    
    def mat_adj(self):
        '''
        adjoint expresion of Lie group
        '''
        pass
    
    @staticmethod
    def hat_adj(vec, LIB = 'numpy'):
        pass

    @staticmethod
    def hat_commute_adj(vec, LIB = 'numpy'):
        pass
    
    @staticmethod
    def exp_adj(vec, a, LIB = 'numpy'):
        pass
    
    @staticmethod
    def exp_integ_adj(vec, a, LIB = 'numpy'):
        pass

    def mat_var_x_arb_vec(self, 
                           arb_vec : Union[np.ndarray, jnp.ndarray],
                           tan_var_vec : Union[np.ndarray, jnp.ndarray],
                           frame : str = 'bframe') -> Union[np.ndarray, jnp.ndarray]:
        '''
        \delta X @ arb_vec = X @ hat(tan_var_vec) @ arb_vec = X @ hat_commute(arb_vec) @ tan_var_vec  (bframe)
        \delta X @ arb_vec = hat(tan_var_vec) @ X @ arb_vec = hat_commute(X @ arb_vec) @ tan_var_vec  (fframe)
        '''
        cls = type(self)
        if frame == 'bframe':
            return self.mat() @ cls.hat_commute_adj(arb_vec, self.lib) @ tan_var_vec
        elif frame == 'fframe':
            return cls.hat_commute_adj(self.mat() @ arb_vec, self.lib) @ tan_var_vec

    def mat_var_x_arb_vec_jacob(self, arb_vec : Union[np.ndarray, jnp.ndarray],
                           frame : str = 'bframe') -> Union[np.ndarray, jnp.ndarray]:
        '''
        \delta X @ arb_vec = X @ hat(tan_var_vec) @ arb_vec = X @ hat_commute(arb_vec) @ tan_var_vec  (bframe)
        \delta X @ arb_vec = hat(tan_var_vec) @ X @ arb_vec = hat_commute(X @ arb_vec) @ tan_var_vec  (fframe)

        @ returns: X @ hat_commute(arb_vec) (bframe)
                    hat_commute(X @ arb_vec) (fframe)
        '''
        cls = type(self)
        if frame == 'bframe':
            return self.mat() @ cls.hat_commute_adj(arb_vec, self.lib)
        elif frame == 'fframe':
            return cls.hat_commute_adj(self.mat() @ arb_vec, self.lib)