

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
