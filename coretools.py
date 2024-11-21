"""
General purpose tools.
"""
from contextlib import contextmanager
import gzip
import os
import unittest
import pathlib
import re
from typing import List, Optional

import numpy as np
import yaml

def ensure_np_array(arg) -> Optional[np.ndarray]:
    """Convert arg to np.array if necessary."""
    if arg is not None and not isinstance(arg, np.ndarray):
        return np.array(arg)
    return arg

def file_or_path(arg, mode: str):
    """Convert a path to an open file object if necessary."""
    if isinstance(arg, str):
        arg = open(arg, mode)
    if isinstance(arg, bytes):
        arg = open(str(arg), mode)
    if re.match(r".*\.gz", arg.name):
        arg = gzip.open(arg, mode)
    return arg

def import_numba_jit():
    """Return a numba.njit object. If import fails, return a fake jit object and emits a warning.

       Currently, the returned object can only be used as @njit (no option). It might be extended
       to allow more jit options.
    """
    try:
        from numba import njit as numba_jit
        return numba_jit
    except ImportError:
        import warnings
        warnings.warn("There was a problem importing numba, jit functions will work but will be MUCH slower.")
        def numba_jit(x):
            return x
        return numba_jit

class AttrDict(dict):
    '''Base class for "dual" objects both class instance and dictionary separately
       (namely, add class attributes/methods to a dictionary, or dict items to a class instance).
       
       Examples
       --------

       my_dict = {"fred": "male", "wilma": "female", "barney": "male"}
       a = AttrDict(my_dict)
       a.x = "Wilma"

       or

       a = AttrDict()
       a['fred'] = 'male'
       a.x = 'Wilma'
    
    '''
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    # def __repr__(self) -> str:
    #     if self.keys():
    #         m = max(map(len, list(self.keys()))) + 1
    #     # when used recursively, the inner repr is properly indented:
    #         return '\n'.join([k.rjust(m) + ': ' + re.sub("\n", "\n"+" "*(m+2), repr(v))
    #                           for k, v in sorted(self.items())])
    #     return self.__class__.__name__ + "()"

class Result(dict):
    # triple ' instead of triple " to allow using docstrings in the example
    '''Base class for "dual" objects both class instance and dictionary jointly,
       like those returning results.

       It allows one to create a return type that is similar to those
       created by `scipy.optimize.minimize`.
       The string representation of such an object contains a list
       of attributes and values and is easy to visualize on notebooks.

       Examples
       --------

       The simplest usage is this one:

       ```python
       from bussilab import coretools

       class MytoolResult(coretools.Result):
           """Result of a mytool calculation."""
           pass

       def mytool():
           a = 3
           b = "ciao"
           return MytoolResult(a=a, b=b)

       m=mytool()
       print(m)
       ```

       Notice that the class variables are dynamic: any keyword argument
       provided in the class constructor will be processed.
       If you want to enforce the class attributes you should add an explicit
       constructor. This will also allow you to add pdoc docstrings.
       The recommended usage is thus:

       ````
       from bussilab import coretools

       class MytoolResult(coretools.Result):
           """Result of a mytool calculation."""
           def __init__(a, b):
               super().__init__()
               self.a = a
               """Documentation for attribute a."""
               self.b = b
               """Documentation for attribute b."""

       def mytool():
           a = 3
           b = "ciao"
           return MytoolResult(a=a, b=b)

       m = mytool()
       print(m)
       ````

    '''

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, item: str, value):
        self[item] = value

    def __delattr__(self, item: str):
        del self[item]

    def __repr__(self) -> str:
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
# when used recursively, the inner repr is properly indented:
            return '\n'.join([k.rjust(m) + ': ' + re.sub("\n", "\n"+" "*(m+2), repr(v))
                              for k, v in sorted(self.items())])
        return self.__class__.__name__ + "()"

    def __dir__(self) -> List[str]:
        return list(sorted(self.keys()))

@contextmanager
def cd(newdir: os.PathLike, *, create: bool = False):
    """Context manager to temporarily change working directory.

       Can be used to change working directory temporarily making sure that at the
       end of the context the working directory is restored. Notably,
       it also works if an exception is raised within the context.

       Parameters
       ----------

       newdir : path
           Path to the desired directory.

       create : bool (default False)
           Create directory first.
           If the directory exists already, no error is reported

       Examples
       --------

       ```python
       from bussilab.coretools import cd
       with cd("/path/to/dir"):
           do_something() # this is executed in the /path/to/dir directory
       do_something_else() # this is executed in the original directory
       ```
    """
    prevdir = os.getcwd()
    path = os.path.expanduser(newdir)
    if create:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prevdir)


''' Preliminary tests for Python packages: 

    - name: Pyflakes
      run: |
        pip install --upgrade pyflakes
        pyflakes MDRefine
    - name: Pylint
      run: |
        pip install --upgrade  pylint
        pylint -E MDRefine
    - name: Flake8
      run: |
        pip install --upgrade flake8
        # stop the build if there are Python syntax errors or undefined names
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        # exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
        flake8 MDRefine bin --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics | tee flake8_report.txt

'''

class TestCase(unittest.TestCase):
    """Improved base class for test cases.

       Extends the `unittest.TestCase` class with some additional assertion.

    """
    class my_testcase(unittest.TestCase):
        def assertEqualObjs(self, obj1, obj2, tol = 1e-4, if_relative = False):
            
            import numpy as np
            import jax.numpy as jnp

            print(obj1, obj2)

            if isinstance(obj1, dict) and isinstance(obj2, dict):
                self.assertSetEqual(set(obj1.keys()), set(obj2.keys()))
                for k in obj1.keys():
                    self.assertEqualObjs(obj1[k], obj2[k])
            
            elif isinstance(obj1, list) and isinstance(obj2, list):
                self.assertEqual(len(obj1), len(obj2))
                for i in range(len(obj1)):
                    self.assertEqualObjs(obj1[i], obj2[i])
            
            elif isinstance(obj1, tuple) and isinstance(obj2, tuple):
                self.assertEqual(len(obj1), len(obj2))
                for i in range(len(obj1)):
                    self.assertEqualObjs(obj1[i], obj2[i])
            
            else:
                if (isinstance(obj1, np.ndarray) or isinstance(obj1, jnp.ndarray)) and (
                        isinstance(obj2, np.ndarray) or isinstance(obj2, jnp.ndarray)):

                    if if_relative == False:
                        q = np.sum((obj1 - obj2)**2)
                        print('value: ', q)
                        self.assertTrue(q < tol)
                    
                    else:

                        wh = np.argwhere(obj1 == 0)
                        if wh.shape[0] != 0:
                            q = np.sum((obj1[obj1 == 0] - obj2[obj1 == 0])**2)
                            print('value: ', q)
                            self.assertTrue(q < tol)

                        wh = np.argwhere(obj1 != 0)
                        if wh.shape[0] != 0:
                            q = np.sum(((obj2[obj1 != 0] - obj1[obj1 != 0])/obj1[obj1 != 0])**2)
                            print('value: ', q)
                            self.assertTrue(q < tol)

                elif isinstance(obj1, bool) and isinstance(obj2, bool):
                    self.assertTrue(obj1 == obj2)
                elif isinstance(obj1, float) and isinstance(obj2, float):
                    self.assertTrue((obj1 - obj2)**2 < tol)
                elif isinstance(obj1, int) and isinstance(obj2, int):
                    self.assertEqual(obj1, obj2)
                else:
                    print('WARNING: obj1 is ', type(obj1), 'while obj2 is ', type(obj2))
                    self.assertEqual(obj1, obj2)
    

    def assertEqualObjs_old(self, obj1, obj2):
        
        import numpy as np
        import jax.numpy as jnp

        print(obj1, obj2)

        if isinstance(obj1, dict) and isinstance(obj2, dict):
            self.assertSetEqual(set(obj1.keys()), set(obj2.keys()))
            for k in obj1.keys():
                self.assertEqualObjs(obj1[k], obj2[k])
        
        elif isinstance(obj1, list) and isinstance(obj2, list):
            self.assertEqual(len(obj1), len(obj2))
            for i in range(len(obj1)):
                self.assertEqualObjs(obj1[i], obj2[i])
        
        elif isinstance(obj1, tuple) and isinstance(obj2, tuple):
            self.assertEqual(len(obj1), len(obj2))
            for i in range(len(obj1)):
                self.assertEqualObjs(obj1[i], obj2[i])
        
        else:
            if (isinstance(obj1, np.ndarray) or isinstance(obj1, jnp.ndarray)) and (
                    isinstance(obj2, np.ndarray) or isinstance(obj2, jnp.ndarray)):
                self.assertAlmostEqual(np.sum((obj1 - obj2)**2), 0)
            elif isinstance(obj1, bool) and isinstance(obj2, bool):
                self.assertTrue(obj1 == obj2)
            elif isinstance(obj1, float) and isinstance(obj2, float):
                self.assertAlmostEqual(obj1, obj2)
            elif isinstance(obj1, int) and isinstance(obj2, int):
                self.assertEqual(obj1, obj2)
            else:
                print('WARNING: obj1 is ', type(obj1), 'while obj2 is ', type(obj2))
                self.assertEqual(obj1, obj2)


    def assertEqualFile(self, file1: os.PathLike, file2: Optional[os.PathLike] = None):
        """Check if two files are equal.

           Parameters
           ----------

           file1: path
               Path to the first file

           file2: path, optional
               Path to the second file. If not provided, defaults to `file1+".ref"`.
        """
        if file2 is None:
            file2 = pathlib.PurePath(str(file1)+".ref")

        try:
            f1=open(file1, "r")
        except FileNotFoundError:
            self.fail("file " +str(file1) + " was not found")

        try:
            f2=open(file2, "r")
        except FileNotFoundError:
            self.fail("file " +str(file2) + " was not found")

        with f1:
            with f2:
                self.assertEqual(f1.read(), f2.read())

def config_path(path: Optional[os.PathLike] = None):
    if path is None:
        path = pathlib.PurePath(os.environ["HOME"]+"/.bussilabrc")
    return path

def config(path: Optional[os.PathLike] = None):
    with open(config_path(path)) as rc:
        return yaml.load(rc,Loader=yaml.BaseLoader)


