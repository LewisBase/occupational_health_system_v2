import time
import warnings
import functools
from loguru import logger
 
def timer(func):
    @functools.wraps(func)    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute.")
        return result
    return wrapper


def deprecated(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(f"{func.__name__} is deprecated and will be removed in future versions.", DeprecationWarning)
        return func(*args, **kwargs)
    return wrapper