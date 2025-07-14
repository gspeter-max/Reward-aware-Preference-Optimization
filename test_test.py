def in_notebook():
    try:
        from IPython import get_ipython
        print(get_ipython().config) 
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

print(in_notebook()) 

