import sharedmem as sm

def my_func(i):
    sm.shared_data[i, i] = i * 10
    result = sum(sm.shared_data[i,:])
    return result
