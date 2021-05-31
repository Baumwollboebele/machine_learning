
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import time


#%%

def timing(function):
    def wrap(*args, **kwargs):
        start = time.time()
        ret = function(*args, **kwargs)
        end = time.time()
        print(f"Function took: {(end-start)/60} s")

        return ret
    return wrap

# %%
numbers = list(range(2,100))

# %%
@timing
def sieve(numbers):
    numbers = list(range(2,numbers))

    for i in numbers:
        for j in numbers:
            if j%i ==0 and j != i:
                numbers.remove(j)


# %%
sieve(100000)

# %%
