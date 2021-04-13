import time

def time_decorator(fun):
    """
    Decorate a function with a simple timing routine
    """

    def inner(*args, **kwargs):

        # Take time before
        t1 = time.time()

        # Call function
        fun(*args, **kwargs)

        # Take time after
        t2 = time.time()

        # Print total time
        s = "\n=============================="
        s += f"\nIn function '{fun.__name__}'"
        s += f"\nTime taken = {t2-t1:.2E} s"
        s += "\n=============================="
        print(s)

    return inner

if __name__ == "__main__":

    @time_decorator
    def test(s):
        print(s)

    test("This is a test on how this decorator works")
