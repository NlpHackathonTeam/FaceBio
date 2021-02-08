import time


def timer_wrap(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        start_t = time.time()
        ans = func(*args, **kwargs)
        stop_t = time.time()
        name = func.__name__
        if not hasattr(self, "_timers"):
            setattr(self, "_timers", {})
        if not name in self._timers:
            self._timers[name] = 0
        self._timers[name] += stop_t - start_t
        return ans
    return wrapper