import cPickle as pickle
import numpy as np
import time


class ReplayMemory:
    def __init__(self, path, max_size=1000000):
        self.max_size = max_size
        self.path = path

        mkdir_p(path)

        self._files = []
        for f in os.listdir(path):
            if f[-2:] == '.p':
                self._files.append(f)

    def sample(self):
        "Returns a random episode"
        i = np.random.randint(0, len(self._files))
        fn = os.path.join(self.path, self._files[i])
        return pickle.load(open(fn, "rb"))

    def store(self, ep):
        fn = self._make_filename()

        with open(fn, 'wb') as file:
            pickle.dump(ep, file)

        self._files.append(fn)

        # Storage limit
        if len(self._files) > self.max_size:
            # delete the oldest file.
            last_fn = self._files.pop(0)
            os.remove(last_fn)

    def _make_filename(self):
        # microsecond resolution. hopefully the os supports it.
        f = str(int(1000000 * time.time())) + '.p'
        return os.path.join(self.path, f)

    def count(self):
        return len(self._files)


import os
import errno


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


if __name__ == "__main__":
    # Test
    path = '/tmp/replay_memory_test'
    if os.path.isdir(path):
        import shutil
        shutil.rmtree(path)

    memory = ReplayMemory(path, 3)
    assert os.path.isdir(path)

    assert memory.count() == 0

    # Store

    ep = {'hello': 'world'}

    memory.store(ep)
    assert memory.count() == 1

    memory.store(ep)
    assert memory.count() == 2

    memory.store(ep)
    assert memory.count() == 3

    memory.store(ep)
    assert memory.count() == 3

    # Sample
    sampled_ep = memory.sample()
    assert sampled_ep == ep
