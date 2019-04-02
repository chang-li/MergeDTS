import pickle


class AbstractDuel(object):
    def __init__(self):
        self.n_disp = int(1e6)

    def get_arms(self):
        raise NotImplementedError("Derived class needs to implement "
                                  "get_arms.")

    def update_scores(self, winner, loser):
        raise NotImplementedError("Derived class needs to implement "
                                  "update_scores.")

    def get_winner(self):
        raise NotImplementedError("Derived class needs to implement "
                                  "get_winner.")

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fname):
        with open(fname, 'rb') as f:
            sv = pickle.load(f)
        return sv


