class Percept:
    def __init__(self, percept: tuple):
        self._step = None
        self._return = None  # this will hold the return G_t
        self._state, self._action, self._reward, self._next_state, self._done = percept

    # R(s, a, s') in state s neem je actie a en komt terecht in s'
    # hier hoort een reward bij
    # het transitiemodel is de kans dat je in s' terecht komt gegeven je in state s bevindt en actie a neemt
    # P(s' | s, a)
    @property
    def state(self):
        return self._state

    @property
    def action(self):
        return self._action

    @property
    def reward(self):
        return self._reward

    @property
    def next_state(self):
        return self._next_state

    @property
    def done(self):
        return self._done

    @property
    def return_(self):
        return self._return

    @return_.setter
    def return_(self, value):
        self._return = value

    def __repr__(self):
        # uses SARS - format as a convention
        return '<in {} do {} get {} -> {}>'.format(self._state, self._action, self._reward, self._next_state)

    def __hash__(self):
        return hash((self.state, self.action, self.reward, self.next_state))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.state == other.state and self.action == other.action and self.reward == other.reward and self.next_state == other.next_state
