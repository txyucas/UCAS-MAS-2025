class straight_agent():
    def __init__(self):
        self.force = 200
        self.angle = 0

    def act(self, obs):
        return [self.force, self.angle]