class DataGenerator(object):
    def __init__(self, args):
        self.args = args
        return

    def generate_input_data(self,):
        raise NotImplementedError

    def generate_output_data(self,):
        raise NotImplementedError