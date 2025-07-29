

class Model:
    def __init__(self, monkey_name='beignet'):
        # You could include a constructor to initialize your model here, but all calls will be made to the load method
        self.monkey_name = monkey_name
        if self.monkey_name == 'beignet':
            print('Beignet')
        elif self.monkey_name == 'affi':
            print('Affi')
        else:
            raise ValueError(f'No such a monkey: {self.monkey_name}')

    def predict(self, X):
        # This method should accept an input of any size (of the given input format) and return predictions appropriately
        return X

    def load(self):
        # This method should load your pretrained model from wherever you have it saved
        pass
