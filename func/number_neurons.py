from math import floor
from io import StringIO 
import sys

def number_neurons(ni_minus1, n_input, n_output, n_layers):
    '''Number of neurons in each layer recursively. Formula taken from autopytorch-tabular paper'''
    if ni_minus1 == n_output:
        print("Finished")
    elif ni_minus1 == 0:
        print(floor(n_input))
        number_neurons(n_input, n_input, n_output, n_layers)    
    else:
        ni = ni_minus1 - ((n_input - n_output)/(n_layers-1))
        print(floor(ni))
        number_neurons(ni, n_input, n_output, n_layers)
        

class Capturing(list):
    '''Shoutout to https://stackoverflow.com/users/416467/kindall. Function taken from https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call'''
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

if __name__ == "__main__":
    with Capturing() as output:
        number_neurons(0, 100, 5, 5)

    print("Captured output: ", output)