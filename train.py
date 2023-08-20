##
# LIBRARIES
from __future__ import print_function

from options import Options
from dataset import load_vib
from lib.model import BiGAN
from lib.my_model import BiVi

##
def train():
    """ Training
    """

    ##
    # ARGUMENTS
    opt = Options().parse()
    ##
    # LOAD DATA
    dataloader = load_vib(opt)
    
    sample = dataloader['test'][0]
    
    print(sample.keys())
    
    # # LOAD MODEL
    # model = BiVi(opt, dataloader)
    # ##
    # # TRAIN MODEL
    # model.train()

if __name__ == '__main__':
    train()