import sys
import caffe
import surgery, score
import numpy as np
import os

import pdb

#import setproctitle
#setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = "/path/to/pretrained_weights.caffemodel"

            
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

       
niter = 60000

test_interval = 1000

for it in range(niter+1): 
     
    solver.step(1) 
    
    if (it%test_interval == 0):
        score.seg_tests(solver, False, layer='score')
    
   
