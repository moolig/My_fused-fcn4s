from __future__ import division
import caffe
import numpy as np
import os
import sys
from datetime import datetime
from PIL import Image
import pdb
import matplotlib.pyplot as plt
import cPickle

MIN_FLT = 1e-12
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)

    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, save_dir, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    if save_dir:
        os.mkdir(save_dir)
    dataset = 3358
    hist = np.zeros((n_cl, n_cl))
    loss = 0

    for test_it in range (dataset):
        net.forward()
        hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
                                net.blobs[layer].data[0].argmax(0).flatten(),
                                n_cl)

        loss += net.blobs['loss'].data.flat[0]
        
        print test_it
        
                       
    return hist, loss/dataset

def seg_tests(solver, save_format, layer='score', gt='label'):
    print '>>>', datetime.now(), 'Begin seg tests'
    solver.test_nets[0].share_with(solver.net)
    do_seg_tests(solver.test_nets[0], solver.iter, save_format, layer, gt)     
    

def do_seg_tests(net, iter, save_format, layer='score', gt='label'):
    if save_format:
        save_format = save_format.format(iter)
    hist, loss = compute_hist(net, save_format, layer, gt)
    
    print '>>>', datetime.now(), 'Iteration', iter, 'loss', loss
    
    # overall accuracy
    overall_acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', overall_acc
    # per-class accuracy
    
    temp_mean_acc = np.diag(hist) / hist.sum(1)
    mean_acc = np.nanmean(temp_mean_acc)
    print '>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(mean_acc)
            
    # per-class IU    
    mean_iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(mean_iu)
                
    return hist
      
