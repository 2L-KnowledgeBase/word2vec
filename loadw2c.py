#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from six.moves import xrange 
with open(os.path.join('voc','vocabulary.dic'),"r",encoding='utf-8') as f:
    voc = f.read()
    
#print(voc)
reverse_dictionary = eval(voc)
print(type(reverse_dictionary))
#print(voc)
#embeddings = tf.Variable(tf.random_uniform([len(reverse_dictionary), 128], -1.0, 1.0),name = 'embeddings')
print_tensors_in_checkpoint_file("./model/model.w2c-400000", None, True)
checkpoint_file = tf.train.get_checkpoint_state("model")
saver = tf.train.import_meta_graph(checkpoint_file.model_checkpoint_path +".meta")
with tf.Session() as sess:
    # Load the saved meta graph and restore variables
    #print("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file.model_checkpoint_path)
    #print(saver)
    final_embeddings = sess.run('embeddings:0')
        
        

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
        
    plt.savefig(filename)

# pylint: disable=g-import-not-at-top
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
#zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\msyh.ttc')
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 500
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [reverse_dictionary[i] for i in xrange(plot_only)]
plot_with_labels(low_dim_embs, labels)
