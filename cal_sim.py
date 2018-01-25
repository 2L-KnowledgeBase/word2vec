import os
import jieba
import tensorflow as tf
import scipy
import csv
#from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

def get_features(s):
	seg_list = jieba.cut(s)	
	return seg_list

with open(os.path.join('dict','words.dict'),"r",encoding='utf-8') as f:
    words_dict = f.read()
    
reverse_dictionary = eval(words_dict)
dictionary = {v: k for k, v in reverse_dictionary.items()}

#print_tensors_in_checkpoint_file("./model/model.w2c-70000", None, True)
checkpoint_file = tf.train.get_checkpoint_state("model")
saver = tf.train.import_meta_graph(checkpoint_file.model_checkpoint_path +".meta")
with tf.Session() as sess:
    saver.restore(sess, checkpoint_file.model_checkpoint_path)
    final_embeddings = sess.run('embeddings:0')
        
def list_avg(l):
	return 

def get_word_vect(name):
	vec = []
	for w in get_features(name):
		if w in dictionary:
			#print(dictionary[w])
			idx = dictionary[w]
		else:
			#print(dictionary['UNK'])
			idx = dictionary['UNK']
	
		#print(final_embeddings[idx,:])
		#print(type(final_embeddings[idx,:]))
		#print(final_embeddings[idx,:].size)
		vec.append([ x for x in final_embeddings[idx,:] ])
	
	return [ sum(x)/float(len(x)) for x in list(map(list, zip(*vec))) ]

def print_cos(a,b):
	a_v = get_word_vect(a)
	b_v = get_word_vect(b)
	
	print(a)
	print(b)
	print(scipy.spatial.distance.cosine(a_v ,b_v))
	
company_csv = csv.reader(open('/home/mark/export.csv'), delimiter=',', quotechar='"')

last_credit_cd = ''

for row in company_csv:
	#print(row)
	if row[3] == last_credit_cd:
		print_cos(row[0], last_row[0])		
	last_row = row
	last_credit_cd = row[3]


