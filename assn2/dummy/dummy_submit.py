import numpy as np
import time as tm
import pickle
import warnings
import os

#%load_ext autoreload
#%autoreload 2

class Merlin:
	def __init__( self, query_max, words ):
		self.words = words
		self.num_words = len( words )
		self.secret = ""
		self.query_max = query_max
		self.arthur = None
		self.win_count = 0
		self.tot_query_count = 0
		self.rnd_query_count = 0
		
	def meet( self, arthur ):
		self.arthur = arthur
	
	def reset( self, secret ):
		self.secret = secret
		self.rnd_query_count = 0
	
	# Receive a message from Arthur
	# Process it and terminate the round or else message Arthur back
	# Arthur can set is_done to request termination of this round after this query
	def msg( self, query_idx, is_done = False ):
	
		# Supplying an illegal value for query_idx is a way for Arthur to request
		# termination of this round immediately without even processing the current query
		# However, this results in query count being set to max for this round
		if query_idx < 0 or query_idx > self.num_words - 1:
			warnings.warn( "Warning: Arthur has sent an illegal query -- terminating this round", UserWarning )
			self.tot_query_count += self.query_max
			return
		
		# Arthur has made a valid query
		# Find the guessed word and increase the query counter
		query = self.words[ query_idx ]
		self.rnd_query_count += 1
		
		# Find out the intersections between the query and the secret
		reveal = [ *( '_' * len( self.secret ) ) ]
		
		for i in range( min( len( self.secret ), len( query ) ) ):
			if self.secret[i] == query[i]:
				reveal[ i ] = self.secret[i]
		
		# The word was correctly guessed
		if '_' not in reveal:
			self.win_count += 1
			self.tot_query_count += self.rnd_query_count
			return
		
		# Too many queries have been made - terminate the round
		if self.rnd_query_count >= self.query_max:
			self.tot_query_count += self.rnd_query_count
			return
		
		# If Arthur is done playing, terminate this round
		if is_done:
			self.tot_query_count += self.rnd_query_countt
			return
		
		# If none of the above happen, continue playing
		self.arthur.msg( ' '.join( reveal ) )
	
	def reset_and_play( self, secret ):
		self.reset( secret )
		self.arthur.msg( ( "_ " * len( self.secret ) )[:-1] )

class Arthur:
	def __init__( self, model ):
		self.dt = model
		self.curr_node = self.dt.root
		self.merlin = None
		self.is_done = False
		
	def meet( self, merlin ):
		self.merlin = merlin
	
	def reset( self ):
		self.curr_node = self.dt.root
		self.is_done = False
	
	def msg( self, response ):
		# If we are not at a leaf, lets go to the appropriate child based on the response
		if not self.curr_node.is_leaf:
			self.curr_node = self.curr_node.get_child( response )
		# If we are at a leaf, we should reqeust Merlin to terminate the round after this query
		else:
			self.is_done = True
		
		# Either way, get the query to be sent to Merlin
		query = self.curr_node.get_query()
		self.merlin.msg( query, self.is_done )

def my_fit( words):
	model = Tree(min_leaf_size = 1, max_depth = 15)
	model.fit( words)
	return model

class Tree:
	def __init__(self, min_leaf_size, max_depth):
		self.root = None
		self.words = None
		self.min_leaf_size = min_leaf_size
		self.max_depth = max_depth
	
	def fit(self, words):
		self.words = words
		self.root = Node(depth = 0, parent = None)
		# The root is trained with all the words
		self.root.fit( all_words = self.words, my_words_idx = np.arange( len( self.words ) ), min_leaf_size = self.min_leaf_size, max_depth = self.max_depth)

class Node:
	# A node stores its own depth (root = depth 0), a link to its parent
	# A link to all the words as well as the words that reached that node
	# A dictionary is used to store the children of a non-leaf node.
	# Each child is paired with the response that selects that child.
	# A node also stores the query-response history that led to that node
	# Note: my_words_idx only stores indices and not the words themselves
	def __init__(self, depth, parent):
		self.depth = depth
		self.parent = parent
		self.all_words = None
		self.my_words_idx = None
		self.children = {}
		self.is_leaf = True
		self.query_idx = None
		self.leaf_num = 0
	
	# Each node must implement a get_query method that generates the
	# query that gets asked when we reach that node. Note that leaf nodes
	# also generate a query which is usually the final answer
	def get_query(self):
		return self.query_idx
	
	# Each non-leaf node must implement a get_child method that takes a
	# response and selects one of the children based on that response
	def get_child(self, response):
		# This case should not arise if things are working properly
		# Cannot return a child if I am a leaf so return myself as a default action
		if self.is_leaf:
			print( "Why is a leaf node being asked to produce a child? Melbot should look into this!!" )
			child = self
		else:
			# This should ideally not happen. The node should ensure that all possibilities
			# are covered, e.g. by having a catch-all response. Fix the model if this happens
			# For now, hack things by modifying the response to one that exists in the dictionary
			if response not in self.children:
				print( f"Unknown response {response} -- need to fix the model" )
				response = list(self.children.keys())[0]
			
			child = self.children[ response ]
			
		return child
	
	# Dummy leaf action -- just return the first word
	def process_leaf(self, my_words_idx):
		return my_words_idx[0]
	
	def reveal(self, word, query):
		# Find out the intersections between the query and the word
		mask = [ *( '_' * len( word ) ) ]
		
		for i in range( min( len( word ), len( query ) ) ):
			if word[i] == query[i]:
				mask[i] = word[i]
		
		return ' '.join( mask )
	
	def calculate_entropy(self, freq_candidate):
		num_elements = np.sum(freq_candidate)
		# An empty or singular set has zero entropy
		if num_elements<=1:
			return 0
		return np.sum((freq_candidate/num_elements) * np.log2(freq_candidate))
	
	def check_against_candidates(self, all_words, my_words_idx, idx):
		# We want to store the number of instances where the particular unique response (mask) is generated
		count_unique_responses = {}
		for candidate in my_words_idx:
			mask = self.reveal(all_words[candidate], all_words[idx])
			if mask not in count_unique_responses:
				count_unique_responses[mask] = 0
			count_unique_responses[mask]+=1
		return self.calculate_entropy(np.array(list(count_unique_responses.values())))
	
	def determine_query_idx(self, all_words, my_words_idx):
		return my_words_idx[np.argmin([self.check_against_candidates(all_words, my_words_idx,idx) for idx in my_words_idx])]
	
	# Used to provide splitting criterion at each node (except leaf)
	# Entropy reduction (Information Gain) based splitting applied
	def process_node(self, all_words, my_words_idx):
		# For the root we do not ask any query -- Melbot simply gives us the length of the secret word
		if self.depth == 0:
			query_idx = -1
			query = ""
		else:
			query_idx = self.determine_query_idx(all_words, my_words_idx)
			query = all_words[query_idx]

		# We want to store the indices of words that generate the particular unique response (mask)
		store_unique_responses = {}
		for idx in my_words_idx:
			mask = self.reveal(all_words[idx], query)
			if mask not in store_unique_responses:
				store_unique_responses[mask] = []
			store_unique_responses[mask].append(idx)
		return (query_idx, store_unique_responses)
	
	def fit(self, all_words, my_words_idx, min_leaf_size, max_depth):
		self.all_words = all_words
		self.my_words_idx = my_words_idx

		# If the node is too small or too deep, make it a leaf
		# In general, can also include purity considerations into account
		if len(my_words_idx) <= min_leaf_size or self.depth >= max_depth:
			self.is_leaf = True
			self.query_idx = self.process_leaf(self.my_words_idx)
		else:
			self.is_leaf = False
			(self.query_idx, split_dict) = self.process_node(self.all_words, self.my_words_idx)

			for ( i, ( response, split ) ) in enumerate( split_dict.items() ):
				# Create a new child for every split
				self.children[response] = Node(depth = self.depth + 1, parent = self)
				# Recursively train this child node
				self.children[response].fit(self.all_words, split, min_leaf_size, max_depth)

def compFreq(words):
	freq_array = np.zeros((12, 26,15))
	for i in range(4,16):	
		for word in words:
			if(len(word) == i):
				pos = 0; 
				for ch in word : 
					ind1 = ord(ch) - 97
					freq_array[i-4,ind1, pos] = freq_array[i-4,ind1, pos]  +1
					pos = pos + 1
	#print(freq_array)

def main():
	with open( r"C:\Users\saksh\OneDrive\Desktop\Life at IITK\Acad\Coursework\8th sem\CS771\assn2\assn2\dummy\dict_secret", 'r' ) as f:
		words = f.read().split( '\n' )[:-1]		# Omit the last line since it is empty
		num_words = len( words )

	print(num_words)
	#compFreq(words)
	query_max = 15
	n_trials = 1

	t_train = 0
	m_size = 0
	win = 0
	query = 0

	for t in range( n_trials ):
		tic = tm.perf_counter()
		model = my_fit( words )
		toc = tm.perf_counter()
		t_train += toc - tic

	with open( f"model_dump_{t}.pkl", "wb" ) as outfile:
		pickle.dump( model, outfile, protocol=pickle.HIGHEST_PROTOCOL )
	
	m_size += os.path.getsize( f"model_dump_{t}.pkl" )

	merlin = Merlin( query_max, words )
	arthur = Arthur( model )
	merlin.meet( arthur )
	arthur.meet( merlin )
	
	for ( i, secret ) in enumerate( words ):
		arthur.reset()
		merlin.reset_and_play( secret )

	win += merlin.win_count / num_words
	query += merlin.tot_query_count / num_words

	t_train /= n_trials
	m_size /= n_trials
	win /= n_trials
	query /= n_trials

	print( t_train, m_size, win, query)

if __name__ == '__main__':
    main()
