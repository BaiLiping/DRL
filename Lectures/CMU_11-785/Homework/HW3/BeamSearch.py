def GreedySearch(SymbolSets, y_probs):
	'''
	SymbolSets: This is the list containing all the symbols i.e. vocabulary (without 				  blank)
	y_probs: Numpy array of (# of symbols+1,Seq_length,batch_size). Note that your 			   batch size for part 1 would always remain 1, but if you plan to use 			your implementation for part 2 you need to incorporate batch_size.

	Return the forward probability of greedy path and corresponding compressed symbol 	  sequence i.e. without blanks and repeated symbols.
	'''

def BeamSearch(SymbolSets, y_probs, BeamWidth):
	'''
	SymbolSets: This is the list containing all the symbols i.e. vocabulary (without 				  blank)
	
	y_probs: Numpy array of (# of symbols+1,Seq_length,batch_size). Note that your 			   batch size for part 1 would always remain 1, but if you plan to use 			your implementation for part 2 you need to incorporate batch_size.
	
	BeamWidth: Width of the beam.
	
	The function should return the symbol sequence with the best path score (forward 	  probability) and a dictionary of all the final merged paths with their scores. 
	'''
