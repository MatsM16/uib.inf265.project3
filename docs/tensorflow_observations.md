This visualization have me very little.  
The position in the visualizer has no correspondance to the words similarities to each other.  
This is inpart due to the word embeddings containing 32 dimensions, but are being squished down to three. (_Two in the image_)  
To highlight this, I have marked the word `surpised` and the five most similar words to it.

This is a problam that is very difficult to overcome, _(maybe impossible?)_ , as we observe the same lack of positional meaning when looking at the proper embeddings like `word2vec`.