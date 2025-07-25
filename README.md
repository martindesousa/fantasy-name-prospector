# Fantasy Names Generator
Fantasy Names Generator is a website that allows users to generate names based on language patterns.
The general idea is to generate names based on a certain learnable patterns in a set of names.  

Fantasy Names Generator is useful for many different purposes. Here is a just few of them:
* Generating fantasy names that have a certain cultural "vibe"
* Generating fresh, new, baby names that match the constraints of a certain language
* Learning how certain patterns appear and repeat in languages 
* Cultivating name ideas in your mind to allow your creative side to do the rest
* Building your own models to generate whatever style of name you want, ie. "futuristic", "pirate", "country", "alien".  

The website currently has these pretrained models that can be used to generate names:
* American
* French
* German
* Chinese
* Greek
* Russian
* More will be coming!

There are two ways to create a model:
* The first option is to run the manual_model_builder.py file, using a text file listing names.  
* The second option is to use the website feature "Input Your Own Names".

The language patterns are learned by an LSTM-based model that is trained to learn bidirectional patterns and bigram commonality. 


