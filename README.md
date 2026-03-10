# Name Prospector
Name Prospector is a website that allows users to generate names based on language patterns.
The general idea is to generate names based on a certain learnable patterns in a set of names.  

You can access the website at https://nameprospector.sapphire-infosystems.com/

Name Prospector is useful for many different purposes. Here is a just few of them:
* Generating fantasy names that have a certain cultural "vibe"
* Generating fresh, new, baby names that match the constraints of a certain language
* Learning how certain patterns appear and repeat in languages 
* Cultivating name ideas in your mind to allow your creative side to do the rest
* Building your own models to generate whatever style of name you want, ie. "futuristic", "pirate", "country", "alien".  

### Creating a Model

There are two ways to create a model:
* The first option is to run the manual_model_builder.py file, using a text file.
* The second option is to use the website feature "Input Your Own Names".   

### List of Existing Models

The website currently has these pretrained models that can be used to generate names:
* Classic American
* New Age American
* French
* German
* Chinese
* Greek
* Russian
* Arabic
* Aztec
* Spanish
* Japanese 
* Tamil

These models are not currently available but are high priority:
* English
* Roman
* Thai
* Hawaiian

### How it Works 
The language patterns are learned by an LSTM-based model that is trained to learn bidirectional patterns and bigram commonality. After this, the model knows which characters to pick based on a sub-portion of a generated name. 
Name generation is tinkered by many generation parameters, many of which the user controls, such as Gender, Temperature, Prefix, and Length. There are also other parameters controlling penalties on ending trigrams, hyphens, and capital letters. 

### Sources:
Names in the text files including French, Greek, Russian, Arabic, Aztec, Spanish, and Japanese are sourced from BabyNames.com, with each respective page being found at https://babynames.com/{language}-baby-names. The list of all languages is here: https://babynames.com/baby-names-by-origin.php.
