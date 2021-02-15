import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load('en_core_web_sm') # load english language model
# doc = nlp("Some text to process")
doc = nlp("Tea is heatlhy and calming, don't you think?")


### TOKENIZING ###

for token in doc:
    print(token)

print(f"Token \t\tLemma \t\tStopword".format('Token', 'Lemma', 'Stopword'))
print("-"*40)
for token in doc:
    print(f"{str(token)}\t\t{token.lemma_}\t\t{token.is_stop}") 
# .lemma_ returns the lemma (base form of a word)
# .is_stop returns a boolean true is the token is a stopword (frequent words that do not contain much info, e.g. the, is, and, but, not)


### PATTERN MATCHING ###

# match tokens/phrases within chunks of text or whole documents
matcher = PhraseMatcher(nlp.vocab, attr = 'LOWER') # match phrases on lowercased text (case insensitive matching)

# list of terms to match in the text
terms = ['Galaxy Note, iPhone 11', 'iPhone XS', 'Google Pixel']
patterns = [nlp(text) for text in terms]
matcher.add("TerminologyList", patterns)

# use phrase matcher to find where the terms occur in the text
text_doc = nlp("Glowing review overall, and some really interestin sidy by side photography " 
                "tests pitting the iPhone 11 Pro against the Galaxy Note 10 Plus and last "
                "year's iPhone XS and Google Pixel 3.")
matches = matcher(text_doc) # tuple of match id and positions of start and end of phrase
print(matches)
match_id, start, end = matches[0]
print(nlp.vocab.strings[match_id], text_doc[start:end])