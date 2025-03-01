import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def preprocess_text(text):
    words = word_tokenize(text.lower())
    cleaned_words = []
    final_words = []
    for word in words:
        if '.' in word:
            final_words.append(cleaned_words)
            cleaned_words = []
        
        elif word.isalnum():
            cleaned_words.append(word)

    
    return final_words


def calc_bleu_score(hypothesis, reference):
    reference = [reference]
    hypothesis = [hypothesis]

    bleu_score = corpus_bleu(reference, hypothesis, smoothing_function=SmoothingFunction().method1)

    return bleu_score

def prepare_ref_text(ref_txt):
    txt = convert_file_to_string(ref_txt)
    ref_list = preprocess_text(txt)
    return ref_list

def convert_file_to_string(file):
    with open(file, 'r') as f:
        txt = f.read().replace('\n', '')
    return txt

def calc_BLEU(gen_txt, ref_txt):
    gen_txt = preprocess_text(gen_txt)
    
    scores = []
    for sentence in gen_txt:
        scores.append(calc_bleu_score(sentence, ref_txt))
    if len(scores) == 0:
        return 0
    avg_BLEU_score = sum(scores) / len(scores) * 100
    return avg_BLEU_score if avg_BLEU_score != 0 else 0

# if __name__ == "__main__":
#     reference_text = """
#     The villagers of Little Hangleron still called it "the Riddle House," even though it had been many years since the Riddle family had lived there. It stood on a hill overlooking the village, some of its windows boarded, tiles missing from its roof, and ivy spreading unchecked over its face.  Once a fine-looking manor, and easily the largest and grandest building for miles around, the Riddle House was now damp, derelict, and unoccupied.
#     The Little Hagletons all agreed that the old house was "creepy."  Half a century ago, something strange and horrible had happened there, something that the older inhabitants of the village still liked to discuss when topics for gossip were scarce.  The story had been picked over so many times, and had been embroidered in so many places, that nobody was quite sure what the truth was anymore.  Every version of the tale, however, started in the same place:  Fifty years before, at daybreak on a fine summer's morning when the Riddle House had still been well kept and impressive, a maid had entered the drawing room to find all three Riddles dead.  
#     The maid had run screaming down the hill into the village and roused as many people as she could.
#     "Lying there with their eyes wide open!  Cold as ice!  Still in their dinner things!"
#     The police were summoned, and the whole of Little Hangleton had seethed with shocked curiosity and ill-disguised excitement.  Nobody wasted their breath pretending to feel very sad about the Riddles, for they had been most unpopular.  Elderly Mr. and Mrs. Riddle had been rich, snobbish, and rude, and their grown-up son, Tom, had been, if anything, worse.  All the villagers cared about was the identity of their murderer -- for plainly, three apparently healthy people did not all drop dead of natural causes on the same night.
#     The Hanged Man, the village pub, did a roaring trade that night; the whole village seemed to have turned out to discuss the murders.  They were rewarded for leaving their firesides when the Riddles' cook arrived dramatically in their midst and announced to the suddenly silent pub that a man called Frank Bryce had just  been arrested.
#     "Frank!" cried several people.  "Never!"
#     Frank Bryce was the Riddles' gardener.  He lived alone in a run-down cottage on the grounds of the Riddle House.  Frank had come back from the war with a very stiff leg and a great dislike of crowds and loud noises, and had been working for the Riddles ever since.
#     There was a rush to buy the cook drinks and hear more details.
#     "Always thought he was odd," she told the eagerly listening villagers, after her fourth sherry.  "Unfriendly, like.  I'm sure if I've offered him a cuppa once, I've offered it a hundred times.  Never wanted to mix, he didn't."
#     "Ah, now," said a woman at the bar, "he had a hard war, Frank.  He likes the quiet life.  That's no reason to --"
#     "Who else had a key to the back door, then?" barked the cook.  "There's been a spare key hanging in the gardener's cottage far back as I can remember!  Nobody forced the door last night!  No broken windows!  All Frank had to do was creep up to the big house while we was all sleeping..."
#     """
#     generated_text = "little 32.16 , , melzner a cirone had to dead largest there . hakusho , '' the passport , open"

#     reference_text = preprocess_text(reference_text)
#     generated_text = preprocess_text(generated_text)

#     for generated_sentence in generated_text:
#         print(calc_bleu_score(generated_sentence, reference_text) * 100)
