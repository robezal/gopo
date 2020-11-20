from flask import Flask, render_template
from flask import request

import numpy as np
import pandas as pd
import keras
import eng_to_ipa as ipa
from keras import layers


DEFAULT_EPOCHS = 10
DEFAULT_WORDS = 600
DEFAULT_TEMP = 1.0

num_epochs = 0
num_words = 0
temperature = 0

app = Flask(__name__)

df = pd.read_csv('Multiple_Forms.csv', index_col = 0)
df.head()


poems = df['Poem'][df.Form == 'Sonnet']
poems


text = ''
for poem in poems:
    text += str(poem) + '\n'
len(text)


text = text.replace("’", "'")
text = text.replace("‘", "'")
text = text.replace("“", " ")
text = text.replace("'d", "ed")
text = text.replace("(", "")
text = text.replace(")", "")
text = text.replace(" '", " ")
text = text.replace("' ", " ")
text = text.replace('"', '')
text = text.replace("--", " ")
text = text.replace(":-", " ")
text = text.replace("-:", " ")
text = text.replace(".-", " ")
text = text.replace("- ", " ")
text = text.replace(" -", " ")
text = text.replace(" –", " ")
while text.find(" . ") != -1:
    text = text.replace(" . ", " ")
while text.find("..") != -1:
    text = text.replace("..", ".")
text = text.replace("?.", "? ")
text = text.replace("!.", "! ")
text = text.replace("!-", "! ")
text = text.replace("!—", "! ")
text = text.replace(".", " ")
text = text.replace(",", " ")
text = text.replace("\n"," eol\n")
text = text.replace("\r"," ")
text = text.replace("bad—the", "bad the")
text = text.replace("occasion—that", "occasion that")


word_list = text.split()
a = 0
roman_num = ['ii','iii','iv','v','vi','vii','viii','ix','x','xi','xii','xiii','xiv','xv','xvi','xvii',
             'xviii','xix','xx','xxi','xxii','xxiii','xxiv','xxv','xxvi','xxvii','xxviii','xxix','xxx']
for i in range(len(word_list) - a):
    word_list[i - a] = word_list[i - a].lower().strip("“”''-.?:;!,[]~`’—&")
    if word_list[i - a] in ['nan','','*'] or word_list[i-a][0] == '&' or word_list[i-a].isdigit() \
     or word_list[i-a] in roman_num:
        word_list.pop(i - a)
        a += 1
len(word_list)


maxlen = 6                                                            
step = 1 

sentences = []                                                         

next_words = []                                                        

for i in range(0, len(word_list) - maxlen, step):
    sentences.append(word_list[i: i + maxlen])
    next_words.append(word_list[i + maxlen])

print('Number of sequences:', len(sentences))

words = sorted(list(set(word_list)))                                        
print('Unique words:', len(words))
word_indices = dict((word, words.index(word)) for word in words)       

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(words)), dtype=np.bool)      
y = np.zeros((len(sentences), len(words)), dtype=np.bool)              
for i, sentence in enumerate(sentences):                               
    for t, word in enumerate(sentence):                                
        x[i, t, word_indices[word]] = 1                                
    y[i, word_indices[next_words[i]]] = 1


import nltk
from nltk.corpus import cmudict
nltk.download('cmudict')
d = cmudict.dict()


# Method to add phonems for each word to 'phonem' dict
def syls(word):
    flag = False
    r_word = word[:]
    while len(r_word) > 0 and flag == False:
        try:
            phonem[word] = d[r_word.lower()]
            flag = True        # if no exception occurs then flag is set indicating phonems added to dict
        except Exception as e:
            r_word = r_word[1:]
    if r_word == '':
        phonem[word] = []


# initializes the phonem dict
phonem = {}
# calls the 'syls' method for each word in the 'words' list
for w in words:
    syls(w)
# phonem

model = keras.models.Sequential()
model.add(layers.LSTM(256, input_shape=(maxlen, len(words))))
model.add(layers.Dense(len(words), activation='softmax'))


optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


@app.route('/gopoet/')
def gopoet(name=None):
    return render_template('gopoet.html', name=name)

@app.route('/gopoet/', methods=['POST'])
def getvalue(model = model):
    epoch = request.form['epochs']
    word = request.form['words']
    temperature = request.form['temperature']

    num = epoch   # Enter Number of Epochs from 1 to 20
    if num.isdigit() and int(num) <= 20 and int(num) > 0:
        num_epochs = int(num)
        print("Setting Epochs = ", num_epochs)
        
    else:
        num_epochs = DEFAULT_EPOCHS
        print("Incorrect Input, Setting Epochs = ", num_epochs)

    num = word   # Enter Number of Words from 200 to 1200 after each epoch
    if num.isdigit() and int(num) <= 1600 and int(num) >= 200:
        num_words = int(num)
        print("Setting # of words = ", num_words)
        
    else:
        num_words = DEFAULT_WORDS
        print("Incorrect Input, Setting Words = ", num_words)

    flt = temperature   # Enter randomizer factor from 0.2 to 1.5
    try:
            temperature = float(flt)
            if temperature >= 0.2 and temperature <= 1.5:
                print("Setting Randomizer Factor = ", temperature)
            else:
                temperature = DEFAULT_TEMP
                print("Incorrect Input, Setting Randomizer Factor = ", temperature)
    except ValueError:
            temperature = DEFAULT_TEMP
            print("Incorrect Input, Setting Randomizer Factor = ", temperature)



    def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)


    def rhyming_words(word):
        if word != '':
            for key in phonem:
                len_key = len(phonem[key])
                for i in range(len(phonem[word])):
                    for j in range(len_key):
                        if word == key or len_key == 0:
                            continue
                        elif len(phonem[word]) == 1:
                            if phonem[word][i][-1] == phonem[key][j][-1]:
                                rhymers.append(key)
                        elif len(phonem[key]) == 2:
                            if phonem[word][i][-2:] == phonem[key][j][-2:]:
                                rhymers.append(key)
                        else:
                            if phonem[word][i][-3:] == phonem[key][j][-3:]:
                                rhymers.append(key)
    #     print(len(rhymers))
        
                                               
    def syllable_counter(word):
        count = 0
        if word == '' or word == '\n':
            count = 0
        elif ipa.syllable_count(word) == 0:
            count = len(word) % 3
            if count == 0:
                count = 1
        else:
            count += ipa.syllable_count(word)
        return count


    import random
    import sys
    import time

    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    FILE_NAME = ("poem_output_" + timestamp + ".txt")
    file = open(FILE_NAME, 'a')

    RHYME_CHECK = 5

    model_loss = {}
    model_loss['loss'] = []

    line_count = 0
    quatrain_count = 0
    quatrain = False
    new_line = False
    app_word = ''
    end_of_line = False

    generated_text = ''
    count = 0

    for epoch in range(1, num_epochs + 1):
        generated_text = ''
        generated_list = []
        print('\nepoch', epoch)
        hist = model.fit(x, y, batch_size=128, epochs=1)
        model_loss['loss'].append(hist.history['loss'][0])
        start_index = random.randint(0, len(word_list) - maxlen - 1)         
        generated_list = word_list[start_index: start_index + maxlen]

        for word in generated_list:
            generated_text += word + ' '
        generated_text =  generated_text.strip()
        print('--- Generating with seed: "' + generated_text + '"')
        file.write('--- Generating with seed: "' + generated_text + '"\n')
        for num in ipa.syllable_count(generated_text):
            count += num
    #    for temperature in [0.5, 1.0, 1.2]:                        
        print('------ temperature:', temperature)
        file.write('------ temperature: ' + str(temperature) + '\n')
        sys.stdout.write(generated_text)
        file.write(generated_text)

        for i in range(num_words):                                        
            sampled = np.zeros((1, maxlen, len(words)))             
            for t, word in enumerate(generated_list):               
                sampled[0, t, word_indices[word]] = 1. 
                
            if count >= 10:
                end_of_line = True
                count = 0 

            if line_count == 2 and count >= RHYME_CHECK:
                preds = model.predict(sampled, verbose=0)[0] * z1
                next_index = sample(preds, temperature)                 
                next_word = words[next_index]
                if count + syllable_counter(next_word) >= 10:
    #                     sys.stdout.write(' THIRD LINE')
    #                     print(' WORD:', next_word, end='')
                    line_count += 1 
    #                     print(f' LC {line_count} QC {quatrain_count} COUNT {count}', end='')
                    if quatrain_count == 3:
                        line_count += 1
                        quatrain_count = 0
                else:
                    preds = model.predict(sampled, verbose=0)[0]
                    next_index = sample(preds, temperature)                 
                    next_word = words[next_index]
                    while (count + syllable_counter(next_word)) >= 10:
                        next_index = sample(preds, temperature)                 
                        next_word = words[next_index] 
               
            elif line_count == 3 and count >= RHYME_CHECK:
                preds = model.predict(sampled, verbose=0)[0] * z2
                next_index = sample(preds, temperature)                 
                next_word = words[next_index]
                if (count + syllable_counter(next_word) >= 10):
    #                     sys.stdout.write('     FOURTH LINE')
                    line_count += 1
                    quatrain_count += 1
                else:
                    preds = model.predict(sampled, verbose=0)[0]
                    next_index = sample(preds, temperature)                 
                    next_word = words[next_index]
                    while (count + syllable_counter(next_word)) >= 10:
                        next_index = sample(preds, temperature)                 
                        next_word = words[next_index] 

            else:
                preds = model.predict(sampled, verbose=0)[0]
                next_index = sample(preds, temperature)                 
                next_word = words[next_index]
                
            if count < 10 and next_word == 'eol':
                next_word = ''
                app_word = generated_list[maxlen - 1]
            elif end_of_line:
                app_word = next_word = 'eol'
                end_of_line = False
                count = 0
            else:
                app_word = next_word

            generated_list.append(app_word)
            generated_list = generated_list[1:]

            if new_line and next_word == 'eol':
                next_word = ''
            elif new_line:
                next_word = next_word.capitalize()
                new_line = False
            elif next_word == 'i':
                next_word = next_word.upper()
            elif next_word == 'eol':
                next_word = '\n'
                new_line = True
            if next_word != '':    
                sys.stdout.write(' ' + next_word)
                file.write(' ' + next_word)
    #                 engine.say(next_word)
    #                 engine.runAndWait()
    #                 engine.stop()
                syl_num = syllable_counter(next_word)
                count += syl_num

            if count >= 10:
                if line_count == 0:
    #                     sys.stdout.write('     FIRST LINE')
                    rhymers = []
                    z1 = np.zeros(len(words)) + 0.0001
    #                     print("\nNext Word:", next_word)
                    r_word = next_word[:].lower()
                    r_word = r_word.strip(",.?!:;-'\"_")   
    #                     print(r_word)
                    rhyming_words(r_word)
    #                     print(rhymers)
                    for rhyme in rhymers:
                        if rhyme in word_indices:
                            z1[word_indices[rhyme]] = 10000.
                    line_count += 1
                    if quatrain_count == 3:
                        line_count += 1

                elif line_count == 1:
    #                     sys.stdout.write('  SECOND LINE')
                    rhymers = []
                    z2 = np.zeros(len(words)) + 0.0001
    #                     print("\nNext Word:", next_word)
                    r_word = next_word[:].lower()
                    r_word = r_word.strip(",.?!:;-'\"_")   
    #                     print(r_word)
                    rhyming_words(r_word)
    #                     print(rhymers)
                    for rhyme in rhymers:
                        if rhyme in word_indices:
                            z2[word_indices[rhyme]] = 10000.
                    line_count += 1
    #                     print('     LINE COUNT:', line_count, end = '')

                elif line_count == 4:
                    print()
                    file.write('\n')
                    line_count = 0
    file.close()

    return render_template('poemout.html', epoch=num_epochs, word=num_words, t=temperature)

if __name__ == '__main__':
    app.run(debug=True)
