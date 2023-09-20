import tkinter
from tkinter import filedialog
import cv2
import functools
import operator
import os
import time

import joblib
import numpy as np
from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
import extract_features

import config

filename=''
video_name=''
caption=''

class VideoDescriptionRealTime(object):
    """
        Initialize the parameters for the model
        """
    def __init__(self, config):
        self.latent_dim = config.latent_dim
        self.num_encoder_tokens = config.num_encoder_tokens
        self.num_decoder_tokens = config.num_decoder_tokens
        self.time_steps_encoder = config.time_steps_encoder
        self.max_probability = config.max_probability

        # models
        self.encoder_model = None
        self.decoder_model = None
        self.inf_encoder_model = None
        self.inf_decoder_model = None
        self.save_model_path = config.save_model_path
        self.test_path = config.test_path
        self.search_type = config.search_type
        self.tokenizer = None
        self.num = 0

    def load_inference_models(self):
        # load tokenizer

        with open(os.path.join(self.save_model_path, 'tokenizer' + str(self.num_decoder_tokens)), 'rb') as file:
            self.tokenizer = joblib.load(file)

        # inference encoder model
        self.inf_encoder_model = load_model(os.path.join(self.save_model_path, 'encoder_model.h5'))

        # inference decoder model
        decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.inf_decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        self.inf_decoder_model.load_weights(os.path.join(self.save_model_path, 'decoder_model_weights.h5'))

    def greedy_search(self, f):
        """

        :param f: the loaded numpy array after creating videos to frames and extracting features
        :return: the final sentence which has been predicted greedily
        """
        inv_map = self.index_to_word()
        states_value = self.inf_encoder_model.predict(f.reshape(-1, 80, 4096))
        target_seq = np.zeros((1, 1, 1500))
        final_sentence = ''
        target_seq[0, 0, self.tokenizer.word_index['bos']] = 1
        for i in range(15):
            output_tokens, h, c = self.inf_decoder_model.predict([target_seq] + states_value)
            states_value = [h, c]
            output_tokens = output_tokens.reshape(self.num_decoder_tokens)
            y_hat = np.argmax(output_tokens)
            if y_hat == 0:
                continue
            if inv_map[y_hat] is None:
                break
            if inv_map[y_hat] == 'eos':
                break
            else:
                final_sentence = final_sentence + inv_map[y_hat] + ' '
                target_seq = np.zeros((1, 1, 1500))
                target_seq[0, 0, y_hat] = 1
        return final_sentence

    def decode_sequence2bs(self, input_seq):
        states_value = self.inf_encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        target_seq[0, 0, self.tokenizer.word_index['bos']] = 1
        self.beam_search(target_seq, states_value, [], [], 0)
        return decode_seq

    def beam_search(self, target_seq, states_value, prob, path, lens):
        """

        :param target_seq: the array that is fed into the model to predict the next word
        :param states_value: previous state that is fed into the lstm cell
        :param prob: probability of predicting a word
        :param path: list of words from each sentence
        :param lens: number of words
        :return: final sentence
        """
        global decode_seq
        node = 2
        output_tokens, h, c = self.inf_decoder_model.predict(
            [target_seq] + states_value)
        output_tokens = output_tokens.reshape(self.num_decoder_tokens)
        sampled_token_index = output_tokens.argsort()[-node:][::-1]
        states_value = [h, c]
        for i in range(node):
            if sampled_token_index[i] == 0:
                sampled_char = ''
            else:
                sampled_char = list(self.tokenizer.word_index.keys())[
                    list(self.tokenizer.word_index.values()).index(sampled_token_index[i])]
            MAX_LEN = 12
            if sampled_char != 'eos' and lens <= MAX_LEN:
                p = output_tokens[sampled_token_index[i]]
                if sampled_char == '':
                    p = 1
                prob_new = list(prob)
                prob_new.append(p)
                path_new = list(path)
                path_new.append(sampled_char)
                target_seq = np.zeros((1, 1, self.num_decoder_tokens))
                target_seq[0, 0, sampled_token_index[i]] = 1.
                self.beam_search(target_seq, states_value, prob_new, path_new, lens + 1)
            else:
                p = output_tokens[sampled_token_index[i]]
                prob_new = list(prob)
                prob_new.append(p)
                p = functools.reduce(operator.mul, prob_new, 1)
                if p > self.max_probability:
                    decode_seq = path
                    self.max_probability = p

    def decoded_sentence_tuning(self, decoded_sentence):
        # tuning sentence
        decode_str = []
        filter_string = ['bos', 'eos']
        uni_gram = {}
        last_string = ""
        for idx2, c in enumerate(decoded_sentence):
            if c in uni_gram:
                uni_gram[c] += 1
            else:
                uni_gram[c] = 1
            if last_string == c and idx2 > 0:
                continue
            if c in filter_string:
                continue
            if len(c) > 0:
                decode_str.append(c)
            if idx2 > 0:
                last_string = c
        return decode_str

    def index_to_word(self):
        # inverts word tokenizer
        index_to_word = {value: key for key, value in self.tokenizer.word_index.items()}
        return index_to_word

    def get_test_data(self,filename,video_name):
        # loads the features array
        #file_list = os.listdir(os.path.join(self.test_path, 'video'))
        # with open(os.path.join(self.test_path, 'testing.txt')) as testing_file:
            # lines = testing_file.readlines()
        # file_name = lines[self.num].strip()
        #file_name = file_list[self.num]
        path = os.path.join(self.test_path, 'feat', video_name + '.npy')
        if os.path.exists(path):
            f = np.load(path)
        else:
            model = extract_features.model_cnn_load()
            f = extract_features.extract_features(video_name,filename, model)
        return f, filename

    def test(self,filename,video_name):
        X_test, filename = self.get_test_data(filename,video_name)
        # generate inference test outputs
        if self.search_type == 'greedy':
            sentence_predicted = self.greedy_search(X_test.reshape((-1, 80, 4096)))
        else:
            sentence_predicted = ''
            decoded_sentence = self.decode_sequence2bs(X_test.reshape((-1, 80, 4096)))
            decode_str = self.decoded_sentence_tuning(decoded_sentence)
            for d in decode_str:
                sentence_predicted = sentence_predicted + d + ' '
        # re-init max prob
        self.max_probability = -1
        return sentence_predicted, filename

    def main(self, filename, caption):
        """

        :param filename: the video to load
        :param caption: final caption
        :return:
        """
        # 1. Initialize reading video object
        cap1 = cv2.VideoCapture(os.path.join(self.test_path, 'video', filename))
        cap2 = cv2.VideoCapture(os.path.join(self.test_path, 'video', filename))
        caption = '[' + ' '.join(caption.split()[1:]) + ']'
        # 2. Cycle through pictures
        while cap1.isOpened():
            ret, frame = cap2.read()
            ret2, frame2 = cap1.read()
            if ret:
                imS = cv2.resize(frame, (480, 300))
                cv2.putText(imS, caption, (100, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                            2, cv2.LINE_4)
                cv2.imshow("VIDEO CAPTIONING", imS)
            if ret2:
                imS = cv2.resize(frame, (480, 300))
                cv2.imshow("ORIGINAL", imS)
            else:
                break

            # Quit playing
            key = cv2.waitKey(25)
            if key == 27:  # Button esc
                break

        # 3. Free resources
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()



# Function Definations

def openFile(event=None):
    global filename
    filename = filedialog.askopenfilename(filetypes=[('Video Files', ['*.mp4', '*.avi'])])
    global video_name
    video_name = filename.split('/')[-1]
    print('Selected:', video_name)


    video_to_text = VideoDescriptionRealTime(config)
    video_to_text.load_inference_models()
    print('.........................\nGenerating Caption:\n')
    start = time.time()
    video_caption, file = video_to_text.test(filename,video_name)
    end = time.time()
    sentence = ''
    print(sentence)
    for text in video_caption.split():
        sentence = sentence + ' ' + text
        print('\n.........................\n')
        print(sentence)
        print('\n.........................\n')
        print('It took {:.2f} seconds to generate caption'.format(end-start))
    
    global caption
    caption = sentence
    print("This is caption...................",caption)
    
    global cap
    cap = cv2.VideoCapture(filename)
    update()

#for playing the videos

def photo_image(img):
    h, w = img.shape[:2]
    data = f'P6 {w} {h} 255 '.encode() + img[..., ::-1].tobytes()
    return tkinter.PhotoImage(width=w, height=h, data=data, format='PPM')

def update():
    ret, frame = cap.read()
    result_text.set(caption)
    if ret:
        photo = photo_image(frame)
        canvas.create_image(360,240, image=photo, anchor='center')
        canvas.image = photo
    else:
        cap.release()
    vplayer_frame.after(42, update)



if __name__ == "__main__":

    # Start of Tkinter window
    main_window = tkinter.Tk()
    main_window.geometry('980x720')
    main_window.title("Video Description Generation System")
    main_window.resizable(width=False, height=False)
    #set window color
    main_window['background']='#B4E4FF'

    # upper_frame
    upper_frame = tkinter.Frame(main_window)
    upper_frame.grid(row=0, column=0)
    #set window color
    upper_frame['background']='#B4E4FF'
    # Label
    header = "Upload a short video"
    tkinter.Label(upper_frame, text=header, bg="#B4E4FF",font=("Calisto MT", 24, "italic"), justify="center" ).grid(row=0, column=1, padx=350, pady=20, sticky='news')

    #video player
    vplayer_frame = tkinter.Frame(main_window)
    vplayer_frame.grid(row=1, column= 0)
    vplayer_frame['background']='#B4E4FF'

    canvas = tkinter.Canvas(vplayer_frame, height=480, width=720, bd=12,bg='black', relief='ridge')
    canvas.grid(row=0, column=0, rowspan=3, columnspan=3)

    button = tkinter.Button(vplayer_frame, text='Upload Video', command=openFile, cursor='heart', width=20, height=1, bg='#5e17eb')
    button.grid(row =3, column=1, pady=8)

    #Description Part
    lower_frame = tkinter.Frame(main_window)
    lower_frame.grid(row=2, column= 0)
    #set window color
    lower_frame['background']='#B4E4FF'
    #Description Displaying Part
    result_text = tkinter.StringVar()
    result = tkinter.Label(lower_frame, textvariable=result_text, font=("Calisto MT", 14, "bold"))
    result.grid(row=2, column=0, columnspan=3)
    #"Description" label 
    tkinter.Label(lower_frame,text='Description:', font=("Calisto MT", 18, "italic"), justify=tkinter.LEFT, bg='#B4E4FF' ).grid(row=0, column=1, pady=7)
    

    main_window.mainloop()
    


