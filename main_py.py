# -*- coding: utf-8 -*-
"""main.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MKNkcJNEOezxhvcY3kNkGLsj9-w8uZLu
"""

class Car_damage_detection:
  """
  Preprocess Image for InceptionV3 Model.

  This function preprocesses an image for compatibility with the InceptionV3 model.

  Parameters:
  - image_path (str): Path to the input image file.

  Returns:
  - np.ndarray: Preprocessed image as a NumPy array.

  """
  def preprocess(image_path):
    img = load_img(image_path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

  """
  Encode Image into Feature Vector Using Pre-trained InceptionV3 Model.

  This function encodes an image into a feature vector using a pre-trained InceptionV3 model.

  Parameters:
  - image (str): Path to the input image file.

  Returns:
  - np.ndarray: Feature vector representing the image.

  """
  def encode(image):
    image = cdd_object.preprocess(image)
    fea_vec = model_new.predict(image)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    print(fea_vec)
    return fea_vec

  """
  Generate Data Batches for Neural Network Training.

  This function generates data batches for training a neural network using image features and corresponding textual descriptions.

  Parameters:
  - descriptions (dict): Dictionary containing image descriptions.
  - photos (dict): Dictionary containing image features.
  - word_to_ix (dict): Dictionary mapping words to indices.
  - max_length (int): Maximum length of input sequences.
  - num_photos_per_batch (int): Number of photos per batch.

  Yields:
  - tuple: A tuple containing two lists (X1, X2) and an array (y).
      - X1 (list): List of image features.
      - X2 (list): List of input sequences.
      - y (np.ndarray): Array of output sequences.

  """

  def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            # retrieve the photo feature
            photo = photos[key+'.jpg']
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)

            if n==num_photos_per_batch:
                yield ([array(X1), array(X2)], array(y))
                X1, X2, y = list(), list(), list()
                n=0

  """
  Generate Image Caption using Greedy Search.

  This function generates an image caption using a greedy search strategy.

  Parameters:
  - photo (np.ndarray): Image feature representing the input image.

  Returns:
  - str: Generated image caption.

  """
  def greedySearch(photo):
      in_text = 'startseq'
      for i in range(max_length):
          sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
          sequence = pad_sequences([sequence], maxlen=max_length)
          yhat = model.predict([photo,sequence], verbose=0)
          yhat = np.argmax(yhat)
          word = ixtoword[yhat]
          in_text += ' ' + word
          if word == 'endseq':
              break

      final = in_text.split()
      final = final[1:-1]
      final = ' '.join(final)
      return final

  """
  Generate Image Caption using Beam Search.

  This function generates an image caption using a beam search strategy.

  Parameters:
  - image (np.ndarray): Image feature representing the input image.
  - beam_index (int): Number of top predictions to consider at each step (default is 3).

  Returns:
  - str: Generated image caption.

  Note:
  - The beam search strategy considers multiple possible sequences and selects the most likely sequence based on probabilities at each step.
  """
  def beam_search_predictions(image, beam_index = 3):
    start = [wordtoix["startseq"]]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=max_length, padding='post')
            preds = model.predict([image,par_caps], verbose=0)
            word_preds = np.argsort(preds[0])[-beam_index:]
            # Getting the top <beam_index>(n) predictions and creating a
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])

        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
    intermediate_caption = [ixtoword[i] for i in start_word]
    final_caption = []

    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption

  """
  Compute Accuracy for Greedy Search and Beam Search.

  This code computes accuracy for both Greedy Search and Beam Search predictions.

  Parameters:
  - y_true (dict): True captions for images.
  - y_pred (dict): Predicted captions for images using both Greedy Search and Beam Search.

  Returns:
  - tuple: Accuracy for Greedy Search and Beam Search.

  """
  def compute_accuracy(y_true, y_pred):
    correct_predictions_gs = 0
    correct_predictions_bs = 0
    for each_key in y_true.keys():
        print("pred0 -", y_pred[each_key][0])
        print("pred1 -", y_pred[each_key][1])
        print("true -", y_true[each_key])
        if y_true[each_key][0] == y_pred[each_key][0]:
            correct_predictions_gs += 1

        if y_true[each_key] == y_pred[each_key][1]:
            correct_predictions_bs += 1
    accuracy_gs, accuracy_bs = correct_predictions_gs/len(y_true), correct_predictions_bs/len(y_true)
    print(correct_predictions_bs)
    return accuracy_gs, accuracy_bs

    def data_paths:
      token_path = "/content/drive/MyDrive/yolov7/data/damage_data.token.txt"
      train_images_path = '/content/drive/MyDrive/yolov7/data/train.txt'
      test_images_path = '/content/drive/MyDrive/yolov7/data/test.txt'
      images_path = '/content/drive/MyDrive/yolov7/Annotated_Cars_Dataset/images/'
      glove_path = '/content/drive/MyDrive/yolov7/data/glove6b/'

    def training_customdata_yolov7:
      """
      Process Descriptions from a String and Store in a Dictionary.

      This code processes a multi-line string containing image descriptions and populates a dictionary where image IDs are keys,
      and corresponding descriptions are stored as lists.

      Parameters:
      - doc (str): Input string containing image descriptions.

      Returns:
      - dict: A dictionary where keys are image IDs, and values are lists of descriptions associated with each image.

      """
      descriptions = dict()
      for line in doc.split('\n'):
        tokens = line.split()
        if len(line) > 2:
          image_id = tokens[0].split('.')[0]
          # print(image_id)
          image_desc = ' '.join(tokens[1:])
          if image_id not in descriptions:
              descriptions[image_id] = list()
          descriptions[image_id].append(image_desc)

      """
      Preprocess Descriptions in a Dictionary by Removing Punctuation and Lowercasing.

      This code iterates over a dictionary containing image descriptions, preprocesses each description,
      and removes punctuation while converting all words to lowercase.

      Parameters:
      - descriptions (dict): A dictionary where keys are image IDs, and values are lists of descriptions associated with each image.

      Returns:
      - None: The descriptions in the input dictionary are modified in-place.

      """
      table = str.maketrans('', '', string.punctuation)
      for key, desc_list in descriptions.items():
          for i in range(len(desc_list)):
              desc = desc_list[i]
              desc = desc.split()
              desc = [word.lower() for word in desc]
              desc = [w.translate(table) for w in desc]
              desc_list[i] =  ' '.join(desc)
      """
      Build a vocabulary from a dictionary of image descriptions.

      Parameters:
      - descriptions (dict): A dictionary where keys are image identifiers and values
      are lists of associated descriptions.

      Returns:
      - set: A set containing unique words in the provided descriptions.


      This function processes the descriptions and updates a vocabulary set with unique words.
      """
      vocabulary = set()
      for key in descriptions.keys():
              [vocabulary.update(d.split()) for d in descriptions[key]]
      print('Original Vocabulary Size: %d' % len(vocabulary))

      """
      Create a List of Lines Combining Image IDs and Descriptions, then Join into a Single String.

      This code processes a dictionary containing image descriptions, combining image IDs with corresponding descriptions.
      It creates a list of lines, where each line is formed by concatenating the image ID and description, and then joins
      these lines into a single string.

      Parameters:
      - descriptions (dict): A dictionary where keys are image IDs, and values are lists of descriptions associated with each image.

      Returns:
      - str: A string formed by joining lines, where each line contains an image ID and its corresponding description.

      """
      lines = list()
      for key, desc_list in descriptions.items():
          for desc in desc_list:
              lines.append(key + ' ' + desc)
      new_descriptions = '\n'.join(lines)

      """
      Read Image Identifiers from a File and Create a Training Dataset.

      This code reads image identifiers from a file, extracts identifiers from each line, and forms a training dataset.

      Parameters:
      - train_images_path (str): Path to the file containing image identifiers.

      Returns:
      - set: A set containing unique image identifiers forming the training dataset.

      """
      doc = open(train_images_path,'r').read()
      dataset = list()
      for line in doc.split('\n'):
          if len(line) > 1:
            identifier = line.split('.')[0]
            dataset.append(identifier)

      train = set(dataset)

      img = glob.glob(images_path + '*.jpg')
      print("img - ", img)
      train_images = set(open(train_images_path, 'r').read().strip().split('\n'))
      print("train_images - ", train_images)
      train_img = []
      for i in img:
          if i[len(images_path):] in train_images:
              train_img.append(i)

      test_images = set(open(test_images_path, 'r').read().strip().split('\n'))
      test_img = []
      for i in img:
          if i[len(images_path):] in test_images:
              test_img.append(i)

      """
      Create Training Descriptions Dictionary from a String.

      This code processes a string containing image identifiers and descriptions, splits the string into lines,
      and forms a dictionary where keys are image IDs present in the training set and values are lists of preprocessed descriptions.

      Parameters:
      - new_descriptions (str): String containing image identifiers and descriptions.
      - train (set): Set of image identifiers forming the training dataset.

      Returns:
      - dict: A dictionary where keys are image IDs from the training set, and values are lists of preprocessed descriptions.

      """
      train_descriptions = dict()
      for line in new_descriptions.split('\n'):
          tokens = line.split()
          image_id, image_desc = tokens[0], tokens[1:]
          if image_id in train:
              if image_id not in train_descriptions:
                  train_descriptions[image_id] = list()
              desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
              train_descriptions[image_id].append(desc)

      all_train_captions = []
      for key, val in train_descriptions.items():
          for cap in val:
              all_train_captions.append(cap)

      """
      Create a Vocabulary from Training Captions Based on Word Count Threshold.

      This code processes a list of training captions, counts the occurrences of each word, and forms a vocabulary based on a specified word count threshold.

      Parameters:
      - all_train_captions (list): List of training captions.
      - word_count_threshold (int): Minimum count of occurrences for a word to be included in the vocabulary.

      Returns:
      - list: A list containing words that meet the specified word count threshold.

      """
      word_count_threshold = 10
      word_counts = {}
      nsents = 0
      for sent in all_train_captions:
          nsents += 1
          for w in sent.split(' '):
              word_counts[w] = word_counts.get(w, 0) + 1
      vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

      print('Vocabulary = %d' % (len(vocab)))

      """
      Create Mapping between Words and Indices for Vocabulary.

      This code processes a vocabulary list, assigns unique indices to each word, and creates mappings between words and their corresponding indices.

      Parameters:
      - vocab (list): List of words forming the vocabulary.

      Returns:
      - dict: A dictionary mapping words to indices.
      - dict: A dictionary mapping indices to words.
      - int: Size of the vocabulary.

      """
      ixtoword = {}
      wordtoix = {}
      ix = 1
      for w in vocab:
          wordtoix[w] = ix
          ixtoword[ix] = w
          ix += 1

      vocab_size = len(ixtoword) + 1

      """
          Find the maximum length of descriptions in a training dataset.

          Parameters:
          - train_descriptions (dict): A dictionary where keys are image identifiers and values
            are lists of associated training descriptions.

          Returns:
          - int: The maximum length of descriptions in the training dataset.

      """
      all_desc = list()
      for key in train_descriptions.keys():
          [all_desc.append(d) for d in train_descriptions[key]]
      lines = all_desc
      max_length = max(len(d.split()) for d in lines)

      print('Description Length: %d' % max_length)

      """
      Load Pre-trained Word Embeddings from GloVe File.

      This code reads pre-trained word embeddings from a GloVe file and stores them in a dictionary.

      Parameters:
      - glove_path (str): Path to the directory containing the GloVe file.
      - embedding_dim (int): Dimensionality of the word embeddings.

      Returns:
      - dict: A dictionary where keys are words and values are arrays representing word embeddings.

      """
      embeddings_index = {}
      f = open(os.path.join(glove_path, 'glove.6B.200d.txt'), encoding="utf-8")
      for line in f:
          values = line.split()
          word = values[0]
          coefs = np.asarray(values[1:], dtype='float32')
          embeddings_index[word] = coefs

      """
      Create Embedding Matrix for Words in Vocabulary.

      This code generates an embedding matrix for words in the vocabulary by assigning pre-trained word embeddings to corresponding positions.

      Parameters:
      - embeddings_index (dict): Dictionary with pre-trained word embeddings.
      - word_to_ix (dict): Dictionary mapping words to indices.
      - vocab_size (int): Size of the vocabulary.
      - embedding_dim (int): Dimensionality of the word embeddings.

      Returns:
      - np.ndarray: A 2D NumPy array representing the embedding matrix.

      """
      embedding_dim = 200
      embedding_matrix = np.zeros((vocab_size, embedding_dim))
      for word, i in wordtoix.items():
          embedding_vector = embeddings_index.get(word)
          if embedding_vector is not None:
              embedding_matrix[i] = embedding_vector
      """
      Instantiate InceptionV3 Model with Pre-trained Weights.

      This code instantiates the InceptionV3 model with pre-trained weights on the ImageNet dataset.

      Parameters:
      - weights (str): Specifies the weight checkpoint to load. If 'imagenet', it loads pre-trained ImageNet weights.

      Returns:
      - InceptionV3: An instance of the InceptionV3 model with specified weights.

      """
      model = InceptionV3(weights='imagenet')

      """
      Create a New Model by Extracting Features from Pre-trained Model.

      This code creates a new model by taking the input from the original pre-trained model and extracting features from a specific layer.

      Parameters:
      - model (keras.Model): Original pre-trained model.
      - output_layer_index (int): Index of the layer from which features are to be extracted.

      Returns:
      - keras.Model: A new model with the same input as the original model but output from a specific layer.

      """
      model_new = Model(model.input, model.layers[-2].output)

      """
      Encode Multiple Images into Feature Vectors Using Pre-trained InceptionV3 Model.

      This code encodes a set of images into feature vectors using a pre-trained InceptionV3 model.

      Parameters:
      - image_paths (list): List of paths to input image files.

      Returns:
      - dict: A dictionary where keys are image names, and values are feature vectors representing the images.

      """
      encoding_train = {}
      for img in train_img:
          print("-", img)
          encoding_train[img[len(images_path):]] = cdd_object.encode(img)
      train_features = encoding_train

      encoding_test = {}
      for img in test_img:
          encoding_test[img[len(images_path):]] = cdd_object.encode(img)

      """
      Create a Neural Network Model for Image Captioning.

      This code defines a neural network model for image captioning using the Keras library.

      Parameters:
      - vocab_size (int): Size of the vocabulary.
      - embedding_dim (int): Dimensionality of the word embeddings.
      - max_length (int): Maximum length of input sequences.

      Returns:
      - keras.Model: A neural network model for image captioning.

      """
      inputs1 = Input(shape=(2048,))
      fe1 = Dropout(0.5)(inputs1)
      fe2 = Dense(256, activation='relu')(fe1)

      inputs2 = Input(shape=(max_length,))
      se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
      se2 = Dropout(0.5)(se1)
      se3 = LSTM(256)(se2)

      decoder1 = add([fe2, se3])
      decoder2 = Dense(256, activation='relu')(decoder1)
      outputs = Dense(vocab_size, activation='softmax')(decoder2)

      model = Model(inputs=[inputs1, inputs2], outputs=outputs)
      model.summary()

      """
      Set Pre-trained Word Embeddings in the Embedding Layer and Freeze its Training.

      This code sets pre-trained word embeddings in the Embedding layer of a neural network model and freezes its training.

      Parameters:
      - model (keras.Model): Neural network model.
      - embedding_matrix (np.ndarray): Pre-trained word embedding matrix.

      Returns:
      - None: The model's Embedding layer weights are set, and its training is frozen.

      """
      model.layers[2].set_weights([embedding_matrix])
      model.layers[2].trainable = False

      """
      Compile a Neural Network Model with Categorical Crossentropy Loss and Adam Optimizer.

      This code compiles a neural network model using the categorical crossentropy loss and the Adam optimizer.

      Parameters:
      - model (keras.Model): Neural network model.
      - loss (str): Loss function to optimize. Here, set to 'categorical_crossentropy'.
      - optimizer (str): Optimization algorithm. Here, set to 'adam'.

      Returns:
      - None: The model is compiled with the specified loss and optimizer.

      """
      model.compile(loss='categorical_crossentropy', optimizer='adam')

      """
      Train a Neural Network Model using Data Generator.

      This code trains a neural network model using a data generator to generate batches of training data.

      Parameters:
      - model (keras.Model): Neural network model to be trained.
      - train_descriptions (dict): Dictionary containing training image descriptions.
      - train_features (dict): Dictionary containing training image features.
      - word_to_ix (dict): Dictionary mapping words to indices.
      - max_length (int): Maximum length of input sequences.
      - batch_size (int): Number of samples in each batch.
      - epochs (int): Number of training epochs.

      Returns:
      - History: Training history containing loss and metric values.

      """
      epochs = 5
      batch_size = 3
      steps = len(train_descriptions)//batch_size

      generator = cdd_object.data_generator(train_descriptions, train_features, wordtoix, max_length, batch_size)
      model.fit(generator, epochs=epochs, steps_per_epoch=steps, verbose=1)

cdd_object = Car_damage_detection()
cdd_object.data_paths()
cdd_object.training_customdata_yolov7()

"""
Generate and Display Image Captions using Different Strategies.

This code generates image captions using both Greedy Search and Beam Search strategies with varying beam widths.

Parameters:
- pic (str): File name of the image.
- image (np.ndarray): Image feature representing the input image.

Returns:
- None: Displays the image and prints captions using different strategies.

"""
pic = '29ced254-125412999.jpg'
image = encoding_test[pic].reshape((1,2048))
x=plt.imread(images_path+pic)
plt.imshow(x)
plt.show()

print("Greedy Search:",cdd_object.greedySearch(image))
print("Beam Search, K = 3:",cdd_object.beam_search_predictions(image, beam_index = 3))
print("Beam Search, K = 5:",cdd_object.beam_search_predictions(image, beam_index = 5))
print("Beam Search, K = 7:",cdd_object.beam_search_predictions(image, beam_index = 7))
print("Beam Search, K = 10:",cdd_object.beam_search_predictions(image, beam_index = 10))


pic = list(encoding_test.keys())[16]
image = encoding_test[pic].reshape((1,2048))
x=plt.imread(images_path+pic)
plt.imshow(x)
plt.show()

print("Greedy:",cdd_object.greedySearch(image))
print("Beam Search, K = 3:",cdd_object.beam_search_predictions(image, beam_index = 3))
print("Beam Search, K = 5:",cdd_object.beam_search_predictions(image, beam_index = 5))
print("Beam Search, K = 7:",cdd_object.beam_search_predictions(image, beam_index = 7))

train_images_list = [i.split('/')[-1] for i in train_img]

"""
Generate Predicted Captions for Training Images.

This code generates predicted captions for training images using both Greedy Search and Beam Search.

Parameters:
- train_images_list (list): List of file names for training images.
- encoding_train (dict): Dictionary containing image features for training images.

Returns:
- dict: Dictionary containing predicted captions for training images.

"""
train_image_captions = dict()
for single_img in train_images_list:
  image = encoding_train[single_img].reshape((1,2048))
  train_image_captions[single_img] = [cdd_object.greedySearch(image), cdd_object.beam_search_predictions(image, beam_index = 3)]

"""
Save Image Captions to a JSON File.

This code saves image captions to a JSON file.

Parameters:
- file_path (str): File path where the JSON file will be saved.
- captions_data (dict): Dictionary containing image captions.

Returns:
- None: Saves the image captions to the specified JSON file.

"""
with open('caption_prediction_data_greedy_beam.json', 'w') as fp:
    json.dump(train_image_captions, fp)

"""
Save a Trained Model in HDF5 Format using Pickle.

This code saves a trained model in HDF5 format using the 'pickle' module.

Parameters:
- filename (str): File name or path where the model will be saved.
- model: Trained model to be saved.

Returns:
- None: Saves the trained model to the specified file in HDF5 format using 'pickle'.

"""
filename = 'caption_generation_model.h5'
pickle.dump(model, open(filename, 'wb'))

"""
Create a Dictionary of Training Image Descriptions.

This code processes and organizes image descriptions into a dictionary for training purposes.

Parameters:
- new_descriptions (str): String containing image descriptions.
- train (set): Set of identifiers for training images.

Returns:
- dict: Dictionary containing image descriptions for training images.

"""
train_descriptions = dict()
for line in new_descriptions.split('\n'):
    tokens = line.split()
    image_id, image_desc = tokens[0], tokens[1:]
    if image_id in train:
        if image_id not in train_descriptions:
            train_descriptions[image_id+'.jpg'] = list()
        train_descriptions[image_id+'.jpg'].append(' '.join(image_desc))

"""
Compute Accuracy for Train Image Captions.

This code computes accuracy for train image captions based on true and predicted captions.

Parameters:
- train_descriptions (dict): True captions for training images.
- train_image_captions (dict): Predicted captions for training images.

Returns:
- tuple: Accuracy for Greedy Search and Beam Search.

"""
cdd_object.compute_accuracy(train_descriptions, train_image_captions)