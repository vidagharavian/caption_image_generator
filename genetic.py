from random import random
import os
import cv2
from keras_preprocessing.image import img_to_array
import numpy as np
from tensorflow.python.keras.models import load_model
import joblib
import pandas as pd
import keras
img_size = 221
labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable'
    , 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]  # convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
                image = img_to_array(resized_arr)
                data.append(image)
            except Exception as e:
                print(e)
    return np.array(np.array(data)) / 255
val = get_data('dataset/test/images')


def latent2im(decoder, compressed):
    x_decoded = decoder.predict(compressed)
    im = x_decoded[0].reshape(img_size, img_size, 3)
    return im


def im2latent(encoder, im):
    compressed = encoder.predict(im, batch_size=1)
    return compressed


encoder = load_model('encoder.h5')
decoder = load_model('decoder.h5')

encoded_images = []


def initial_population(population_count):
    population = []
    for i in range(population_count):
        chromosome = []
        sum = 0
        for j in range(10):
            value = round(random(), 2)
            sum += value
            chromosome.append(value)
        for i, gen in enumerate(chromosome):
            chromosome[i] = (gen / sum) * 100
        population.append(chromosome)
    return population


def matching_pool(fitness_set_population: dict, selection_number):
    fitness_set_population = dict(sorted(fitness_set_population.items()))
    best_chromosome_value = list(fitness_set_population.keys())[-1]
    output = list(fitness_set_population.values())[-selection_number:]
    return output, best_chromosome_value


def probable_selector(image, label):
    model = load_model('model1.h5',compile=False)
    if len(image.shape) == 3:
        image=image.reshape(1,img_size,img_size,3)
        y = model.predict(image)
        for i,l in enumerate(labels):
            if l == label:
                y=y[0][i]
        return y
    else:
        output={}
        y=model.predict(image)
        for i,l in enumerate(labels):
            if l == label:
                for j,la in enumerate(y):
                    output[la[i]]=image[j]
        return output



def fitness_calculation(population, label):
    image_population = []
    # 10 chosen image
    output = {}
    for chromosome in population:
        image_population.append(create_image(chromosome))
    # for i, image in enumerate(image_population):
    #     key = probable_selector(image, label)
    #     output[key] = population[i]
    labeled_images=probable_selector(np.array(image_population),label)
    for i,key in enumerate(list(labeled_images.keys())):
        output[key]=population[i]
        # {darsadnazdiki:chromosome}
    return output


def cross_over(parents: list, weights: tuple = (2, 1)) -> list:
    parent_1 = parents[0]
    parent_2 = parents[1]
    children_1 = []
    children_2 = []
    for i, item in parent_1:
        children_1.append((item * weights[0] + parent_2[i] * weights[1]) / sum(weights))
        children_2.append((item * weights[1] + parent_2[i] * weights[0]) / sum(weights))
    return [children_1, children_2]


def create_new_population(caption):
    init_count = 1000
    iteration = 300
    normalized_caption=normalize_caption(caption)
    label = get_caption_label(normalized_caption)
    k_best_images = get_chosen_k_image(10, label)
    # encode
    for i, img in enumerate(k_best_images):
        img = np.array(img.resize([img_size, img_size]))
        img = img.reshape((1, np.prod(img.shape[0:])))
        encoded_images[i] = im2latent(encoder, img)
    init_pop = initial_population(init_count)
    new_population = init_pop
    best_chromosome = None
    max_fitness = 0
    for i in range(iteration):
        fited_population = fitness_calculation(new_population, label)
        matched_parent, best_curr_fitness = matching_pool(fited_population, init_count / 2)
        if best_curr_fitness > max_fitness:
            best_chromosome = matched_parent[-1]
            max_fitness = best_curr_fitness
        children = []
        for i in range(0, len(matched_parent) - 1, 2):
            children.extend(cross_over(matched_parent[i], matched_parent[i + 1]))
        new_population = matched_parent
        new_population.extend(children)
        new_population.extend(initial_population(1))
    last_image = create_image(best_chromosome)
    cv2.imwrite('output.jpg', last_image)
    return last_image


def create_image(weights):
    sum = 0
    weights_sum = 0
    for i in range(10):
        sum += weights[i] * encoded_images[i]
        weights_sum += weights[i]
    sum /= weights_sum
    # decode
    image = latent2im(decoder, sum)
    return image


def get_caption_label(caption):
    model = joblib.load('caption_model.pkl')
    label = model.predict(list(caption))
    return label[0]


def get_chosen_k_image(k, label):
    # load data
    data =get_data('dataset/test/images')
    output = probable_selector(data, label)
    output = dict(sorted(output.items()))
    output = list(output.values())[-k:]
    return output


def similar(a, b):
    if a == 'woman' and b == 'women':
        return True
    if a == 'man' and b == 'men':
        return True
    return a + 's' == b or a + 'a' == b


def normalize_caption(caption):
    headers = pd.read_excel('train.xlsx', sheet_name='Sheet1', engine='openpyxl').columns
    normalized_caption = {}
    for header in headers:
        normalized_caption[header] = caption.count(header)

    output = {}
    for header in headers:
        keys = list(output.keys())
        output[header] = []
        for key in keys:
            if similar(header, key) or similar(key, header):
                output[key].append(header)
                try:
                    del output[header]
                except:
                    pass
    for key,values in output.items():
        if len(values)!=0:
            for value in values:
                normalized_caption[key]+=normalized_caption[value]
                del normalized_caption[value]
    return np.array(list(normalized_caption.values())[:-1]).reshape(1, -1)


create_new_population(
    'Two gentleman talking in front of propeller plane. Two men are conversing next to a small airplane.Two men talking in front of a planeTwo men talking in front of a small plane.Two men talk while standing next to a small passenger plane at an airport.')
#initial population: get random 10 number