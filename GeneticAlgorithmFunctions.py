import cv2
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def read_image(path, multiplier = 1.5):
    img = Image.open(path)
    width, height = img.size
    N = round(min(width, height) * multiplier)
    img = img.resize((N, N))
    img = img.convert('L')
    img_array = np.array(img)
    img_array[img_array < 255] = 0
    return img_array

def get_points(img_array, point_count = 360):
    N = img_array.shape[0]
    r = N / 2
    center = (r, r)
    all_points = []
    
    for theta in np.linspace(0, 2 * np.pi, point_count):
        x = int(round(r * np.cos(theta) + center[0] - 1))
        y = int(round(r * np.sin(theta) + center[1] - 1))
        all_points.append((x, y))
    return all_points
    
def draw_lines(ind_points, img_size):
    matris = np.full((img_size, img_size), 255, dtype=np.uint8)
    for i in range(len(ind_points) - 1):
        cv2.line(matris, ind_points[i], ind_points[i + 1], 0, 1)
    return matris
    
def fitness(image, ind_points):
    N = len(image)
    new_matris = draw_lines(ind_points, N)
    count = np.sum(image == new_matris)
    return count / (N ** 2)
    
def crossover(p1_points, p2_points):
    cutoff = random.randint(0, len(p1_points) - 1)
    child1 = p1_points[:cutoff] + p2_points[cutoff:]
    return child1
    
def adjust_mutation_rate(mutation_rate, increase_factor = 1.5):
    mutation_rate = min(mutation_rate * increase_factor, 0.5)
    
    return mutation_rate
    
def mutation(ind_points, all_points, threshold=0.05):
    if random.random() < threshold:
        mutate_index = random.randint(0, len(ind_points) - 1)
        ind_points[mutate_index] = random.choice(all_points)
        
    return ind_points

def sort_population(population):
    population.sort(key=lambda x: x[1], reverse=True)
    return population

def natural_selection(population, survive_count):
    return population[:survive_count]

