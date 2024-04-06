import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import GeneticAlgorithmFunctions as ga

def main():
    input_path = input("Input path : ")
    population_size = int(input("Population size : ")) #1000
    num_lines = int(input("Number of lines : ")) # 200
    num_gen = int(input("Number of generations : ")) # 500
    mutation_rate = float(input("Mutation rate : ")) # 0.2
    narutal_selection_rate = float(input("Natural selection rate : ")) # 0.25
    fresh_population = int(input("Fresh population size : ")) #1000
    survive_count = int(population_size * narutal_selection_rate)
        
    img_array = ga.read_image(input_path)
    circle_points = ga.get_points(img_array)
    
    population = []
    
    for i in range(population_size):
        individual = random.sample(circle_points, num_lines)
        
        individual_score = ga.fitness(img_array, individual)
        population.append((individual, individual_score))
    
    population = ga.natural_selection(population, survive_count)
    
    all_time_best_individual = population[0]
    score_history = []
    best_fitness_score = -float('inf')
    no_improvement_streak = 0
    mutation_multiplexer = 2
    mutation_threshold = 10
    
    for i in range(num_gen):
        new_generation = []
        
        for individual in population:
            mutated_individual = ga.mutation(individual[0], circle_points,  mutation_rate)
            mutated_individual_score = ga.fitness(img_array, mutated_individual)
            
            new_generation.append((mutated_individual, mutated_individual_score))
            
        for j in range(0, survive_count, 2):
            parent1, parent2 = population[j][0], population[j + 1][0]
            child = ga.crossover(parent1, parent2)
            
            child_score = ga.fitness(img_array, child)
            
            new_generation.append((child, child_score))
            
        for _ in range(survive_count):
            parent1 = random.choice(population)[0]
            parent2 = random.choice(population)[0]
            
            child = ga.crossover(parent1, parent2)
            child = ga.mutation(child, circle_points, mutation_rate)
            child_score = ga.fitness(img_array, child)
            
            new_generation.append((child, child_score))
        
        for individual_tuple in population:
            individual_points = individual_tuple[0]
            
            random_num_lines = random.randint(0, num_lines - 1)
            new_part = random.sample(circle_points, random_num_lines)
            cutoff = num_lines - random_num_lines
            
            new_individual = new_part + individual_points[:cutoff]
            new_individual_score = ga.fitness(img_array, new_individual)
            new_generation.append((new_individual, new_individual_score))
            
            new_individual = individual_points[-cutoff:] + new_part
            new_individual_score = ga.fitness(img_array, new_individual)
            new_generation.append((new_individual, new_individual_score))
        
        population.extend(new_generation)
        
        for _ in range(fresh_population):
            individual = random.sample(circle_points, num_lines)
            
            score = ga.fitness(img_array, individual)
            
            population.append((individual, score))
        
        population = ga.sort_population(new_generation)
        population = population[:population_size]
        population = ga.natural_selection(population, survive_count)
        
        best_individual = population[0]
        score_history.append(best_individual[1])
        current_best_fitness_score = best_individual[1]
        
        if best_individual[1] > all_time_best_individual[1]:
            all_time_best_individual = best_individual
        
        if mutation_rate < 0.5:
            if current_best_fitness_score > best_fitness_score:
                best_fitness_score = current_best_fitness_score
                no_improvement_streak = 0
            else:
                no_improvement_streak += 1
                
            if no_improvement_streak >= mutation_threshold:
                mutation_rate = ga.adjust_mutation_rate(mutation_rate, mutation_multiplexer)

        print('-'*11, f"Generation: {i}",'-'*11, 
                f"\n Generation top individual score: {current_best_fitness_score:.3f}"
                f"\n Generation mutation rate: {mutation_rate:.3f}")
        matris = ga.draw_lines(best_individual[0], img_array.shape[0])
        plt.imshow(matris, cmap='gray')
        plt.axis('off')
        plt.title(f"Generation: {i} - Best individual score: {best_individual[1]:.2f}")
        plt.pause(0.1)
        plt.clf()

    matris = ga.draw_lines(all_time_best_individual[0], img_array.shape[0])
    image = Image.fromarray(matris.astype(np.uint8), mode='L')
    image.save("output.png")
    print("Output image saved.")

    plt.plot(score_history)
    plt.xlabel("Generation")
    plt.ylabel("Similarity Score")
    plt.title("Similarity Score Graph")
    plt.savefig("accuracy.png")
    plt.show()

if __name__ == "__main__":
    main()