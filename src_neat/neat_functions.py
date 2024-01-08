import pygame
import neat
from engine import Tetris
import os
import numpy as np
import pickle

pygame.init()
shapes_to_index = {
    "[(0, 0), (-1, 0), (1, 0), (0, -1)]" : 0,
    "[(0, 0), (-1, 0), (0, -1), (0, -2)]": 1,
    "[(0, 0), (1, 0), (0, -1), (0, -2)]": 2,
    "[(0, 0), (-1, 0), (0, -1), (1, -1)]": 3,
    "[(0, 0), (-1, -1), (0, -1), (1, 0)]": 4,
    "[(0, 0), (0, -1), (0, -2), (0, -3)]": 5,
    "[(0, 0), (0, -1), (-1, 0), (-1, -1)]": 6,
}

class Tetris_game:
    def __init__(self, draw = False) -> None:
        self.game = Tetris(10, 20)
        self.draw = draw
                    

    def train_ai(self, genome, config):
        lines_to_clear = 10
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        genome.fitness = 0
        
        while True:
            
            reward, done = self.make_move(net)
            genome.fitness += reward
            if self.draw:
                self.game.render(genome.fitness)
                        
            if done or genome.fitness == 10000:
                genome.fitness = float(genome.fitness)
                break
        
    
    def get_fitness(self, info):
        weight_lines = 50
        
        return int(sum(info["statistics"].values()) + weight_lines * info["lines_cleared"])
    
    def make_move(self, net):
        best_action = None
        best_value = None
        all_states = self.game.get_next_states()
        
        for action, state in zip(all_states.keys(), all_states.values()):
            output = net.activate(
                (state))
            if best_value is None or output > best_value:
                best_value = output
                best_action = action
        return self.game.step(best_action)
        
    
    def get_input(self, board):
        heights = []
        holes = []
        for collum in board:
            if 1 in collum:
                heights.append(20 - np.where(collum == 1)[0][0])
                
            else:
                heights.append(0)
            
            holes.append(heights[-1] - np.sum(collum))
        
        input_result = heights + holes
        return input_result
    

def eval_genomes(genomes, config):
    
    for (genome_id, genome) in genomes:
        
        tetris = Tetris_game(draw=False)
        tetris.train_ai(genome=genome, config=config)

def run_neat(config):
    #p = neat.Checkpointer.restore_checkpoint('checkpoint/neat-checkpoint-49')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(50, filename_prefix='src/checkpoint_neat/neat-checkpoint-'))

    winner = p.run(eval_genomes, 500)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


