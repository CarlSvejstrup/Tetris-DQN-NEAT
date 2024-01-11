import pygame
import neat
import os
import numpy as np
import pickle
import sys
import random
Draw = False
max_score = 100_000_000

# skift directory for at kunne importere Tetris fra tetris_engine
current_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.join(current_dir, "..")
sys.path.append(project_dir)
from tetris_engine import Tetris

class Tetris_game:
    def __init__(self) -> None:
        self.game = Tetris(10, 20)
        self.draw = Draw
                    

    def train_ai(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        genome.fitness = 0
        
        while True:
            
            reward, done = self.make_move(net)
            genome.fitness += reward
            
            if self.draw:
                self.game.render1(genome.fitness, framerate=60)
            
            if genome.fitness > max_score:
                genome.fitness = max_score + 100_000 * self.game.tetris_amount
                done = True
                
            if done:
                break
        
    def make_move(self, net):
        best_action = None
        best_value = None
        all_states = self.game.merge_next_states()
        
        for action, state in zip(all_states.keys(), all_states.values()):
            output = net.activate(
                (state))
            if best_value is None or output > best_value:
                best_value = output
                best_action = action
        return self.game.step(best_action)

def eval_genomes(genomes, config):
    if Draw:
        pygame.init()
        width, height = 300, 700
        pygame.display.set_mode((width, height))

    for (genome_id, genome) in genomes:
        
        tetris = Tetris_game()
        tetris.train_ai(genome=genome, config=config)
    if Draw:
        pygame.quit()

def run_neat(config, seed=random.randint(1,1_000_000)):
    random.seed(seed)
    #p = neat.Checkpointer.restore_checkpoint('src_neat/checkpoint_neat/neat-checkpoint-43')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10, filename_prefix='src_neat/checkpoint_neat/neat-checkpoint-'))

    winner = p.run(eval_genomes, 50)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


def test_ai(config, out, make_video=False):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    tetris = Tetris_game()
    score = 0
    while True:
            
        reward, done = tetris.make_move(winner_net)
        score += reward

        if make_video:
            frame = tetris.game.render_save_video(score)
            out.write(frame)

        
        elif Draw:
            tetris.game.render1(score, framerate=60)
                    
        if done:
            return score
        