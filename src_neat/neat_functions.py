import pygame
import neat
import os
import numpy as np
import pickle
import sys
import random
import cv2 as cv
import csv


draw = False
max_score = 3_000_000
tetris_bonus = 1_200

# skift directory for at kunne importere Tetris fra tetris_engine
current_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.join(current_dir, "..")
sys.path.append(project_dir)
from tetris_engine import Tetris

tetris = None

class Tetris_game:
    def __init__(self, seed=random.randint(1,1_000_000)) -> None:
        self.game = Tetris(10, 20, seed)

    def train_ai(self, genome, config) -> None:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        genome.fitness = 0

        while True:
            reward, done = self.make_move(net, with_held=True)
            genome.fitness += reward

            if draw:
                self.game.render1(genome.fitness, framerate=60)

            if genome.fitness > max_score:
                genome.fitness = genome.fitness + tetris_bonus * self.game.tetris_clear
                done = True

            if done:
                print("done", int(genome.fitness))
                genome.fitness = int(genome.fitness)
                return genome.fitness

    def make_move(self, net, with_held=True):
        best_action = None
        best_value = None

        # unless otherwise specified, we do not use the hold funtion, this is to speed up training since we now only need to calculate half as many states
        # when testing our models we would ofcouse use the hold function
        if with_held:
            all_states = self.game.merge_next_states()
        else:
            all_states = self.game.get_next_states(
                self.game.shape, self.game.anchor, False
            )

        for action, state in zip(all_states.keys(), all_states.values()):
            output = net.activate((state))
            if best_value is None or output > best_value:
                best_value = output
                best_action = action
        return self.game.step(best_action)


def eval_genomes(genomes, config):
    if draw:
        pygame.init()
        width, height = 300, 700
        pygame.display.set_mode((width, height))
    results = np.zeros(50)
    for i, (genome_id, genome) in enumerate(genomes):
        global tetris
        results[i-1] = tetris.train_ai(genome=genome, config=config)
    
    with open ("NEAT_results.csv", "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames = ["mean", "std", "max"])
        writer.writerow({"mean": np.mean(results), "std": np.std(results), "max": results.max()})

    if draw:
        pygame.quit()


def test_ai(winner_net, out, test_draw, seed = random.randint(1,1_000_000)):

    tetris = Tetris_game(seed)
    score = 0
    if test_draw and out is None:
        pygame.init()
        width, height = 300, 700
        pygame.display.set_mode((width, height))
    moves = 0
    while True:
        reward, done = tetris.make_move(winner_net, with_held=True)
        score += reward
        moves += 1
        if out is not None:
            frame = tetris.game.render_save_video(score, "Neat")
            frame = cv.convertScaleAbs(frame)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            out.write(frame)

        elif test_draw:
            tetris.game.render(score, framerate=60)

        if done or moves >= 5_000:
            if out is not None:
                out.release()
            elif test_draw:
                pygame.quit()
            return score, tetris.game.types_of_clears


def run_neat(config, seed=random.randint(1, 1_000_000)):
    random.seed(seed)
    p = neat.Checkpointer.restore_checkpoint('src_neat/checkpoint_neat/neat-checkpoint-37')
    #p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(
        neat.Checkpointer(1, filename_prefix="src_neat/checkpoint_neat/neat-checkpoint-")
        )
    global tetris
    tetris = Tetris_game(12)

    winner = p.run(eval_genomes, 3)
    with open("neat_best.pickle", "wb") as f:
        pickle.dump(winner, f)
