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
max_score = 10_000_000
tetris_bonus = 1_000

# skift directory for at kunne importere Tetris fra tetris_engine
current_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.join(current_dir, "..")
sys.path.append(project_dir)
from tetris_engine import Tetris


class Tetris_game:
    def __init__(self) -> None:
        self.game = Tetris(10, 20)

    def train_ai(self, genome, config) -> None:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        genome.fitness = 0

        while True:
            reward, done = self.make_move(net, with_held=False)
            genome.fitness += reward

            if draw:
                self.game.render1(genome.fitness, framerate=60)

            if genome.fitness > max_score:
                genome.fitness = genome.fitness + tetris_bonus * self.game.tetris_clear
                done = True

            if done:
                genome.fitness = int(genome.fitness)
                return genome.fitness

    def make_move(self, net, with_held=False):
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
        tetris = Tetris_game()
        results[i-1] = tetris.train_ai(genome=genome, config=config)
    
    with open ("NEAT_results.csv", "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames = ["mean", "std", "max"])
        writer.writerow({"mean": np.mean(results), "std": np.std(results), "max": results.max()})

    if draw:
        pygame.quit()


def test_ai(config, out, test_draw):
    with open("neat_best.pickle", "rb") as f:
        winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    tetris = Tetris_game()
    score = 0
    if test_draw and out is None:
        pygame.init()
        width, height = 300, 700
        pygame.display.set_mode((width, height))
    while True:
        reward, done = tetris.make_move(winner_net, with_held=True)
        score += reward

        if out is not None:
            frame = tetris.game.render_save_video(score, "Neat")
            frame = cv.convertScaleAbs(frame)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            out.write(frame)

        elif test_draw:
            tetris.game.render1(score, framerate=60)

        if done:
            if out is None:
                out.release()
            elif test_draw:
                pygame.quit()
            return score


def run_neat(config, seed=random.randint(1, 1_000_000)):
    random.seed(seed)
    p = neat.Checkpointer.restore_checkpoint('src_neat/checkpoint_neat/neat-checkpoint-25')
    #p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(
        neat.Checkpointer(
            1, filename_prefix="src_neat/checkpoint_neat/neat-checkpoint-"
        )
    )

    winner = p.run(eval_genomes, int(10_000/50))
    with open("neat_best.pickle", "wb") as f:
        pickle.dump(winner, f)
