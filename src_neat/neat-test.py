from neat_functions import *
import visualize
import csv
make_video = False
draw = False

out = None
times_to_repeat = 125
scores = np.zeros(times_to_repeat)
types_of_clears = []
height = 610
width = 250

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config.txt')

config = neat.Config(neat.DefaultGenome,
                     neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, 
                     neat.DefaultStagnation,
                     config_path)
with open("neat_best.pickle", "rb") as f:
    winner = pickle.load(f)

node_names = {-4: 'cleared_lines', -3: 'holes', -2: 'bumpiness', -1: 'height', 0: 'score'}
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
visualize.draw_net(config, winner, True, node_names=node_names)

visualize.plot_stats(stats, ylog=False, view=True)

visualize.plot_species(stats, view=True)
exit()
for i in range(times_to_repeat):
    if make_video:
        # Set the video output settings
        output_file = f'output_video_{i}.mp4'
        fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec based on the file extension
        fps = 12  # Adjust the frames per second as needed

        # Create a VideoWriter object
        out = cv.VideoWriter(output_file, fourcc, fps, (width, height))  # Replace width and height with the size of your frames
    score, type_cleared = test_ai(winner_net, out, draw, seed=i+1)
    scores[i] = score
    types_of_clears.append(type_cleared)
    print(type_cleared)


    with open ("experiment_data/study/NEAT_study_data_lines_cleared.csv", "w", newline="") as file:
        writer = csv.writer(file)
        for clears in types_of_clears:
            writer.writerow(
                [clears["1"],
                clears["2"],
                clears["3"],
                clears["4"]]
            )
