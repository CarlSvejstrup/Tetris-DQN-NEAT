from neat_functions import *
import csv
make_video = False
draw = False

out = None
times_to_repeat = 125
scores = np.zeros(times_to_repeat)
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
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
#neat.visualize.draw_net(winner, view=True, filename="xor2-all.gv")
for i in range(times_to_repeat):
    if make_video:
        # Set the video output settings
        output_file = f'output_video_{i}.mp4'
        fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec based on the file extension
        fps = 12  # Adjust the frames per second as needed

        # Create a VideoWriter object
        out = cv.VideoWriter(output_file, fourcc, fps, (width, height))  # Replace width and height with the size of your frames

    scores[i] = test_ai(winner_net, out, draw, seed=i+1)
    print(i+1)


    with open ("experiment_data/study/NEAT_study_data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        for score in scores:
            writer.writerow([score])
