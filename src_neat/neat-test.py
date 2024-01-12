from neat_functions import *
make_video = True


out = None
times_to_repeat = 1
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

for i in range(times_to_repeat):
    if make_video:
        # Set the video output settings
        output_file = f'output_video_{i}.mp4'
        fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec based on the file extension
        fps = 12  # Adjust the frames per second as needed

        # Create a VideoWriter object
        out = cv.VideoWriter(output_file, fourcc, fps, (width, height))  # Replace width and height with the size of your frames

    scores[i] = test_ai(config, out, True)
print(f"mean of score: {np.mean(scores)}\nstd of scores: {np.std(scores)}")