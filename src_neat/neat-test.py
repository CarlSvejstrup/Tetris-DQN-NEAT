from neat_functions import *
import cv2 as cv

make_video = False
out = None
times_to_repeat = 10
scores = np.zeros(times_to_repeat)

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config.txt')

config = neat.Config(neat.DefaultGenome,
                     neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, 
                     neat.DefaultStagnation,
                     config_path)

for i in times_to_repeat:
    if make_video and not os.path.exists("output_video.mp4"):
        # Set the video output settings
        output_file = 'output_video.mp4'
        fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec based on the file extension
        fps = 30.0  # Adjust the frames per second as needed

        # Create a VideoWriter object
        out = cv.VideoWriter(output_file, fourcc, fps, (width, height))  # Replace width and height with the size of your frames

    scores[i] = test_ai(config, out, True)
    if make_video and not os.path.exists("output_video.mp4"):
        # Release the VideoWriter object
        out.release()
print(f"mean score: {sco}")