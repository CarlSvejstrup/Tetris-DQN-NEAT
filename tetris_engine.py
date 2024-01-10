import numpy as np
import cv2 as cv
import random
import concurrent.futures
import multiprocessing
import pygame

shapes = {
    "T": [(0, 0), (-1, 0), (1, 0), (0, -1)],
    "J": [(0, 0), (-1, 0), (0, -1), (0, -2)],
    "L": [(0, 0), (1, 0), (0, -1), (0, -2)],
    "Z": [(0, 0), (-1, 0), (0, -1), (1, -1)],
    "S": [(0, 0), (-1, -1), (0, -1), (1, 0)],
    "I": [(0, 0), (0, -1), (0, -2), (0, -3)],
    "O": [(0, 0), (0, -1), (-1, 0), (-1, -1)],
}
shape_names = ["T", "J", "L", "Z", "S", "I", "O"]
green = (156, 204, 101)
black = (0, 0, 0)
white = (255, 255, 255)


def rotated(shape):
    return [(-j, i) for i, j in shape]


def is_occupied(shape, anchor, board):
    for i, j in shape:
        x, y = anchor[0] + i, anchor[1] + j
        if y < 0:
            continue
        if x < 0 or x >= board.shape[0] or y >= board.shape[1] or board[x, y]:
            return True
    return False


def soft_drop(shape, anchor, board):
    new_anchor = (anchor[0], anchor[1] + 1)
    return (
        (shape, anchor)
        if is_occupied(shape, new_anchor, board)
        else (shape, new_anchor)
    )


def hard_drop(shape, anchor, board):
    soft_count = 0
    while True:
        _, anchor_new = soft_drop(shape, anchor, board)
        if anchor_new == anchor:
            return shape, anchor_new, soft_count
        soft_count += 1
        anchor = anchor_new


class Tetris:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.board = np.zeros(shape=(width, height), dtype=np.float64)
        self.render_enabled = True

        # State size (Clearede lines, bumpiness, holes, height)
        self.state_size = 4

        # For running the engine
        self.score = -1
        self.anchor = None
        self.shape = None

        # Holding a piece
        self.held_shape = None
        self.held_anchor = None

        # Used for generating shapes
        self._shape_counts = [0] * len(shapes)

        # Reset after initializing
        self.reset()

    def _choose_shape(self):
        max_count = max(self._shape_counts)

        tetromino = None
        valid_tetrominos = [
            shape_names[i]
            for i in range(len(shapes))
            if self._shape_counts[i] < max_count
        ]
        if len(valid_tetrominos) == 0:
            tetromino = random.sample(shape_names, 1)[0]
        else:
            tetromino = random.sample(valid_tetrominos, 1)[0]
        self._shape_counts[shape_names.index(tetromino)] += 1
        return shapes[tetromino]

    def _new_piece(self):
        self.anchor = (self.width / 2, 1)
        self.shape = self._choose_shape()

    def _has_dropped(self):
        return is_occupied(self.shape, (self.anchor[0], self.anchor[1] + 1), self.board)

    def _clear_lines(self):
        can_clear = [np.all(self.board[:, i]) for i in range(self.height)]
        new_board = np.zeros_like(self.board)
        j = self.height - 1
        for i in range(self.height - 1, -1, -1):
            if not can_clear[i]:
                new_board[:, j] = self.board[:, i]
                j -= 1
        self.score += sum(can_clear)
        self.board = new_board

        return sum(can_clear)

    def valid_action_count(self):
        valid_action_sum = 0

        for value, fn in self.value_action_map.items():
            # If they're equal, it is not a valid action
            if fn(self.shape, self.anchor, self.board) != (self.shape, self.anchor):
                valid_action_sum += 1

        return valid_action_sum

    def reward_function(self):
        cleared_lines = self._clear_lines()
        # step_reward = cleared_lines**3 * self.width + self.soft_count

        if cleared_lines == 1:
            return 40 + self.soft_count
        elif cleared_lines == 2:
            return 100 + self.soft_count
        elif cleared_lines == 3:
            return 300 + self.soft_count
        elif cleared_lines == 4:
            return 1200 + self.soft_count
        return self.soft_count

    def step(self, action):
        pos = [action[0], 0]

        # Rotate shape n times
        for rot in range(action[1]):
            self.shape = rotated(self.shape)

        self.shape, self.anchor, self.soft_count = hard_drop(
            self.shape, pos, self.board
        )

        reward = 0
        done = False

        self._set_piece(True, self.shape, self.anchor)
        # cleared_lines = self._clear_lines()
        step_reward = self.reward_function()
        reward += step_reward
        if np.any(self.board[:, 0]):
            self.reset()
            done = True
            reward -= 25
        else:
            self._new_piece()

        return reward, done

    def reset(self):
        self.time = 0
        self.score = 0
        self._new_piece()
        self.board = np.zeros_like(self.board)

        return np.array([0 for _ in range(self.state_size)])

    def _set_piece(self, on, shape, anchor):
        """To lock a piece in the board"""
        for i, j in shape:
            x, y = i + anchor[0], j + anchor[1]
            if x < self.width and x >= 0 and y < self.height and y >= 0:
                self.board[int(anchor[0] + i), int(anchor[1] + j)] = on

    def _clear_line_dqn(self, board):
        can_clear = [np.all(board[:, i]) for i in range(self.height)]
        new_board = np.zeros_like(board)
        j = self.height - 1
        for i in range(self.height - 1, -1, -1):
            if not can_clear[i]:
                new_board[:, j] = self.board[:, i]
                j -= 1
        self.score += sum(can_clear)
        board = new_board

        return sum(can_clear), board

    def get_bumpiness_height(self, board):
        bumpiness = 0
        columns_height = [0 for _ in range(self.width)]

        for i in range(self.width):
            for j in range(self.height):
                if board.T[j][i]:
                    columns_height[i] = self.height - j
                    break
        for i in range(1, len(columns_height)):
            bumpiness += abs(columns_height[i] - columns_height[i - 1])

        return bumpiness, sum(columns_height)

    def get_holes(self, board):
        holes = 0

        for col in zip(*board.T):
            row = 0
            while row < self.height and col[row] == 0:
                row += 1
            holes += len([x for x in col[row + 1 :] if x == 0])

        return holes

    def get_current_state(self, board):
        # Getting lines which can be cleared and the new cleared board
        cleared_lines, board = self._clear_line_dqn(board)

        # Getting number of holes that are impossible to fill
        holes = self.get_holes(board)

        # Getting bumpiness / sum of difference between each adjacent column
        bumpiness, height = self.get_bumpiness_height(board)

        return np.array([cleared_lines, holes, bumpiness, height])

    def get_next_states(self, shape, anchor, held):
        old_shape = shape
        old_anchor = anchor

        states = {}
        # Loop to try each possibility for the current shape

        for rotation in range(4):
            max_x = int(max([s[0] for s in shape]))
            min_x = int(min([s[0] for s in shape]))

            for x in range(abs(min_x), self.width - max_x):
                # Try current position
                pos = [x, 0]
                while not is_occupied(shape, pos, self.board):
                    pos[1] += 1
                pos[1] -= 1

                anchor = pos
                self._set_piece(True, shape, anchor)
                states[(x, rotation, held)] = self.get_current_state(self.board[:])
                self._set_piece(False, shape, anchor)
                anchor = old_anchor

            shape = rotated(shape)

        return states

    # Merges states from held piece and not held piece
    def merge_next_states(self):
        if self.held_shape == None:
            self.held_shape, self.held_anchor = self.shape, self.anchor
            self._new_piece
        next_state = self.get_next_states(self.shape, self.anchor, held=False)
        if self.shape != self.held_shape:
            next_state_held = self.get_next_states(
                self.held_shape, self.held_anchor, held=True
            )
            next_state.update(next_state_held)
        return next_state

    def hold_shape(self):
        self.held_shape = self.shape
        self.held_anchor = self.anchor

    def get_shape_letter(self, shape):
        # Reverse lookup to find the shape letter for the given coordinates
        for letter, shape_coords in shapes.items():
            if shape_coords == shape:
                return letter
        return "None"

    def toggle_render(self):
        self.render_enabled = not self.render_enabled

    def render(self, score, framerate=1):
        if self.render_enabled:
            self._set_piece(True, self.shape, self.anchor)
            board = self.board[:].T
            board = [
                [green if board[i][j] else black for j in range(self.width)]
                for i in range(self.height)
            ]
            self._set_piece(False, self.shape, self.anchor)

            img = np.array(board).reshape((self.height, self.width, 3)).astype(np.uint8)
            img = cv.resize(
                img, (self.width * 25, self.height * 25), interpolation=cv.INTER_NEAREST
            )

            # To draw lines every 25 pixels
            img[[i * 25 for i in range(self.height)], :, :] = 0
            img[:, [i * 25 for i in range(self.width)], :] = 0

            # Add extra spaces on the top to display game score and holding piece
            extra_spaces = np.zeros((5 * 25, self.width * 25, 3))

            cv.putText(
                extra_spaces,
                "Score: " + str(score),
                (15, 35),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                white,
                2,
                cv.LINE_AA,
            )

            # Convert to shape letter
            # held_shape_letter = self.reverse_shape(self.held_shape)

            # Checks if there is a held_shape
            if self.held_shape:
                held_shape_letter = self.get_shape_letter(self.held_shape)

                cv.putText(
                    extra_spaces,
                    "Hold: " + held_shape_letter,
                    (15, 80),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    white,
                    2,
                    cv.LINE_AA,
                )

            # Add extra spaces to the board image
            img = np.concatenate((extra_spaces, img), axis=0)

            # Draw horizontal lines to separate board and extra space area
            img[90, :, :] = white

            # Convert the image to a Pygame surface
            pygame_img = pygame.surfarray.make_surface(img.swapaxes(0, 1))

            # Display the Pygame surface
            pygame.display.get_surface().blit(pygame_img, (0, 0))

            # Use Pygame clock to control frame rate
            pygame.time.Clock().tick(framerate)

            # Update display
            pygame.display.flip()

            # Wait for a short time to allow other events to be handled
            pygame.time.wait(1)

    def render1(self, score, framerate=1):
        if self.render_enabled:
            self._set_piece(True, self.shape, self.anchor)
            board = self.board[:].T
            board = [
                [green if board[i][j] else black for j in range(self.width)]
                for i in range(self.height)
            ]
            self._set_piece(False, self.shape, self.anchor)

            img = np.array(board).reshape((self.height, self.width, 3)).astype(np.uint8)
            img = cv.resize(
                img, (self.width * 25, self.height * 25), interpolation=cv.INTER_NEAREST
            )

            # To draw lines every 25 pixels
            img[[i * 25 for i in range(self.height)], :, :] = 0
            img[:, [i * 25 for i in range(self.width)], :] = 0

            # Add extra spaces on the top to display game score and holding piece
            extra_spaces = np.zeros((5 * 25, self.width * 25, 3))

            cv.putText(
                extra_spaces,
                "Score: " + str(score),
                (15, 35),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                white,
                2,
                cv.LINE_AA,
            )

            # Convert to shape letter
            # held_shape_letter = self.reverse_shape(self.held_shape)

            # Checks if there is a held_shape

            if self.held_shape:
                held_shape_letter = self.get_shape_letter(self.held_shape)

                cv.putText(
                    extra_spaces,
                    "Hold: " + held_shape_letter,
                    (15, 80),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    white,
                    2,
                    cv.LINE_AA,
                )

            # Add extra spaces to the board image
            img = np.concatenate((extra_spaces, img), axis=0)

            # Draw horizontal lines to separate board and extra space area
            img[90, :, :] = white

            cv.imshow("DQN Tetris", img)
            cv.waitKey(framerate)
