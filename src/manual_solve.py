#!/usr/bin/python

import os, sys
import json
import numpy as np
import re

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.

"""
    Name:         Patrick Dorrian
    Student ID:   17372533
    GitHub Repo:  github.com/PDorrian/ARC
"""


def solve_6ecd11f4(input_grid):
    """

    Input:
        -  A large black grid of variable size
        -  A small multi-coloured square of variable size placed anywhere on the grid.
        -  A large pattern made up of n*n blocks of pixels, similar to an old video game sprite. The pixel squares are
           of variable size and the pattern itself is also made up of a variable number of pixel squares.

    Output:
        -  The large pattern simplified to its smallest form, the pixels of which will be coloured with the colours
           of the small multi-coloured square.

    Transformation:
        1. The regions in which the large pattern and the small multi-coloured square lie are identified.
        2. The dimensions of the coloured square are used to find the locations of the first pixel in each
           n*n pixel square. This is used to determine the pattern in the output grid.
        3. The multi-coloured square is sampled to get the correct colourings for the output pattern.

    Training grids solved:  3/3
    Testing grids solved:   1/1

    """

    # Find color of main section.
    main_color = max(range(1, 10), key=lambda i: np.sum(input_grid == i))

    # Find bounds and size of main shape.
    main_shape = np.where(input_grid == main_color)       # Locations of cells containing the main color
    main_shape = [(x,y) for x,y in zip(*main_shape)]      # Reformat as pairs of coordinates.
    main_shape = [(x,y) for x,y in main_shape             # Discard stray main colors (from color square).
                  if any((a,b) in main_shape for a,b in [(x,y+1), (x,y-1), (x+1,y), (x-1,y)])]

    # Calculate size of bounding square of the main shape.
    main_shape_size = max(main_shape, key=lambda pair: pair[0])[0] - min(main_shape, key=lambda pair: pair[0])[0] + 1
    # Top left coordinate of the main shape.
    main_shape_start_x, main_shape_start_y = main_shape[0]

    # Create copy of input and remove the main shape from it.
    # All that should remain is the color square.
    color_square = np.copy(input_grid)
    for x,y in main_shape:
        # Pixels of the main shape are set to 0.
        color_square[x,y] = 0

    # All 0 rows are removed to determine the size of the color square.
    color_square_size = len(color_square[np.any(color_square > 0, axis=1)])
    # Size of a tile in the main shape can now be determined.
    main_tile_size = main_shape_size // color_square_size

    # Remove 0 values and reshape into a square to get the color block.
    color_square = color_square[color_square > 0].reshape(color_square_size, color_square_size)

    # Loop over each tile of the color square.
    for i in range(color_square_size):
        for j in range(color_square_size):
            # Set tile to 0 if the corresponding tile in the main shape is also 0.
            if input_grid[main_shape_start_x + main_tile_size * i][main_shape_start_y + main_tile_size * j] == 0:
                color_square[i][j] = 0

    return color_square


def solve_2dd70a9a(input_grid):
    """

    Input:
        -  A large black grid of variable size
        -  A 2x1 or 1x2 green rectangle placed somewhere on the grid
        -  A 2x1 or 1x2 red rectangle placed somewhere on the grid
        -  Many cyan pixels scattered across the grid

    Output:
        -  The input grid with a green continuous line connecting the red and green rectangles while avoiding the cyan.
           The green line emanates from the green rectangle along its axis of direction.
           The line only changes direction when it is about to touch a cyan pixel

    Transformation:
        1. The locations and orientations of the red and green rectangles are determined
        2. One of the two possible directions are chosen at random and a walk begins from the green rectangle in
           that direction.
        3. When a cyan pixel is encountered, the walk changes direction. The direction is determined by using
           a heuristic measurement of the distance from the next potential step to the red rectangle.
        4. If a path is not found, the process repeats but using the other starting direction.

    Training grids solved:  3/3
    Testing grids solved:   1/1

    """

    # Identify the start and end positions.
    # Green rectangle position.
    green_rect = np.where(input_grid == 3)
    green_rect = [(x,y) for x,y in zip(*green_rect)]        # Format as (x,y) pairs.

    # Red rectangle position.
    red_rect = np.where(input_grid == 2)
    red_rect = [(x,y) for x,y in zip(*red_rect)]            # Format as (x,y) pairs.

    # Determine axis of green rectangle.
    green_ax = 'y' if green_rect[0][0] == green_rect[1][0] else 'x'

    # Try both possible directions.
    for s in [1, -1]:
        # End goal = centre of red rectangle.
        end = (red_rect[0][0] + red_rect[1][0])/2, (red_rect[0][1] + red_rect[1][1])/2
        # Start from one of the green pixels.
        start = green_rect[s]

        x, y = start        # Current position
        axis = green_ax     # Current axis of movement

        direction = 1 if axis == 'y' else -1        # Should movement be in the positive or negative direction?
        direction *= s                              # Reversed when starting from the other pixel.

        #    Explanation of direction and axis:
        #           +    -
        #       x   R    L
        #       y   D    U

        # A copy of the grid that can be safely edited.
        grid = np.copy(input_grid)

        found = False
        for i in range(50):
            nx, ny = x, y         # Candidates for next position.

            # Movement based on current axis and direction.
            if axis == 'x':
                nx += direction
            else:
                ny += direction

            # Check if next position would be a wall.
            if input_grid[nx][ny] == 8:
                nx, ny = x, y   # Reset candidate position

                # Change axis and determine new direction.
                if axis == 'x':
                    axis = 'y'
                    # Next position and direction determined by which has the shortest heuristic path to the end.
                    ny = min(ny+1, ny-1, key=lambda d: manhattan_distance((nx, d), end))
                    direction = 1 if ny > y else -1

                elif axis == 'y':
                    axis = 'x'
                    # Next position and direction determined by which has the shortest heuristic path to the end.
                    nx = min(nx+1, nx-1, key=lambda d: manhattan_distance((d, ny), end))
                    direction = 1 if nx > x else -1

            # Lock in candidate coordinates
            x, y = nx, ny

            # Check that we are within bounds
            if x<0 or y<0 or x>=len(grid) or y>=len(grid):
                break

            # Check if we have reached the end goal.
            if grid[x][y] == 2:
                return grid

            # Check that we have not intersected our own path.
            if grid[x][y] == 3 and i > 1:
                break

            # Color path green.
            grid[x][y] = 3

    # Raise exception if a valid solution is not found.
    raise Exception("No solution found.")


# Auxiliary function for solve_2dd70a9a
# Calculates Manhattan distance between two pairs of coordinates.
def manhattan_distance(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def solve_7df24a62(input_grid):
    """

        Input:
            -  A large black grid of variable size.
            -  Yellow cells peppered throughout the black grid.
            -  A blue box surrounding a small arrangement of yellow cells.

        Output:
            -  The input grid but with all clusters of yellow cells that have the same pattern as the highlighted
               cells (or some rotation of the pattern) are also highlighted in blue.

        Transformation:
            1. Identify the highlighted pattern and generate all possible rotations of said pattern.
            2. Iterate through every cell of the grid and check if the nearby cells are identical to the
               highlighted pattern.
            3. If there is a match, highlight this area.

        Training grids solved:  3/3
        Testing grids solved:   1/1

    """

    # Find location of blue rectangle (the stencil).
    ys, xs = np.where(input_grid == 1)

    # Width and height of the blue area
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    # Identify pattern within the blue area
    pattern = input_grid[ys[0]+1:ys[0]+height, xs[0]+1:xs[0]+width]
    pattern = np.where(pattern == 1, 0, pattern)

    # Generate a list of each possible pattern rotation
    pattern_rotations = [pattern, np.rot90(pattern), np.rot90(np.rot90(pattern)), np.rot90(np.rot90(np.rot90(pattern)))]

    # Iterate through each pattern rotation
    for rot in pattern_rotations:
        height, width = rot.shape    # Width and height of rotated shape
        # Iterate over each pixel in the grid
        for i in range(0, len(input_grid)-height+1):
            for j in range(0, len(input_grid[0])-width+1):
                # Generate segment with width and height of the pattern starting at the current pixel
                segment = input_grid[i:i+height, j:j+width]
                # Check if the segment is identical to the pattern
                if (segment == rot).all():
                    # Select area that will be colored blue
                    blue_area = input_grid[max(i-1, 0):i+height+1, max(0,j-1):j+width+1]
                    # Change area's black pixels to blue
                    input_grid[max(i-1, 0):i+height+1, max(0, j-1):j+width+1] = np.where(blue_area == 0, 1, blue_area)

    # Return modified grid
    return input_grid


"""

Summary/Reflection
Python features and libraries used:
    - The only external library used in any of the solve functions is numpy.
    - All other features used are standard Python functions and operations, like max() and slicing.
    
Commonalities/Differences:
    - Each function makes use of np.where() to identify regions with a specific colour.
    - The first and third functions both use a nested for loop to iterate over all cells in a region.
    - All 3 functions use .reshape to change the shapes of arrays.
    
"""


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)


def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))


if __name__ == "__main__": main()

