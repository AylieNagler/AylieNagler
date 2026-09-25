"""
Aylie Lapierre Nagler
Student #: 0439007
Personal project
GitHub profile — self-solving Rubik's cube
2026-09-24
"""

import datetime
import math
import random
import sys

import numpy as np
from PIL import Image, ImageDraw


# Output size in pixels and supersampling factor for smooth edges
SIZE = 260
SUPERSAMPLE = 3

# Number of moves in each daily scramble
SCRAMBLE_LENGTH = 20

# Frames spent on each move while scrambling and while solving
SCRAMBLE_FRAMES = 4
SOLVE_FRAMES = 7

# Frames held still between phases
HOLD_SCRAMBLED = 12
HOLD_SOLVED = 30

# Milliseconds per frame
FRAME_MS = 40

# Camera tilt so the top face is visible
CAMERA_TILT = math.radians(28)

# Face normals for each move letter
MOVE_AXES = {
    "U": (0, 1, 0),
    "D": (0, -1, 0),
    "R": (1, 0, 0),
    "L": (-1, 0, 0),
    "F": (0, 0, 1),
    "B": (0, 0, -1),
}

# Sticker colours per face normal
FACE_COLOURS = {
    (0, 1, 0): (255, 255, 255),
    (0, -1, 0): (255, 213, 0),
    (1, 0, 0): (196, 30, 58),
    (-1, 0, 0): (255, 88, 0),
    (0, 0, 1): (0, 158, 96),
    (0, 0, -1): (0, 81, 186),
}

# Colour of the cube body
BODY_COLOUR = (18, 18, 18)

# Backgrounds for GitHub dark and light themes
THEMES = {
    "dark": (13, 17, 23),
    "light": (255, 255, 255),
}


# Define function to build a rotation matrix around an axis
def rotation_matrix(axis, angle):
    """
    Build a 3x3 rotation matrix for a rotation around a unit axis

    Parameters:
        axis (tuple of 3 ints or floats)
        angle (float, radians)

    Returns:
        3x3 numpy array
    """
    # Normalise the axis
    x, y, z = np.array(axis, dtype=float) / np.linalg.norm(axis)

    # Precompute sine and cosine
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c

    # Apply the Rodrigues rotation formula
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


# Define function to create a solved cube
def solved_cube():
    """
    Create the 26 visible cubies of a solved cube

    Parameters:
        none

    Returns:
        list of cubie dicts with position and sticker colours
    """
    # Start with an empty list of cubies
    cubies = []

    # Loop through every grid position except the hidden centre
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                if (x, y, z) == (0, 0, 0):
                    continue

                # Give the cubie a sticker on each outward face
                stickers = {}
                for normal, colour in FACE_COLOURS.items():
                    if np.dot(normal, (x, y, z)) == 1:
                        stickers[normal] = colour

                # Store the cubie
                cubies.append({"pos": (x, y, z), "stickers": stickers})

    return cubies


# Define function to split a move into its axis and angle
def parse_move(move):
    """
    Convert move notation like R, U' or F2 into a layer axis and angle

    Parameters:
        move (string)

    Returns:
        axis tuple, angle in radians
    """
    # Look up the face normal for the move letter
    axis = MOVE_AXES[move[0]]

    # Clockwise from the face is a negative turn around its normal
    angle = -math.pi / 2

    # Adjust the angle for prime and double moves
    if move.endswith("'"):
        angle = math.pi / 2
    elif move.endswith("2"):
        angle = -math.pi

    return axis, angle


# Define function to permanently apply a move to the cube
def apply_move(cubies, move):
    """
    Rotate one layer of the cube by a full move

    Parameters:
        cubies (list of cubie dicts)
        move (string)

    Returns:
        new list of cubie dicts
    """
    # Get the rotation for this move
    axis, angle = parse_move(move)
    matrix = rotation_matrix(axis, angle)

    # Start with an empty list for the new cube state
    turned = []

    # Loop through cubies and rotate the ones in the layer
    for cubie in cubies:
        if np.dot(cubie["pos"], axis) != 1:
            turned.append(cubie)
            continue

        # Rotate the position and snap to the grid
        pos = tuple(int(v) for v in np.rint(matrix @ cubie["pos"]))

        # Rotate each sticker normal and snap to the grid
        stickers = {}
        for normal, colour in cubie["stickers"].items():
            new_normal = tuple(int(v) for v in np.rint(matrix @ normal))
            stickers[new_normal] = colour

        # Store the rotated cubie
        turned.append({"pos": pos, "stickers": stickers})

    return turned


# Define function to make a random scramble
def make_scramble(seed):
    """
    Generate a random scramble with no face turned twice in a row

    Parameters:
        seed (int)

    Returns:
        list of move strings
    """
    # Seed the generator so each day gives a repeatable scramble
    rng = random.Random(seed)

    # Build the scramble one move at a time
    scramble = []
    while len(scramble) < SCRAMBLE_LENGTH:
        face = rng.choice(list(MOVE_AXES))

        # Skip a face that was just turned
        if scramble and scramble[-1][0] == face:
            continue

        # Add the face with a random turn type
        scramble.append(face + rng.choice(["", "'", "2"]))

    return scramble


# Define function to invert a sequence of moves
def invert_moves(moves):
    """
    Reverse a move sequence so it undoes itself

    Parameters:
        moves (list of move strings)

    Returns:
        list of move strings
    """
    # Start with an empty list of inverted moves
    inverted = []

    # Walk backwards and flip each turn direction
    for move in reversed(moves):
        if move.endswith("'"):
            inverted.append(move[0])
        elif move.endswith("2"):
            inverted.append(move)
        else:
            inverted.append(move + "'")

    return inverted


# Define function to draw one frame of the cube
def render_frame(cubies, yaw, background, move=None, progress=0.0):
    """
    Draw the cube at a camera angle, with one layer part-way through a turn

    Parameters:
        cubies (list of cubie dicts)
        yaw (float, radians)
        background (tuple of 3 ints)
        move (string or None)
        progress (float from 0 to 1)

    Returns:
        PIL image
    """
    # Set up the canvas at supersampled size
    size = SIZE * SUPERSAMPLE
    image = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(image)
    scale = size / 5.6

    # Build the camera rotation
    camera = rotation_matrix((1, 0, 0), CAMERA_TILT) @ rotation_matrix(
        (0, 1, 0), yaw
    )

    # Build the partial rotation for the turning layer
    layer_axis = None
    layer_matrix = np.identity(3)
    if move:
        layer_axis, angle = parse_move(move)
        eased = 0.5 - 0.5 * math.cos(math.pi * progress)
        layer_matrix = rotation_matrix(layer_axis, angle * eased)

    # Split cubies into the turning layer and the rest of the cube
    layer = []
    rest = []
    for cubie in cubies:
        if layer_axis and np.dot(cubie["pos"], layer_axis) == 1:
            layer.append((cubie, camera @ layer_matrix))
        else:
            rest.append((cubie, camera))

    # Draw whichever group sits farther from the camera first
    groups = [rest, layer]
    if layer_axis and (camera @ np.array(layer_axis))[2] < 0:
        groups = [layer, rest]

    # Draw each group's cubies from back to front
    for group in groups:
        group.sort(key=lambda item: (item[1] @ item[0]["pos"])[2])
        for cubie, transform in group:
            draw_cubie(draw, cubie, transform, scale, size)

    # Shrink to final size for smooth edges
    return image.resize((SIZE, SIZE), Image.LANCZOS)


# Define function to draw the visible faces of one cubie
def draw_cubie(draw, cubie, transform, scale, size):
    """
    Draw the camera-facing faces of a cubie with their stickers

    Parameters:
        draw (PIL ImageDraw)
        cubie (dict)
        transform (3x3 numpy array)
        scale (float, pixels per unit)
        size (int, canvas size in pixels)

    Returns:
        none
    """
    # Loop through the six face directions of the cubie
    for normal in FACE_COLOURS:
        view_normal = transform @ np.array(normal, dtype=float)

        # Skip faces pointing away from the camera
        if view_normal[2] <= 0.01:
            continue

        # Draw the black body face
        body = face_corners(cubie["pos"], normal, 0.5, 0.5)
        draw.polygon(project(body, transform, scale, size), fill=BODY_COLOUR)

        # Draw the sticker if this face has one
        colour = cubie["stickers"].get(normal)
        if colour:
            sticker = face_corners(cubie["pos"], normal, 0.42, 0.502)
            shade = shade_colour(colour, view_normal[2])
            draw.polygon(project(sticker, transform, scale, size), fill=shade)


# Define function to find the corners of a square on a cubie face
def face_corners(pos, normal, half_width, offset):
    """
    Find the four 3D corners of a square on one face of a cubie

    Parameters:
        pos (tuple of 3 ints)
        normal (tuple of 3 ints)
        half_width (float)
        offset (float, distance from cubie centre)

    Returns:
        list of four numpy arrays
    """
    # Find the face centre
    centre = np.array(pos, dtype=float) + offset * np.array(normal)

    # Pick the two axes that lie flat on the face
    axes = [np.eye(3)[i] for i in range(3) if normal[i] == 0]
    u, v = axes[0] * half_width, axes[1] * half_width

    # Walk the corners in order around the square
    return [centre + u + v, centre + u - v, centre - u - v, centre - u + v]


# Define function to project 3D points onto the canvas
def project(points, transform, scale, size):
    """
    Project 3D points to 2D pixel coordinates with an orthographic camera

    Parameters:
        points (list of numpy arrays)
        transform (3x3 numpy array)
        scale (float, pixels per unit)
        size (int, canvas size in pixels)

    Returns:
        list of (x, y) tuples
    """
    # Transform each point and map it to screen space
    screen = []
    for point in points:
        x, y, _ = transform @ point
        screen.append((size / 2 + x * scale, size / 2 - y * scale))

    return screen


# Define function to darken colours on faces angled away from the camera
def shade_colour(colour, facing):
    """
    Apply simple lighting based on how directly a face points at the camera

    Parameters:
        colour (tuple of 3 ints)
        facing (float from 0 to 1)

    Returns:
        tuple of 3 ints
    """
    # Blend between 65% and 100% brightness
    light = 0.65 + 0.35 * facing
    return tuple(int(channel * light) for channel in colour)


# Define function to build every frame of the animation
def build_frames(scramble, background):
    """
    Render the full loop of scrambling, holding, solving and holding

    Parameters:
        scramble (list of move strings)
        background (tuple of 3 ints)

    Returns:
        list of PIL images
    """
    # Lay out the timeline as (move, frame count) steps
    timeline = [(move, SCRAMBLE_FRAMES) for move in scramble]
    timeline.append((None, HOLD_SCRAMBLED))
    timeline += [(move, SOLVE_FRAMES) for move in invert_moves(scramble)]
    timeline.append((None, HOLD_SOLVED))

    # Spread one full camera spin across the loop
    total = sum(count for _, count in timeline)
    frames = []
    cubies = solved_cube()

    # Render every step of the timeline
    for move, count in timeline:
        for i in range(count):
            yaw = math.radians(-35) + 2 * math.pi * len(frames) / total
            progress = (i + 1) / count
            frames.append(render_frame(cubies, yaw, background, move, progress))

        # Lock the finished move into the cube state
        if move:
            cubies = apply_move(cubies, move)

    return frames


# Define function to save frames as a looping GIF
def save_gif(frames, path):
    """
    Save a list of frames as an optimised looping GIF

    Parameters:
        frames (list of PIL images)
        path (string)

    Returns:
        none
    """
    # Build one shared palette from a sample of frames so colours stay stable
    sample = Image.new("RGB", (SIZE * 8, SIZE))
    for i in range(8):
        sample.paste(frames[i * len(frames) // 8], (SIZE * i, 0))
    palette = sample.quantize(colors=48, method=Image.Quantize.MEDIANCUT)

    # Map every frame onto the shared palette
    palette_frames = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE)
        for frame in frames
    ]

    # Write the looping GIF
    palette_frames[0].save(
        path,
        save_all=True,
        append_images=palette_frames[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
        disposal=1,
    )


# Define function to run the generator
def main():
    """
    Generate today's scramble and write dark and light cube GIFs

    Parameters:
        none (optional output folder as the first command-line argument)

    Returns:
        none
    """
    # Pick the output folder
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    # Seed the scramble from today's date
    seed = int(datetime.date.today().strftime("%Y%m%d"))
    scramble = make_scramble(seed)
    print("Scramble:", " ".join(scramble))

    # Render and save one GIF per theme
    for name, background in THEMES.items():
        path = f"{out_dir}/cube-{name}.gif"
        save_gif(build_frames(scramble, background), path)
        print("Saved", path)


if __name__ == "__main__":
    main()
