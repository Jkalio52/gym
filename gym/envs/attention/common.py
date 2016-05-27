from skimage.io import imread
from scipy.misc import imresize
import numpy as np

num_categories = 3
num_directional_actions = 4  # up, right, down, left
num_zoom_actions = 2  # zoom in, zoom out
num_actions = num_categories + num_directional_actions + num_zoom_actions

DEBUG = False


def action_human_str(action):
    action_type, category = action_str(action)
    if action_type == 'category':
        return 'c' + str(category)
    elif action_type == 'zoom_in':
        return 'zi'
    elif action_type == 'zoom_out':
        return 'zo'
    return action_type[:1]


def action_str(action):
    assert isinstance(action, int)
    if action < num_categories:
        return "category", action
    elif action < num_categories + num_directional_actions:
        direction = action - num_categories
        if direction == 0:
            return "up", None
        elif direction == 1:
            return "right", None
        elif direction == 2:
            return "down", None
        elif direction == 3:
            return "left", None
    else:
        zoom = action - (num_categories + num_directional_actions)
        if zoom == 0:
            return "zoom_in", None
        elif zoom == 1:
            return "zoom_out", None

    assert False, 'unreachable'


def make_observation(img, glimpse_size, y, x, zoom):
    # Output is always glimpse_size x glimpse_size x 3

    img_shape = img.shape
    img_height = img_shape[0]
    img_width = img_shape[1]
    assert 3 == img_shape[2], "img should be RGB"

    y_min, y_max, x_min, x_max = attention_bounds(y, x, zoom)

    y_min_px, y_max_px, x_min_px, x_max_px = float_to_pixel(
        img_shape, y_min, y_max, x_min, x_max)

    if y_min_px >= img_height or y_max_px <= 0 or \
       x_min_px >= img_width  or x_max_px <= 0:
        return np.zeros((glimpse_size, glimpse_size, 3))

    pad_top = max(0, -y_min_px)
    pad_bottom = max(0, y_max_px - img_height)
    pad_left = max(0, -x_min_px)
    pad_right = max(0, x_max_px - img_width)

    y_min_px = max(0, y_min_px)
    y_max_px = min(img_height, y_max_px)
    x_min_px = max(0, x_min_px)
    x_max_px = min(img_width, x_max_px)

    crop = img[y_min_px:y_max_px, x_min_px:x_max_px, :]

    pad = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

    padded_crop = np.pad(crop, pad, 'constant', constant_values=0)

    #print "img shape", img_shape
    #print "pad", pad
    #print "crop shape", crop.shape
    #print padded_crop.shape
    #print "mean padded_crop", np.mean(padded_crop)

    observation = imresize(padded_crop, (glimpse_size, glimpse_size))
    assert observation.shape == (glimpse_size, glimpse_size, 3)

    if DEBUG:
        print "mean observation before scale", np.mean(observation)

    observation = observation / 255.0  # scale between 0 and 1
    observation -= 0.5  # between -0.5 and 0.5
    observation *= 2.0  # between -1 and 1

    if DEBUG:
        print "mean observation after scale", np.mean(observation)
        assert -1.0 <= np.min(observation) and np.max(observation) <= 1.0

    return observation


def float_to_pixel(img_shape, y1, y2, x1, x2):
    """
    Coordinate transform from unit image [-1,1]^2 to pixels. 
    Returns integer pairs representing pixels in the image.
    Returned values may be negative or beyond the bounds of the image.
    """
    height = img_shape[0]
    width = img_shape[1]
    longer_side = max(height, width)
    scale = (longer_side / 2.0)

    y_center = (height / 2.0)
    x_center = (width / 2.0)

    y1_px = int(scale * y1 + y_center)
    y2_px = int(scale * y2 + y_center)

    x1_px = int(scale * x1 + x_center)
    x2_px = int(scale * x2 + x_center)

    return y1_px, y2_px, x1_px, x2_px


def get_half_attention_size(zoom):
    # If zoom is 0, we're zoomed all the way out. Since image fills
    # [-1,1] x [-1,1], we want the attention square to be 2 x 2
    # Thus half the attention should be 1.0.
    # When zoom is 9, we want it to zoom in a factor of 10. So 
    # attention square should stretch from -0.1 to 0.1
    assert isinstance(zoom, int)
    assert 0 <= zoom and zoom <= 9
    half_attention_size = ((10 - zoom) / 10.0)
    return half_attention_size


def attention_bounds(y, x, zoom):
    """
    Arguments y and x should be abstract coordinates float in the [-1,1]
    scale.

    Returns four floats (y_min, y_max, x_min, x_max) in unit image scale.
    Give the boundaries of the attention given the input image size.

    Use float_to_pixel to change these to integer pixel values.
    """
    half_attention_size = get_half_attention_size(zoom)

    y_max = y + half_attention_size
    y_min = y - half_attention_size

    x_min = x - half_attention_size
    x_max = x + half_attention_size

    return y_min, y_max, x_min, x_max
