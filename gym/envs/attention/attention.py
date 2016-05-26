from scipy.misc import imresize
from skimage.io import imread
from gym import spaces
import gym
import time
import os
import re
import sys
import numpy as np

from synset import *
from common import *


human_size = 400
human_bottom_padding = 1
red = [255.0, 0, 0]

DEBUG = False

def log(msg):
    if DEBUG: print msg



class AttentionEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        #"video.frames_per_second" : 2,
    }

    def __init__(self, glimpse_size):
        self.epochs_complete = 0 
        self.viewer = None
        self.num_steps = 0
        self.index = 0
        # following are set by reset()
        self.y = None
        self.x = None
        self.zoom = None

        self.data_dir = os.environ.get('IMAGENET_DIR')
        if not self.data_dir:
            print "Set IMAGENET_DIR env variable"
            sys.exit(1)

        self.glimpse_size = glimpse_size

        self.data = load_data(self.data_dir)
        np.random.shuffle(self.data)

        print "num actions %d" % num_actions
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(-1.0, 1.0,
                                            (glimpse_size, glimpse_size, 3))


    def backdoor_observation_params(self):
        """
        Because storing a many observation images takes up a lot of space
        (which is what is required in DQN). We provide this backdoor function
        to get the parameters of an observation. Then the agent can store only
        the parameters instead of the entire image. The agent can then call
        make_observation to reconstruct the observation.
        """
        return self.img_fn, self.y, self.x, self.zoom

    def _human(self, glimpse):
        # This will always return a 400 x 400 image
        # with the current image (self.img) shown in the middle
        # it will draw a red line around the attention box.

        # first draw the image in the middle.
        img_shape = self.img.shape
        img_height = img_shape[0]
        img_width = img_shape[1]
        assert 3 == img_shape[2], "img should be RGB"

        longer_side = max(img_height, img_width)
        shorter_side = min(img_height, img_width)

        scale = float(human_size) / float(longer_side)

        new_shape_shorter = int(shorter_side * scale)
        shorter_padding = human_size - new_shape_shorter

        pad_a = int(shorter_padding / 2)
        pad_b = shorter_padding - pad_a
        if img_height < img_width:
            pad = [[pad_a, pad_b], [0, 0], [0, 0]]
            new_shape = (new_shape_shorter, human_size)
        else:
            pad = [[0, 0], [pad_a, pad_b], [0, 0]]
            new_shape = (human_size, new_shape_shorter)

        # add room at the bottom to display what we're padding to the network:
        pad[0][1] += 2 * human_bottom_padding + self.glimpse_size

        resized = imresize(self.img, new_shape)
        human_img = np.pad(resized, pad, 'constant', constant_values=0)

        # now draw the attention box
        y_min, y_max, x_min, x_max = attention_bounds(self.y, self.x, self.zoom)

        y_min_px, y_max_px, x_min_px, x_max_px = float_to_pixel((human_size, human_size),
            y_min, y_max, x_min, x_max)

        y_max_px -= 1
        x_max_px -= 1


        def in_bounds(px):
            return px >= 0 and px < human_size

        def bound(px):
            if px < 0:
                #print "too small", px
                px = 0
            if px >= human_size:
                #print "too big", px
                px = human_size - 1
            return px

        # top
        if in_bounds(y_min_px):
            human_img[y_min_px,bound(x_min_px):bound(x_max_px),:] = red 

        # bottom
        if in_bounds(y_max_px):
            human_img[y_max_px,bound(x_min_px):bound(x_max_px),:] = red 

        # left
        if in_bounds(x_min_px):
            human_img[bound(y_min_px):bound(y_max_px),x_min_px,:] = red

        # right
        if in_bounds(x_max_px):
            human_img[bound(y_min_px):bound(y_max_px),x_max_px,:] = red

        # Now plce the glimpse at the bottom
        glimpse_y_min = human_size + human_bottom_padding
        glimpse_y_max = glimpse_y_min + self.glimpse_size
        glimpse_x_min = human_bottom_padding 
        glimpse_x_max = human_bottom_padding + self.glimpse_size
        #print "(glimpse_y_min, glimpse_y_max, glimpse_x_min, glimpse_x_max)", (glimpse_y_min, glimpse_y_max, glimpse_x_min, glimpse_x_max)
        human_img[glimpse_y_min:glimpse_y_max,\
                  glimpse_x_min:glimpse_x_max, :] = 255 * glimpse


        return human_img

    def epoch_complete(self):
        self.epochs_complete += 1
        np.random.shuffle(self.data)
        self.index = 0

    def load_img(self):
        if self.index >= len(self.data):
            self.epoch_complete()

        self.current = self.data[self.index]
        img_fn = self.current['filename']

        self.img_fn = img_fn
        full_img_fn = os.path.join(self.data_dir, img_fn)
        self.img = imread(full_img_fn)

        if len(self.img.shape) == 2:
            self.img = np.dstack([self.img, self.img, self.img])

        if DEBUG:
            m = np.mean(self.img)
            assert 0.0 <= m and m <= 1.0

        log("loaded %s" % img_fn)

    def _reset(self):
        self.num_steps = 0
        self.y = 0
        self.x = 0
        self.zoom = 0

        while True:
            self.index += 1
            if self.index > 100000:
                raise NotImplementedError
            try:
                self.load_img()
            except IOError:
                pass
            else:
                break

        return self._make_observation()

    def _make_observation(self):
        return make_observation(img=self.img,
            glimpse_size=self.glimpse_size, 
            y=self.y, 
            x=self.x,
            zoom=self.zoom)

    def _render(self, mode="human", close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
            return

        glimpse = self._make_observation()

        if mode == 'rgb_array':
            return glimpse

        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            human_img = self._human(glimpse)
            self.viewer.imshow(human_img)

    def up(self):
        half_attention_size = get_half_attention_size(self.zoom)
        self.y = self.y - half_attention_size
        log("up")

    def down(self):
        half_attention_size = get_half_attention_size(self.zoom)
        self.y = self.y + half_attention_size
        log("down")

    def left(self):
        half_attention_size = get_half_attention_size(self.zoom)
        self.x = self.x - half_attention_size
        log("left")

    def right(self):
        half_attention_size = get_half_attention_size(self.zoom)
        self.x = self.x + half_attention_size
        log("right")

    def zoom_in(self):
        self.zoom = min(9, self.zoom + 1)
        log("zoom in")

    def zoom_out(self):
        self.zoom = max(0, self.zoom - 1)
        log("zoom out")

    def _step(self, action):
        self.num_steps += 1
        max_steps = 10
        done = False
        reward = 0

        action_type, category = action_str(action)

        if self.num_steps >= max_steps:
            done = True
            reward = -1.0

        elif action_type == "category":
            done = True
            if category == self.current["label_index"]:
                # More steps means less reward.
                #reward = 1.0 / self.num_steps
                reward = 1.0
            else:
                #reward = (-1.0 / max_steps) * self.num_steps
                #reward = -1.0
                pass
        elif action_type == "up":
            self.up()
        elif action_type == "right":
            self.right()
        elif action_type == "down":
            self.down()
        elif action_type == "left":
            self.left()
        elif action_type == "zoom_in":
            self.zoom_in()
        elif action_type == "zoom_out":
            self.zoom_out()
        else:
            assert False

        observation = self._make_observation()
        info = self.current["label_index"]

        return observation, reward, done, info


def file_list(data_dir):
    cmd = 'cd %s && find . | grep JPEG' % data_dir
    filenames = os.popen(cmd).read().splitlines()
    return filenames


def load_data(data_dir):
    data = []
    i = 0

    log("listing files in %s" % data_dir)
    start_time = time.time()
    files = file_list(data_dir)
    duration = time.time() - start_time
    log("took %f sec" % duration)

    for img_fn in files:
        assert '.JPEG' == os.path.splitext(img_fn)[1]

        label_name = re.search(r'(n\d+)', img_fn).group(1)

        label_index = synset_map[label_name]["index"]

        data.append({
            "filename": img_fn,
            "label_name": label_name,
            "label_index": label_index,
            #"desc": synset[label_index],
        })

    return data

