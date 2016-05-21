from time import sleep
from attention import *
from numpy.testing import assert_allclose


assert_allclose(get_half_attention_size(0), 1.0)
assert_allclose(get_half_attention_size(9), 0.1)

# centered zoomed out should return unit box
assert_allclose(attention_bounds(y=0, x=0, zoom=0), (-1, 1, -1, 1))

assert_allclose(attention_bounds(y=-0.5, x=0, zoom=0), (-1.5, 0.5, -1, 1))
assert_allclose(attention_bounds(y=-0.5, x=0.2, zoom=0), (-1.5, 0.5, -0.8, 1.2))
# zoomed in
assert_allclose(attention_bounds(y=0, x=0, zoom=9), (-0.1, 0.1, -0.1, 0.1))


img_shape = (250, 500, 3)
assert_allclose(float_to_pixel(img_shape, -1, 1, -1, 1), (-125, 375, 0, 500))

attention_env = AttentionEnv(64)
attention_env._reset()
assert attention_env.y == 0
assert attention_env.x == 0
assert attention_env.zoom == 0
glimpse = attention_env.make_observation()
assert glimpse.shape == (64, 64, 3)
assert (0.0 <= glimpse).all() and (glimpse <= 1.0).all()
assert not np.allclose(glimpse, np.zeros(glimpse.shape)), "Not all zero"


# if we move up twice, everything should be zero
attention_env.up()
attention_env.up()
glimpse = attention_env.make_observation()
assert np.allclose(glimpse, np.zeros(glimpse.shape)), "Not all zero"
