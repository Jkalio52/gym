
num_categories = 3
num_directional_actions = 4 # up, right, down, left
num_zoom_actions = 2 # zoom in, zoom out
num_actions = num_categories + num_directional_actions + num_zoom_actions

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
