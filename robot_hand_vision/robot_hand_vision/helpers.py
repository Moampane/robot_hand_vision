# get max x or y dimension of a hand
def get_max_dim(hand, dim):
    if dim == 'y':
        return max([landmark.y for landmark in hand.landmark])
    elif dim == 'x':
        return max([landmark.x for landmark in hand.landmark])
    
# get min x or y dimension of a hand
def get_min_dim(hand, dim):
    if dim == 'y':
        return min([landmark.y for landmark in hand.landmark])
    elif dim == 'x':
        return min([landmark.x for landmark in hand.landmark])