import math
def correct(r, x, k):
    z = math.sqrt(math.pow(r, 2) + math.pow(x, 2))
    fird = math.atan2(x, r)

    fist = 180*fird/math.pi
    fist = fist+k
    fird = fist*math.pi/180

    return z*math.cos(fird), z*math.sin(fird)

