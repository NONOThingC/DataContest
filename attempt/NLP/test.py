import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

# w: width of the building.
# h: height of the building.
w, h = [int(i) for i in input().split()]
n = int(input())  # maximum number of turns before game over.
x0, y0 = [int(i) for i in input().split()]
alpha=1
# game loop
while True:
    bomb_dir = input()  # the direction of the bombs from batman's current location (U, UR, R, DR, D, DL, L or UL)

    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr)
    m=0
    n=0

    if(bomb_dir[0]=='U'):
        if(y0-alpha<0):
            alpha=1
        y0=y0-alpha
        if(bomb_dir=='UR'):
            if(x0+alpha>w):
                alpha=1
            x0=x0+alpha
        if(bomb_dir=='UL'):
            if(x0-alpha<0):
                alpha=1
            x0=x0-alpha  
    if(bomb_dir[0]=='D'):
        if(y0+alpha>h):
            alpha=1
        y0=y0+alpha
        if(bomb_dir=='DR'):
            if(x0+alpha>w):
                alpha=1
            x0=x0+alpha
        if(bomb_dir=='DL'):
            if(x0-alpha<0):
                alpha=1
            x0=x0-alpha
    if(bomb_dir[0]=='L'):
        if(x0-alpha<0):
            alpha=1
        X0=X0-alpha
    if(bomb_dir[0]=='R'):
        if(x0+alpha>w):
            alpha=1
        X0=X0+alpha
    alpha=alpha*2

    print("{} {}".format(x0,y0))
    
    
    # the location of the next window Batman should jump to.
    