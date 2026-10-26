import numpy as np
W, H = 16, 10
grid = np.ones((W, H), dtype=np.int32)
walls = []
def wall(x, y):
    grid[x, y] = 0; walls.append((x, y))
for y in [0,1,3,4,5,6,8,9]: wall(4, y)
for y in [0,2,3,4,6,7,9]: wall(8, y)
for y in [0,1,2,4,5,7,8,9]: wall(12, y)
for x in [0,2,3]: wall(x, 5)
for x in [5,7]: wall(x, 3)
for x in [9,11]: wall(x, 7)
for x in [9,10]: wall(x, 2)
for x in [13,15]: wall(x, 4)
for x in [13,14]: wall(x, 8)
for (x, y) in [(2,2),(1,7),(6,8),(5,0),(7,1),(10,4),(14,1),(13,6),(3,9),(9,6)]:
    wall(x, y)
DOORS = [(4,2),(4,7),(8,1),(8,5),(8,8),(12,3),(12,6),(1,5),(6,3),(10,7),(11,2),(14,4),(15,8)]
GAUGES = {"G1": ((1,8), 2.5), "G2": ((6,0), 5.8), "G3": ((14,7), 8.1), "G4": ((2,1), 3.7), "G5": ((10,5), 6.4), "G6": ((13,2), 4.2)}
START = (0,0)
def bfs(start, goal):
    from collections import deque
    q = deque([start]); prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal: break
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nxt = (cur[0]+dx, cur[1]+dy)
            if 0 <= nxt[0] < W and 0 <= nxt[1] < H and grid[nxt] == 1 and nxt not in prev:
                prev[nxt] = cur; q.append(nxt)
    if goal not in prev: return None
    path = []; cur = goal
    while cur is not None:
        path.append(cur); cur = prev[cur]
    return path[::-1]
if __name__ == "__main__":
    print(f"GRID {W}x{H} free={int(grid.sum())} walls={len(walls)} doors={len(DOORS)} {DOORS}")
    gmap = {}
    for name,(pos,_) in GAUGES.items(): gmap[pos] = name[1]
    for pos in DOORS: gmap[pos] = "D"
    gmap[START] = "S"
    for y in range(H-1, -1, -1):
        row = ""
        for x in range(W):
            if (x,y) in gmap: row += gmap[(x,y)]
            elif grid[x,y] == 0: row += "#"
            else: row += "."
        print(f"y={y} {row}")
    for name,(pos,val) in GAUGES.items():
        p = bfs(START, pos)
        print(f"{name} {pos} {val}bar BFS_len={len(p) if p else None}")
    # unreachable free cells
    from collections import deque
    seen = {START}; q = deque([START])
    while q:
        cur = q.popleft()
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nxt = (cur[0]+dx, cur[1]+dy)
            if 0 <= nxt[0] < W and 0 <= nxt[1] < H and grid[nxt]==1 and nxt not in seen:
                seen.add(nxt); q.append(nxt)
    print("UNREACHABLE_FREE", sorted([(x,y) for x in range(W) for y in range(H) if grid[x,y]==1 and (x,y) not in seen]))
