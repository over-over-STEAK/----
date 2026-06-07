import random
import math

# 測試資料：店家與客人座標
points = [
    (0, 0),   # 店家
    (2, 3),   # 1號客人
    (5, 4),   # 2號客人
    (1, 6),   # 3號客人
    (7, 2)    # 4號客人
]

def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def total_distance(route):
    total = 0
    for i in range(len(route) - 1):
        total += distance(points[route[i]], points[route[i + 1]])
    return total

def height(route):
    return -total_distance(route)

def neighbor(route):
    new_route = route[:]
    
    i, j = sorted(random.sample(range(1, len(route) - 1), 2))
    
    new_route[i:j+1] = reversed(new_route[i:j+1])
    return new_route

def hill_climbing():
    n = len(points) - 1

    # 初始解：店家 → 1 → 2 → ... → n → 店家
    current_route = list(range(n + 1)) + [0]

    current_height = height(current_route)

    for _ in range(1000):
        new_route = neighbor(current_route)
        new_height = height(new_route)

        if new_height > current_height:
            current_route = new_route
            current_height = new_height

    return current_route, total_distance(current_route)

best_route, best_distance = hill_climbing()

print("最佳路線：", best_route)
print("總距離：", best_distance)