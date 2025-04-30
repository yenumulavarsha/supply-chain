# def knapsack_01(weights, values, capacity):
#     n = len(values)
#     dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

#     for i in range(1, n + 1):
#         for w in range(1, capacity + 1):
#             if weights[i - 1] <= w:
#                 dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
#             else:
#                 dp[i][w] = dp[i - 1][w]

#     return dp[n][capacity]

# def print_knapsack_items(weights, values, capacity):
#     n = len(values)
#     dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

#     for i in range(1, n + 1):
#         for w in range(1, capacity + 1):
#             if weights[i - 1] <= w:
#                 dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
#             else:
#                 dp[i][w] = dp[i - 1][w]

#     w = capacity
#     items = []
#     for i in range(n, 0, -1):
#         if dp[i][w] != dp[i - 1][w]:
#             items.append(i - 1)
#             w -= weights[i - 1]

#     return items[::-1]

# def main():
#     n = int(input("Enter the number of items: "))
#     weights = list(map(int, input("Enter weights (space-separated): ").split()))
#     values = list(map(int, input("Enter profits (space-separated): ").split()))
#     capacity = int(input("Enter the knapsack capacity: "))

#     if len(weights) != n or len(values) != n:
#         print("Invalid input length.")
#         return

#     max_value = knapsack_01(weights, values, capacity)
#     items = print_knapsack_items(weights, values, capacity)

#     print(f"\nMaximum value: {max_value}")
#     print("Selected items:")
#     for i in items:
#         print(f"Item {i + 1} (Weight: {weights[i]}, Value: {values[i]})")

# if __name__ == "__main__":
#     main()


def tsp(d):
    n = len(d)
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1][0] = 0
    for mask in range(1, 1 << n):
        for i in range(n):
            if (mask & (1 << i)) == 0:
                continue
            for j in range(n):
                if i == j or (mask & (1 << j)) == 0:
                    continue
                dp[mask][i] = min(dp[mask][i], dp[mask ^ (1 << i)][j] + d[j][i])

    return min(dp[-1][i] + d[i][0] for i in range(1, n))

def main():
    n = int(input("Enter number of cities: "))
    d = []
    for i in range(n):
        d.append(list(map(int, input(f"{i + 1}: ").split())))

    print("Minimum distance:", tsp(d))

if __name__ == "__main__":
    main()






def floyd_warshall(d):
    n = len(d)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    return d

def print_distance_matrix(d):
    n = len(d)
    for i in range(n):
        for j in range(n):
            if d[i][j] == float('inf'):
                print("INF", end=" ")
            else:
                print(d[i][j], end=" ")
        print()

def main():
    n = int(input("Enter number of vertices: "))
    d = []
    for i in range(n):
        d.append(list(map(lambda x: float('inf') if x == 'INF' else int(x), input(f"Enter distances from vertex {i + 1} (space-separated), use INF for infinity: ").split())))
    
    print("Original Distance Matrix:")
    print_distance_matrix(d)
    
    d = floyd_warshall(d)
    
    print("Shortest Distance Matrix:")
    print_distance_matrix(d)

if __name__ == "__main__":
    main()
    
    
    


def warshall(r):
    n = len(r)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                r[i][j] = r[i][j] or (r[i][k] and r[k][j])
    return r

def print_matrix(r):
    n = len(r)
    for i in range(n):
        for j in range(n):
            print(int(r[i][j]), end=" ")
        print()

def main():
    n = int(input("Enter number of vertices: "))
    r = []
    for i in range(n):
        r.append(list(map(int, input(f"Enter adjacency matrix row {i + 1} (space-separated): ").split())))
    
    print("Original Adjacency Matrix:")
    print_matrix(r)
    
    r = warshall(r)
    
    print("Transitive Closure Matrix:")
    print_matrix(r)

if __name__ == "__main__":
    main()
