import numpy as np


# Функция для создания матрицы смежности
def create_adjacency_matrix(edges, n):
    matrix = np.full((n, n), 10000)  # Используем 10000 для отсутствующих путей
    for i in range(n):
        matrix[i][i] = 0  # расстояние до самого себя равно 0
    for u, v, w in edges:
        matrix[u - 1][v - 1] = w
        matrix[v - 1][u - 1] = w  # граф считается двунаправленным
    return matrix


# Алгоритм Дейкстры с сохранением данных для таблицы
def dijkstra_with_table(matrix, start, end):
    n = len(matrix)
    # Изменено на float('inf') для несуществующих путей
    dist = [float('inf')] * n
    dist[start] = 0
    visited = [False] * n
    path = [-1] * n
    iteration_data = []  # Для хранения данных итераций
    iteration = 1  # Начнем с 1, чтобы соответствовать первой итерации

    # Запоминаем первую строку (начальный узел)
    iteration_data.append((dist.copy(), iteration, -1))

    while not visited[end]:
        min_dist = float('inf')
        min_index = -1
        for v in range(n):
            if not visited[v] and dist[v] < min_dist:
                min_dist = dist[v]
                min_index = v
        if min_index == -1:
            break  # Нет доступных узлов

        visited[min_index] = True

        # Обновляем расстояния до соседей
        for v in range(n):
            if matrix[min_index][v] != 1000 and not visited[v]:  # 1000 означает отсутствие ребра
                new_dist = dist[min_index] + matrix[min_index][v]
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    path[v] = min_index

        # Запоминаем данные итерации
        iteration += 1

        # Указываем элемент, который имеет минимальное значение впервые
        min_element = None
        for idx, d in enumerate(dist):
            if not visited[idx] and d < float('inf'):
                if min_element is None or d < dist[min_element]:
                    min_element = idx

        # Запоминаем данные итерации, где элемент - это минимальный узел
        iteration_data.append((dist.copy(), iteration, min_element))

        # Если дошли до конечного узла, выходим из цикла
        if visited[end]:
            break

    return dist, path, iteration_data


# Восстановление пути по таблице Дейкстры
def restore_path(path, start, end):
    result = []
    current = end
    if path[current] == -1:
        return None  # Путь не найден
    while current != start:
        result.append(current + 1)
        current = path[current]
    result.append(start + 1)
    return result[::-1]


# Алгоритм Беллмана-Форда с выводом таблицы
def bellman_ford_with_table(matrix, start):
    n = len(matrix)
    dist = [10000] * n  # Начинаем с 10000 для отсутствующих путей
    dist[start] = 0  # Начальная вершина имеет расстояние 0 до самой себя
    iteration_data = []  # Для хранения данных итераций
    visited = [False] * n  # Массив для отслеживания уже минимальных элементов
    iteration = 2  # Нумерация итераций начинается с 2

    # Первая строка данных - это строка, соответствующая стартовому узлу
    first_row = matrix[start].copy()
    iteration_data.append((first_row, 1, -1))

    # Заполняем первую строку данными
    for i in range(n):
        if first_row[i] != 10000:  # Пропускаем 10000 (отсутствующие пути)
            dist[i] = first_row[i]

    for i in range(n - 1):  # n - 1 итераций по количеству вершин минус 1
        min_dist = float('inf')
        min_index = -1

        for u in range(n):
            # Пропускаем начальный узел
            if not visited[u] and u != start and dist[u] < min_dist:
                min_dist = dist[u]
                min_index = u

        if min_index == -1:  # Если минимальный элемент не найден, прекращаем
            break

        visited[min_index] = True  # Отмечаем элемент как посещённый

        # Обновляем расстояния до соседей
        for v in range(n):
            if matrix[min_index][v] != 10000 and dist[min_index] + matrix[min_index][v] < dist[v]:
                dist[v] = dist[min_index] + matrix[min_index][v]

        # Запоминаем текущую строку данных и минимальный элемент
        iteration_data.append((dist.copy(), iteration, min_index))
        iteration += 1

    return dist, iteration_data


# Основная функция
def main():
    # Ввод данных: количество узлов и рёбер, список рёбер и их весов
    variant_number = 13
    n = 8
    edges = []

    with open('coefficients.txt') as coefficients:
        lines = [tuple(map(int, line.split(','))) for line in coefficients]

    with open("edges.txt") as edges_file:
        for edge in edges_file:
            edge = list(map(int, edge.split(',')))
            edge[2] = lines[edge[2] - 1][0] * \
                variant_number + lines[edge[2] - 1][1]
            edges.append(tuple(edge))

    __import__('pprint').pprint(edges)

    start = 3  # Начальный узел
    end = 7  # Конечный узел

    # Матрица смежности
    matrix = create_adjacency_matrix(edges, n)
    print("Матрица смежности:")
    print(matrix)

    # Алгоритм Дейкстры с выводом таблицы
    dist_dijkstra, path_dijkstra, iteration_data_dijkstra = dijkstra_with_table(
        matrix, start - 1, end - 1)

    # Вывод таблицы итераций Дейкстры
    print("\nАлгоритм Дейкстры:")
    print(f"{'':<5}", end="")
    for i in range(1, n + 1):
        print(f"{i:<5}", end="")
    print("№\tmin\telement")

    # Начинаем с первой итерации
    for idx, data in enumerate(iteration_data_dijkstra[1:], start=1):
        row, iteration_num, min_index = data
        # Заменяем бесконечность на "∞" в строке
        formatted_row = [f"{d}" if d != float('inf') else "∞" for d in row]

        # Указываем элемент, в который пришли из предыдущего узла
        element = min_index + 1 if min_index is not None else "N/A"
        print(
            f"{' '.join(f'{d:<5}' for d in formatted_row)}\t{iteration_num}\t{row[min_index] if min_index is not None else 'N/A'}\t{element}")

        # Проверяем, является ли текущий минимальный элемент конечным узлом
        if min_index == end - 1:
            break  # Прекращаем вывод, если достигли конечного узла

    # Вывод кратчайшего пути от начального узла к конечному (Дейкстра)
    shortest_path = restore_path(path_dijkstra, start - 1, end - 1)
    if shortest_path:
        path_length = dist_dijkstra[end - 1]
        print(
            f"\nКратчайший путь от узла {start} до узла {end}: {' -> '.join(map(str, shortest_path))}, длина пути: {path_length}")
    else:
        print(f"\nПуть от узла {start} до узла {end} не найден")

    # Алгоритм Беллмана-Форда с выводом таблицы
    dist_bellman, iteration_data_bellman = bellman_ford_with_table(
        matrix, start - 1)

    # Вывод таблицы итераций Беллмана-Форда
    print("\nАлгоритм Беллмана-Форда:")
    print(f"{'':<5}", end="")
    for i in range(1, n + 1):
        print(f"{i:<5}", end="")
    print("№\tmin\telement")

    # Начинаем со 2-й итерации
    for idx, data in enumerate(iteration_data_bellman[1:], start=2):
        row, iteration_num, min_index = data
        # Заменяем 10000 на "∞" в строке
        formatted_row = [f"{d}" if d != 10000 else "∞" for d in row]

        # Указываем элемент, в который пришли из предыдущего узла
        element = min_index + 1 if min_index != -1 else "N/A"
        print(
            f"{' '.join(f'{d:<5}' for d in formatted_row)}\t{iteration_num}\t{row[min_index] if min_index != -1 else 'N/A'}\t{element}")

    # Вывод кратчайших путей от узла
    print("\nКратчайшие пути от узла", start, ":")
    for i in range(n):
        if i != start - 1:
            shortest_path = restore_path(path_dijkstra, start - 1, i)
            path_length = dist_dijkstra[i]
            if path_length == 10000:  # Проверка на отсутствие пути
                print(f"Путь до узла {i + 1}: не существует")
            elif shortest_path:
                print(
                    f"Путь до узла {i + 1}:  {' -> '.join(map(str, shortest_path))},  длина пути: {path_length}")
            else:
                print(f"Путь до узла {i + 1} не найден")


if __name__ == "__main__":
    main()
