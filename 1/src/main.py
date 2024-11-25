import numpy as np
import pygraphviz as pgv


class MatixAnalyzer():

    def __init__(self, file_path: str = "input.txt") -> None:
        self.a = self.read_adjacency_matix(file_path)
        self.a_powered = np.copy(self.a)
        self.b = np.copy(self.a)
        self.pow = 1

        self.To = [-1 for _ in range(1, len(self.a) + 1)]
        self.Tg = [-1 for _ in range(1, len(self.a) + 1)]
        self.Tx = [-1 for _ in range(1, len(self.a) + 1)]

        self.input_elements = self.calclulate_input_elements(self.a)
        self.output_elements = self.calclulate_output_elements(self.a)

        self.calculate_To()
        self.calculate_Tg()
        self.calculate_Tx()

    def read_adjacency_matix(self, file_path):
        file = open(file_path)
        matrix = []
        for line in file.readlines():
            tmp = []
            for item in line.split():
                tmp.append(int(item))
            matrix.append(tmp)

        matrix = np.array(matrix)

        return matrix

    def calclulate_output_elements(self, matrix):
        output_elements = []

        for i, row_sum in enumerate(matrix.sum(axis=1)):
            if row_sum == 0:
                output_elements.append(i + 1)

        return output_elements

    def calclulate_input_elements(self, matrix):
        input_elements = []
        for i, column_sum in enumerate(matrix.sum(axis=0)):
            if column_sum == 0:
                input_elements.append(i + 1)
                self.To[i] = 0

        return input_elements

    def calculate_To(self):
        while self.a_powered.sum() != 0:
            self.pow += 1
            self.a_powered = np.linalg.matrix_power(self.a, self.pow)
            for i, column_sum in enumerate(self.a_powered.sum(axis=0)):
                if column_sum == 0 and self.To[i] == -1:
                    self.To[i] = self.pow - 1
            self.b += self.a_powered

    def calculate_Tg(self):
        for i in self.output_elements:
            self.Tg[i-1] = self.To[i-1]

        for i, line in enumerate(self.a):
            _max = 0
            for item_index, item in enumerate(line):
                if item != 0:
                    x = item * self.To[item_index]
                    if x > _max:
                        _max = x
            if self.Tg[i] == -1:
                self.Tg[i] = int(_max)

    def calculate_Tx(self):
        for i in range(len(self.a)):
            self.Tx[i] = self.Tg[i] - self.To[i]

    def print_results(self):
        print("Output elements:", *self.output_elements)
        print("Input elements:", *self.input_elements)

        print("Order:", self.pow - 1)
        print("Formation tacts:", self.To)
        print("Quenching tacts:", self.Tg)
        print("Storage tacts:", self.Tx)
        print("Adjacency matrix for ordered graph:", self.b)

    # TODO: This could be better
    def show_ordered_graph(self, adjacency_matrix):
        dot = pgv.AGraph(strict=False, directed=True)
        rows, columns = adjacency_matrix.shape
        for i in range(rows):
            dot.add_node(f"{i+1}")

        for i in range(rows):
            for j in range(columns):
                if adjacency_matrix[i][j] == 1:
                    dot.add_edge(f"{i+1}", f"{j+1}")
        dot.draw("graphviz_output/graph.png", prog='dot')


matrix_analyzer = MatixAnalyzer("input2.txt")
matrix_analyzer.print_results()
matrix_analyzer.show_ordered_graph(matrix_analyzer.a)
