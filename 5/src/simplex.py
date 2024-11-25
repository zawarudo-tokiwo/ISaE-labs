import numpy as np


class SimplexMatrix:
    def __init__(self, a_matrix, c_vector, constraints):
        if a_matrix.shape[1] != c_vector.shape[0]:
            raise ValueError(
                'Matrices of weights and target function coefficients do not correspond by shape')
        if a_matrix.shape[0] != constraints.shape[0]:
            raise ValueError(
                'Matrices of weights and constraints do not correspond by shape')

        add_values = np.zeros(
            [a_matrix.shape[0], a_matrix.shape[0]], dtype=float)
        np.fill_diagonal(add_values, 1)

        a_matrix_float = np.zeros(a_matrix.shape, dtype=float)

        for y in range(a_matrix.shape[0]):
            for x in range(a_matrix.shape[1]):
                a_matrix_float[y, x] = a_matrix[y, x]

        last_row = []
        for i in range(a_matrix.shape[1] + constraints.shape[0] + 1):
            last_row.append(-c_vector[i] if i < c_vector.shape[0] else 0)

        self.__a_matrix = a_matrix_float
        self.__add_values = add_values
        self.__constraints = constraints
        self.__last_row = np.array(last_row, dtype=float)

        # print(self.__a_matrix, "\n",
        #       self.__add_values, "\n",
        #       self.__constraints, "\n",
        #       self.__last_row, "\n")

    def get_allow_coords(self):
        allow_column = self.__get_allow_column_index()
        allow_row = self.__get_allow_row_index(allow_column)

        return (allow_row, allow_column)

    @property
    def size_x(self):
        return self.__a_matrix.shape[1] + self.__constraints.shape[0] + 1

    @property
    def size_y(self):
        return self.__a_matrix.shape[0] + 1

    def __parse_index(self, x, y):
        if not (0 <= y < self.size_y and 0 <= x < self.size_x):
            raise ValueError('Coordinates are out of range')

        a_matrix = self.__a_matrix
        constraints = self.__constraints

        if y >= a_matrix.shape[0]:
            # last row
            return x, self.__last_row

        if x < a_matrix.shape[1]:
            # within a_matrix
            return (y, x), a_matrix

        if x < a_matrix.shape[1] + constraints.shape[0]:
            # within add_values
            return (y, x - a_matrix.shape[1]), self.__add_values

        return y, constraints

    def __getitem__(self, item):
        y, x = item
        index, section = self.__parse_index(x, y)

        return section[index]

    def __setitem__(self, key, value):
        y, x = key
        index, section = self.__parse_index(x, y)

        section[index] = value

    def __get_allow_column_index(self):
        last_row = self.__last_row
        min_index = 0

        for i in range(1, len(last_row)):
            if last_row[i] < last_row[min_index]:
                min_index = i

        return min_index

    def __get_allow_row_index(self, allow_col_index):
        a_matrix = self.__a_matrix
        constraints = self.__constraints

        min_index = None
        min_value = None

        for i in range(0, constraints.shape[0]):

            if a_matrix[i, allow_col_index] < 0:
                continue

            value = constraints[i] / a_matrix[i, allow_col_index]

            if min_value is None or value < min_value:
                min_value = value
                min_index = i

        return min_index

    def get_full_matrix(self):
        result = []

        for y in range(self.size_y):
            row = []

            for x in range(self.size_x):
                row.append(self[y, x])

            result.append(row)

        return np.array(result)

    def transform(self):
        allow_y, allow_x = self.get_allow_coords()
        allow_element = self[allow_y, allow_x]

        size_y = self.size_y
        size_x = self.size_x

        # print(allow_element, allow_y, allow_x)

        old_values = np.zeros([size_y, size_x])

        for y in range(size_y):
            for x in range(size_x):
                old_values[y, x] = self[y, x]

        for y in range(size_y):
            for x in range(size_x):
                if y == allow_y or x == allow_x:
                    self[y, x] = old_values[y, x] / allow_element
                    continue

                self[y, x] = (old_values[y, x] * allow_element -
                              old_values[allow_y, x] * old_values[y, allow_x]) / allow_element

    def apply_simplex(self, target_value, precision=0):
        while round(self.__last_row[len(self.__last_row) - 1], precision) < round(target_value, precision):
            self.transform()
