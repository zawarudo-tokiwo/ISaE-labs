import numpy as np
from scipy.optimize import minimize

from simplex import SimplexMatrix
from matrix_table import MatrixTable, print_table

np.set_printoptions(suppress=True)

# Var13
A = np.array([[7, 3, 4, 6, 2],
              [3, 2, 1, 5, 4],
              [5, 2, 3, 5, 6]])

B = np.array([130, 100, 90])

C = np.array([300, 240, 170, 90, 220])


def build_constraint(a, b):
    def f_constraint(x):
        return b - a @ x

    return f_constraint


def build_target_func(c):
    def f_target(x):
        return c @ x

    return f_target


def stringify_vector(v, round_precision=2):
    v_output = []

    for el in v.tolist():
        el = round(el, round_precision)

        if el == -0:
            el = 0

        v_output.append(str(el))

    return ', '.join(v_output)


def get_simplex_output(a, b, simplex_full_matrix):
    return MatrixTable(
        [str(i) for i in range(a.shape[0] + 1)],
        [f'x-{i}'.rjust(8, ' ')
         for i in range(a.shape[1] + b.shape[0])] + ['b'],
        simplex_full_matrix
    )


def main():
    x = np.zeros([C.shape[0]])

    f_revenue = build_target_func(C)
    f_constr_revenue = build_constraint(A, B)

    revenue_constraints = [
        {'type': 'ineq', 'fun': f_constr_revenue},
        {'type': 'ineq', 'fun': lambda x: x}
    ]

    x = minimize(lambda x: -1 * f_revenue(x), x,
                 constraints=revenue_constraints)['x']
    revenue = f_revenue(x)
    print('Рекомендации по производству продукции:', stringify_vector(x))
    print('Ожидаемая прибыль:', round(revenue))

    materials_needed = A @ x
    print('Необходимо материалов:')
    for mat, est in zip(materials_needed.tolist(), B.tolist()):
        print(f'{round(mat, 2)}/{est}')

    y = np.zeros([B.shape[0]])

    f_materials = build_target_func(B)
    f_constr_materials = build_constraint(A.transpose() * -1, C * -1)

    materials_constraints = [
        {'type': 'ineq', 'fun': f_constr_materials},
        {'type': 'ineq', 'fun': lambda y: y}
    ]

    print()

    y = minimize(f_materials, y, constraints=materials_constraints)['x']
    print('Ресурсы в дефиците (-> 0)/избыточные ресурсы (чем больше значение, тем выше дефицит):',
          stringify_vector(y), sep='\n')
    print('Ожидаемая прибыль:', round(f_materials(y)))

    optimal_resource_values = A.transpose() @ y
    print('Оптимальная стоимость ресурсов:')
    for val, current in zip(optimal_resource_values.tolist(), C.tolist()):
        print(f'{current} -> {round(val, 2)}')

    # Симплекс-метод
    print('\nСимплекс-метод - начало работы:')
    simplex = SimplexMatrix(A, C, B)
    print_table(get_simplex_output(A, B, simplex.get_full_matrix()), 2)
    simplex.apply_simplex(revenue)

    print('\nСимплекс-метод - результат:')
    print_table(get_simplex_output(A, B, simplex.get_full_matrix()), 2)


if __name__ == '__main__':
    main()
