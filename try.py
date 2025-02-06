import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import sympy as sp
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QFrame
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import mplcursors
import re

def format_expression(expr):
    """Форматирует введённое выражение, чтобы оно стало корректным для sympy"""
    expr = expr.replace(" ", "").replace("^", "**")  # Удаляем пробелы и заменяем ^ на **
    
    # Убираем "K(p) = ", если есть
    if "=" in expr:
        expr = expr.split("=")[-1]

    # Добавляем * между числом и переменной s (например, 3s -> 3*s)
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)

    # Проверим выражение на наличие некорректных частей
    expr = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', expr)  # Обрабатываем случай типа s2 -> s*2

    print(f"Отформатированное выражение: {expr}")  # Добавляем отладочную информацию
    return expr

def parse_transfer_function(expr):
    """Парсит введённую передаточную функцию и возвращает коэффициенты"""
    try:
        s = sp.Symbol('p')  # Объявляем переменную s
        expr = format_expression(expr)  # Форматируем выражение

        expr = sp.sympify(expr)  # Преобразуем текст в математическое выражение

        if not expr.is_rational_function(s):
            raise ValueError("Функция должна быть дробно-рациональной!")

        # Получаем числитель и знаменатель
        num, den = sp.fraction(expr)

        # Преобразуем в полином и получаем коэффициенты
        num_coeffs = sp.Poly(num, s).all_coeffs()
        den_coeffs = sp.Poly(den, s).all_coeffs()

        print(f"Числитель: {num_coeffs}, Знаменатель: {den_coeffs}")  # Добавляем отладочную информацию
        
        # Приводим числитель и знаменатель к одинаковой длине
        if len(num_coeffs) > len(den_coeffs):
            den_coeffs = [0] * (len(num_coeffs) - len(den_coeffs)) + den_coeffs
        elif len(den_coeffs) > len(num_coeffs):
            num_coeffs = [0] * (len(den_coeffs) - len(num_coeffs)) + num_coeffs

        print(f"Числитель после выравнивания: {num_coeffs}, Знаменатель после выравнивания: {den_coeffs}")  # Отладочная информация
        return [float(c) for c in num_coeffs], [float(c) for c in den_coeffs]

    except Exception as e:
        print(f"Ошибка при разборе передаточной функции: {e}")  # Добавляем отладочную информацию
        return None, None

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Анализ передаточной функции")
        self.setGeometry(100, 100, 900, 700)

        # Создание интерфейса
        self.layout = QVBoxLayout()

        self.frame_input = QHBoxLayout()
        self.label_func = QLabel("Введите передаточную функцию K(p):", self)
        self.frame_input.addWidget(self.label_func)

        self.entry_func = QLineEdit(self)
        self.frame_input.addWidget(self.entry_func)

        self.layout.addLayout(self.frame_input)

        self.btn_plot = QPushButton("Построить графики", self)
        self.btn_plot.clicked.connect(self.plot_transfer_function)
        self.layout.addWidget(self.btn_plot)

        # Кнопки для управления масштабом
        self.frame_zoom = QHBoxLayout()
        self.btn_zoom_in = QPushButton("Увеличить", self)
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        self.frame_zoom.addWidget(self.btn_zoom_in)

        self.btn_zoom_out = QPushButton("Уменьшить", self)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.frame_zoom.addWidget(self.btn_zoom_out)

        self.layout.addLayout(self.frame_zoom)

        # Рамка для графиков
        self.frame_graphs = QFrame(self)
        self.layout.addWidget(self.frame_graphs)

        self.setLayout(self.layout)

        self.current_axes = None
        self.canvas = None  # Добавим переменную для canvas

    def plot_transfer_function(self):
        ##########################################
        #Построение графиков передаточной функции#
        ##########################################
        expr = self.entry_func.text().strip()
        if not expr:
            print("Ошибка: Передаточная функция не введена.")  # Отладочная информация
            return

        print(f"Введенная передаточная функция: {expr}")  # Отладочная информация
        num, den = parse_transfer_function(expr)

        if num is None or den is None or len(den) == 0 or den[0] == 0:
            print("Ошибка: Невозможно построить график. Проверьте передаточную функцию.")  # Отладочная информация
            return
        ###############################
        # Создаём передаточную функцию#
        ###############################
        system = signal.TransferFunction(num, den)
        print("Передаточная функция создана.")  # Отладочная информация

        # Очистка предыдущих графиков
        if self.canvas:
            self.canvas.close()

        # Создание фигуры
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        fig.subplots_adjust(hspace=0.5, wspace=0.3)
        
        ################################
        # 1. АЧХ и ФЧХ (Bode-диаграмма)#
        ################################
        w, mag, phase = signal.bode(system)
        print("Графики Боде построены.")  # Отладочная информация

        axes[0, 0].semilogx(w, mag)
        axes[0, 0].set_title("АЧХ")
        axes[0, 0].set_xlabel("Частота (рад/с)")
        axes[0, 0].set_ylabel("Амплитуда (дБ)")
        axes[0, 0].grid()

        axes[0, 1].semilogx(w, phase)
        axes[0, 1].set_title("ФЧХ")
        axes[0, 1].set_xlabel("Частота (рад/с)")
        axes[0, 1].set_ylabel("Фаза (градусы)")
        axes[0, 1].grid()
        
        ####################################
        # 2. Импульсная характеристика (ИХ)#
        ####################################
        t_imp, h_imp = signal.impulse(system)
        print("График ИХ построен.")  # Отладочная информация
        axes[1, 0].plot(t_imp, h_imp)
        axes[1, 0].set_title("ИХ (Импульсная характеристика)")
        axes[1, 0].set_xlabel("Время (с)")
        axes[1, 0].set_ylabel("Амплитуда")
        axes[1, 0].grid()
        
        ####################################
        # 3. Переходная характеристика (ВХ)#
        ####################################
        t_step, h_step = signal.step(system)
        print("График ВХ построен.")  # Отладочная информация
        axes[1, 1].plot(t_step, h_step)
        axes[1, 1].set_title("ВХ (Переходная характеристика)")
        axes[1, 1].set_xlabel("Время (с)")
        axes[1, 1].set_ylabel("Амплитуда")
        axes[1, 1].grid()
        
        ###############################
        # Добавляем курсоры на графики#
        ###############################
        mplcursors.cursor(axes[0, 0], hover=True)
        mplcursors.cursor(axes[0, 1], hover=True)
        mplcursors.cursor(axes[1, 0], hover=True)
        mplcursors.cursor(axes[1, 1], hover=True)

        # Встраиваем график в PyQt
        self.canvas = FigureCanvas(fig)
        self.canvas.draw()
        self.canvas.setParent(self.frame_graphs)
        self.canvas.show()

    def zoom_in(self):
        """Функция для увеличения графиков"""
        if self.current_axes is not None and len(self.current_axes.flat) > 0:
            for ax in self.current_axes.flat:
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()

                ax.set_xlim(x_min * 0.9, x_max * 0.9)
                ax.set_ylim(y_min * 0.9, y_max * 0.9)

            self.frame_graphs.findChildren(FigureCanvas)[0].draw()

    def zoom_out(self):
        #Функция для уменьшения графиков
        ########################################
        #Надо переделать(не работает корректно)#
        ########################################
        if self.current_axes is not None and len(self.current_axes.flat) > 0:
            for ax in self.current_axes.flat:
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()

                ax.set_xlim(x_min * 1.1, x_max * 1.1)
                ax.set_ylim(y_min * 1.1, y_max * 1.1)

            self.frame_graphs.findChildren(FigureCanvas)[0].draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())