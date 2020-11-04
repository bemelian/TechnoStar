import cv2
import os
import argparse

# Парсинг аргументов
args_parser = argparse.ArgumentParser()
args_parser.add_argument("-test", required=False, help="Path to the test image")
args_parser.add_argument("-template", required=False, help="Path to the template image")
args_parser.add_argument("-thresh", required=False, help="Min score of matching (0 - 1)")
args = vars(args_parser.parse_args())

# Загрузка и проверка на корректность изображений
test = cv2.imread(args["test"] if args["test"] else "images/example_1/test.jpg", cv2.IMREAD_COLOR)
if test is None:
    print("Test image is incorrect")
    exit(0)

template = cv2.imread(args["template"] if args["template"] else "images/example_1/template_0.jpg", cv2.IMREAD_COLOR)
if template is None:
    print("Template image is incorrect")
    exit(0)

# Перевод в grayscale для дальнейшей предобработки
test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Вычисление ядра размытия в зависимости от размера шаблона
h, w = template_gray.shape
size = int(max(3, h * 0.02, w * 0.02))
k_size = (size, size)

# Детекция граней с помощью оператора Канни
template_gray = cv2.Canny(template_gray, 25, 220)
# Размытие изображения шаблона
template_gray = cv2.blur(template_gray, k_size)

# Детекция граней с помощью оператора Канни
test_gray = cv2.Canny(test_gray, 25, 220)
# Размытие изображения тестового изображения
test_gray = cv2.blur(test_gray, k_size)

# Отображение предобратонных шаблона и тестового изображений (до нажатия клавиши)
cv2.imshow("TEST_IMAGE", test_gray)
cv2.imshow("TEMPLATE", template_gray)
cv2.waitKey()

# Посик шаблона
score = cv2.matchTemplate(test_gray, template_gray, cv2.TM_CCOEFF_NORMED)
_, val, _, loc = cv2.minMaxLoc(score)

thresh = float(args["thresh"]) if args["thresh"] else 0.7
if val > thresh:
    print(f"Template was found with confidence {val*100:.2f}")
    # Отрисовка ограничивающего квадрата шаблона на изображении в случае его наличия
    cv2.rectangle(test, loc, (loc[0] + w, loc[1] + h), (0,0,255), 2)

    # Вычисление параметров изменения размера
    fx = 800 / test_gray.shape[0] if test_gray.shape[0] > 800 else 1
    # Отображение шаблона и тестового изображения с обведенным шаблоном (в случае его наличия)
    cv2.imshow("TEST_IMAGE", cv2.resize(test, None, fx=fx, fy=fx))
    cv2.imshow("TEMPLATE", template)
    cv2.waitKey(10000)
else:
    print(f"Template wasn't found!")
