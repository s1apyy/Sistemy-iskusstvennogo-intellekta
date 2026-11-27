# -*- coding: utf-8 -*-
"""Генерация отчётов по ЛР1–ЛР4 в формате как IoT_1_lab.docx (без листингов кода)."""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, Twips
from docx.enum.style import WD_STYLE_TYPE
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "Отчёты"
OUT_DIR.mkdir(exist_ok=True)

STUDENT = "Салимзянов И.Ф."
GROUP = "ИВТИИбд-31"
TEACHER = "Хайруллин И.Д."
DISCIPLINE = "Системы искусственного интеллекта"
YEAR = "2026"
CITY = "Ульяновск"

# first line indent ~1.25 cm (как в примере: 450215 EMU ≈ 1.25 cm)
FIRST_LINE = Cm(1.25)


def set_run_font(run, size=14, bold=False, italic=False, name="Times New Roman"):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def add_para(
    doc,
    text="",
    *,
    align=WD_ALIGN_PARAGRAPH.JUSTIFY,
    bold=False,
    size=14,
    first_line=None,
    space_before=None,
    space_after=None,
    line_spacing=1.5,
):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    if first_line is not None:
        pf.first_line_indent = first_line
    else:
        pf.first_line_indent = Cm(0)
    if space_before is not None:
        pf.space_before = space_before
    else:
        pf.space_before = Pt(0)
    if space_after is not None:
        pf.space_after = space_after
    else:
        pf.space_after = Pt(0)
    if line_spacing == 1.5:
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    elif line_spacing == 1.0:
        pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    else:
        pf.line_spacing = line_spacing
    if text:
        run = p.add_run(text)
        set_run_font(run, size=size, bold=bold)
    return p


def add_heading_center(doc, text):
    return add_para(
        doc,
        text,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        bold=True,
        size=14,
        space_before=Pt(12),
        space_after=Pt(6),
        line_spacing=1.0,
    )


def add_body(doc, text):
    return add_para(
        doc,
        text,
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        first_line=FIRST_LINE,
        line_spacing=1.5,
    )


def add_empty(doc, n=1):
    for _ in range(n):
        add_para(doc, "", align=WD_ALIGN_PARAGRAPH.CENTER, line_spacing=1.0)


def add_figure(doc, image_path, caption, width_cm=14.0):
    """Вставляет рисунок по центру и подпись «Рисунок N – ...»."""
    image_path = Path(image_path)
    if not image_path.exists():
        add_body(doc, f"[Рисунок не найден: {image_path.name}]")
        return

    # подгоняем ширину, сохраняя пропорции, max width_cm
    with Image.open(image_path) as im:
        w, h = im.size
    max_w = Cm(width_cm)
    # также ограничим высоту ~10 см
    max_h = Cm(10)
    # Cm -> EMU later via python-docx; compute display size in cm
    aspect = h / w
    disp_w = width_cm
    disp_h = disp_w * aspect
    if disp_h > 10:
        disp_h = 10
        disp_w = disp_h / aspect

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(6)
    pf.space_after = Pt(3)
    pf.first_line_indent = Cm(0)
    run = p.add_run()
    run.add_picture(str(image_path), width=Cm(disp_w))

    cap = add_para(
        doc,
        caption,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        size=14,
        space_before=Pt(3),
        space_after=Pt(6),
        line_spacing=1.0,
    )
    return cap


def make_title_page(doc, lab_no, lab_title):
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.left_margin = Cm(3.0)
    sec.right_margin = Cm(1.5)
    sec.top_margin = Cm(2.0)
    sec.bottom_margin = Cm(2.0)

    header_lines = [
        "Министерство науки и высшего образования Российской Федерации",
        "Федеральное государственное бюджетное образовательное учреждение",
        "высшего образования",
        "«Ульяновский государственный технический университет»",
        "Кафедра «Вычислительная техника»",
        f"Дисциплина «{DISCIPLINE}»",
    ]
    for line in header_lines:
        add_para(doc, line, align=WD_ALIGN_PARAGRAPH.CENTER, line_spacing=1.0)

    add_empty(doc, 6)

    add_para(
        doc,
        f"Лабораторная работа № {lab_no}",
        align=WD_ALIGN_PARAGRAPH.CENTER,
        bold=True,
        line_spacing=1.5,
    )
    add_para(
        doc,
        f"({lab_title})",
        align=WD_ALIGN_PARAGRAPH.CENTER,
        bold=True,
        line_spacing=1.5,
    )

    add_empty(doc, 4)

    for line in [
        "Выполнил",
        f"Студент группы {GROUP}",
        STUDENT,
        "Проверил:",
        "преподаватель кафедры «ВТ»",
        TEACHER,
    ]:
        add_para(doc, line, align=WD_ALIGN_PARAGRAPH.RIGHT, line_spacing=1.5)

    add_empty(doc, 2)
    add_para(doc, CITY, align=WD_ALIGN_PARAGRAPH.CENTER, line_spacing=1.0)
    add_para(doc, YEAR, align=WD_ALIGN_PARAGRAPH.CENTER, line_spacing=1.0)


def build_lab1():
    doc = Document()
    make_title_page(doc, 1, "Классификация изображений с помощью глубокого обучения")

    add_heading_center(doc, "Постановка задачи")
    add_body(
        doc,
        "1. Необходимо разработать модель глубокого обучения с использованием библиотеки "
        "PyTorch для решения задачи классификации изображений.",
    )
    add_body(
        doc,
        "2. Датасет должен загружаться с диска. В качестве набора данных использован "
        "локальный датасет MiniShapes (классы: circle, square, triangle, cross, line), "
        "сохранённый в структуре папок train/test.",
    )
    add_body(
        doc,
        "3. Необходимо разработать средства визуализации результатов: графики обучения "
        "и примеры классификации с указанием истинного и предсказанного класса.",
    )

    add_heading_center(doc, "Реализация")
    add_body(
        doc,
        "Лабораторная работа посвящена построению свёрточной нейронной сети для "
        "классификации изображений. Данные подготовлены и сохранены на диске, после "
        "чего загружаются через ImageFolder.",
    )
    add_body(
        doc,
        "1. Датасет MiniShapes содержит 5 классов геометрических фигур. Для обучения "
        "использовано 1000 изображений, для теста — 250. Примеры исходных данных "
        "показаны на рисунке 1.",
    )
    add_figure(
        doc,
        ROOT / "ЛР1_классификация" / "results" / "dataset_samples.png",
        "Рисунок 1 – Примеры изображений датасета MiniShapes",
        width_cm=14,
    )
    add_body(
        doc,
        "2. Архитектура модели — простая CNN из трёх блоков Conv2d–ReLU–MaxPool2d "
        "и двух полносвязных слоёв с Dropout. Функция потерь — CrossEntropyLoss, "
        "оптимизатор — Adam (learning rate = 0.001).",
    )
    add_body(
        doc,
        "3. Обучение выполнялось в течение 5 эпох с размером батча 64. На каждой эпохе "
        "фиксировались loss и accuracy на обучающей и тестовой выборках.",
    )
    add_body(
        doc,
        "4. По итогам обучения на тестовой выборке достигнута точность около 0.864. "
        "Динамика обучения представлена на рисунке 2.",
    )
    add_figure(
        doc,
        ROOT / "ЛР1_классификация" / "results" / "training_curves.png",
        "Рисунок 2 – Графики loss и accuracy при обучении модели",
        width_cm=14,
    )
    add_body(
        doc,
        "5. Для визуализации результатов классификации построены примеры предсказаний "
        "модели: для каждого изображения указаны истинный класс, предсказанный класс "
        "и уверенность модели (рисунок 3).",
    )
    add_figure(
        doc,
        ROOT / "ЛР1_классификация" / "results" / "classification_examples.png",
        "Рисунок 3 – Примеры классификации (зелёный — верно, красный — ошибка)",
        width_cm=13,
    )
    add_body(
        doc,
        "Таким образом, реализована и обучена модель классификации изображений с "
        "загрузкой данных с диска и визуализацией результатов.",
    )

    add_heading_center(doc, "Тестирование")
    add_body(
        doc,
        "Тестирование проводилось на отложенной выборке MiniShapes/test. Проверялись "
        "сходимость функции потерь, рост accuracy и качество визуальных предсказаний. "
        "Итоговая accuracy на тесте составила 0.864. Ошибочные случаи в основном "
        "связаны с похожими классами (например, line и стороны square) при сильном "
        "шуме и поворотах.",
    )

    add_heading_center(doc, "Вывод")
    add_body(
        doc,
        "В ходе выполнения лабораторной работы разработана свёрточная нейронная сеть "
        "на PyTorch для классификации изображений. Датасет загружался с диска, "
        "обучение выполнено за 5 эпох, достигнута тестовая точность около 86%. "
        "Реализована визуализация кривых обучения и примеров классификации, что "
        "позволяет оценивать качество модели не только по метрикам, но и качественно.",
    )

    out = OUT_DIR / "ЛР1_Классификация.docx"
    doc.save(out)
    print("saved", out)


def build_lab2():
    doc = Document()
    make_title_page(doc, 2, "Аугментация данных для повышения точности классификации")

    add_heading_center(doc, "Постановка задачи")
    add_body(
        doc,
        "1. Необходимо использовать механизмы аугментации данных для задачи "
        "классификации изображений из первой лабораторной работы.",
    )
    add_body(
        doc,
        "2. Цель — повысить точность модели (или устойчивость к вариациям входных "
        "данных) по сравнению с обучением без аугментации.",
    )
    add_body(
        doc,
        "3. Следует сохранить ту же архитектуру сети, чтобы сравнение было корректным, "
        "и визуально показать примеры аугментированных изображений.",
    )

    add_heading_center(doc, "Реализация")
    add_body(
        doc,
        "В лабораторной работе использован тот же датасет MiniShapes и та же архитектура "
        "SimpleCNN, что и в ЛР1. Отличие состоит в преобразовании обучающей выборки.",
    )
    add_body(
        doc,
        "1. Для обучающей выборки применены следующие аугментации: RandomHorizontalFlip, "
        "RandomRotation, ColorJitter и RandomAffine (небольшие сдвиги). На тестовой "
        "выборке аугментация не применялась.",
    )
    add_body(
        doc,
        "2. Примеры аугментированных изображений показаны на рисунке 1. Видно, что "
        "одна и та же фигура может выглядеть по-разному за счёт поворота, сдвига "
        "и изменения яркости/контраста.",
    )
    add_figure(
        doc,
        ROOT / "ЛР2_аугментация" / "results" / "augmentation_examples.png",
        "Рисунок 1 – Примеры аугментации обучающих изображений",
        width_cm=14,
    )
    add_body(
        doc,
        "3. Обучение проводилось 12 эпох с теми же гиперпараметрами оптимизатора, "
        "что и в ЛР1. За счёт аугментации модель дольше «раскачивается» на train, "
        "но лучше обобщает на тесте.",
    )
    add_body(
        doc,
        "4. Итоговая accuracy на тесте составила около 0.920, тогда как в ЛР1 без "
        "аугментации — около 0.864. Сравнение динамики test accuracy показано "
        "на рисунке 2.",
    )
    add_figure(
        doc,
        ROOT / "ЛР2_аугментация" / "results" / "comparison.png",
        "Рисунок 2 – Сравнение test accuracy ЛР1 (без аугментации) и ЛР2 (с аугментацией)",
        width_cm=14,
    )
    add_body(
        doc,
        "Таким образом, аугментация позволила улучшить качество классификации "
        "примерно на 5–6 процентных пунктов при той же архитектуре модели.",
    )

    add_heading_center(doc, "Тестирование")
    add_body(
        doc,
        "Проверка выполнялась на той же тестовой выборке, что и в ЛР1. Контролировались "
        "графики loss/accuracy и итоговая метрика. Train accuracy при аугментации "
        "растёт медленнее (модель каждый раз видит «новые» варианты картинок), "
        "однако test accuracy выше и стабильнее, что говорит о снижении переобучения.",
    )

    add_heading_center(doc, "Вывод")
    add_body(
        doc,
        "В ходе лабораторной работы к задаче классификации из ЛР1 добавлены механизмы "
        "аугментации данных. Эксперимент показал рост тестовой точности с 0.864 до 0.920. "
        "Полученный результат подтверждает полезность аугментации даже на небольшом "
        "датасете и при относительно простой архитектуре сети.",
    )

    out = OUT_DIR / "ЛР2_Аугментация.docx"
    doc.save(out)
    print("saved", out)


def build_lab3():
    doc = Document()
    make_title_page(doc, 3, "Генерация текстов с помощью нейронной сети")

    add_heading_center(doc, "Постановка задачи")
    add_body(
        doc,
        "1. Необходимо разработать модель генерации текстов по выбранному датасету.",
    )
    add_body(
        doc,
        "2. В качестве корпуса использованы фрагменты произведений А. С. Пушкина "
        "(файл pushkin.txt). Модель должна уметь продолжать текст по заданному началу (seed).",
    )
    add_body(
        doc,
        "3. Требуется обучить модель и продемонстрировать примеры сгенерированных текстов, "
        "а также динамику функции потерь.",
    )

    add_heading_center(doc, "Реализация")
    add_body(
        doc,
        "Для решения задачи выбран подход character-level: сеть предсказывает следующий "
        "символ по предыдущей последовательности. Такой вариант проще в реализации "
        "и не требует внешнего токенизатора.",
    )
    add_body(
        doc,
        "1. Текст загружается с диска, строится словарь символов (char2idx / idx2char). "
        "Корпус нарезается на последовательности фиксированной длины (60 символов) "
        "с перекрытием.",
    )
    add_body(
        doc,
        "2. Архитектура модели: Embedding → LSTM (2 слоя, hidden=256) → Linear. "
        "Обучение выполнялось оптимизатором Adam, функция потерь — CrossEntropyLoss. "
        "Для устойчивости применялось обрезание градиентов.",
    )
    add_body(
        doc,
        "3. Обучение проводилось 40 эпох. График убывания функции потерь приведён "
        "на рисунке 1.",
    )
    add_figure(
        doc,
        ROOT / "ЛР3_генерация_текста" / "results" / "loss.png",
        "Рисунок 1 – Изменение loss при обучении LSTM",
        width_cm=12,
    )
    add_body(
        doc,
        "4. Генерация выполняется семплированием символов с температурой около 0.75–0.85. "
        "В качестве seed использовались фразы «Мой дядя», «Зима», «У лукоморья», "
        "«Я вас любил». Фрагменты полученных текстов показаны на рисунке 2.",
    )
    add_figure(
        doc,
        ROOT / "ЛР3_генерация_текста" / "results" / "generated_preview.png",
        "Рисунок 2 – Примеры генерации текста по заданному seed",
        width_cm=14,
    )
    add_body(
        doc,
        "5. Сгенерированный текст местами воспроизводит ритм и обороты исходного корпуса, "
        "однако встречаются искажения слов и «склейки» разных фрагментов. Для "
        "character-level LSTM на небольшом датасете это ожидаемо.",
    )

    add_heading_center(doc, "Тестирование")
    add_body(
        doc,
        "Проверка модели заключалась в генерации продолжений для нескольких seed-фраз "
        "и качественной оценке связности результата. Также контролировалось снижение "
        "loss на обучающих последовательностях. Модель устойчиво генерирует текст "
        "в «пушкинском» стиле, сохраняя кириллицу, знаки препинания и переносы строк.",
    )

    add_heading_center(doc, "Вывод")
    add_body(
        doc,
        "В лабораторной работе реализована и обучена модель генерации текста на основе "
        "LSTM. Датасет загружался с диска, обучение выполнено с нуля без предобученных "
        "языковых моделей. Полученные примеры показывают, что даже относительно простая "
        "character-level сеть способна формировать текст, похожий на исходный корпус, "
        "хотя полностью устранить ошибки генерации на малом объёме данных сложно.",
    )

    out = OUT_DIR / "ЛР3_Генерация_текста.docx"
    doc.save(out)
    print("saved", out)


def build_lab4():
    doc = Document()
    make_title_page(doc, 4, "Сегментация изображений с помощью U-Net")

    add_heading_center(doc, "Постановка задачи")
    add_body(
        doc,
        "1. Необходимо разработать модель сегментации изображений по выбранному датасету.",
    )
    add_body(
        doc,
        "2. Использован учебный датасет Shapes (круг, квадрат, треугольник и фон) "
        "в формате пар «изображение + маска», сохранённых на диске. Формат соответствует "
        "типовым датасетам сегментации.",
    )
    add_body(
        doc,
        "3. Требуется обучить модель, оценить качество по pixel accuracy и mean IoU, "
        "а также визуализировать предсказанные маски.",
    )

    add_heading_center(doc, "Реализация")
    add_body(
        doc,
        "В качестве архитектуры выбран упрощённый U-Net: encoder–decoder со skip-connections. "
        "Такая схема хорошо подходит для задач плотной разметки пикселей.",
    )
    add_body(
        doc,
        "1. Датасет сгенерирован и сохранён на диск: 400 пар для обучения и 80 для "
        "валидации, размер изображений 128×128. На одном кадре может быть несколько "
        "перекрывающихся фигур. Пример пары image/mask показан на рисунке 1.",
    )
    add_figure(
        doc,
        ROOT / "ЛР4_сегментация" / "results" / "dataset_sample.png",
        "Рисунок 1 – Пример изображения и соответствующей маски сегментации",
        width_cm=12,
    )
    add_body(
        doc,
        "2. Модель U-Net обучалась 8 эпох, функция потерь — CrossEntropyLoss с "
        "пониженным весом класса фона (фон занимает большую долю пикселей). "
        "Оптимизатор — Adam.",
    )
    add_body(
        doc,
        "3. В процессе обучения отслеживались train loss, pixel accuracy и mean IoU "
        "на валидации. Графики приведены на рисунке 2.",
    )
    add_figure(
        doc,
        ROOT / "ЛР4_сегментация" / "results" / "curves.png",
        "Рисунок 2 – Динамика loss, pixel accuracy и mean IoU",
        width_cm=14,
    )
    add_body(
        doc,
        "4. По итогам обучения на валидации получены высокие значения метрик "
        "(pixel accuracy ≈ 0.999, mIoU ≈ 0.998). Визуальное сравнение истинных "
        "и предсказанных масок показано на рисунке 3.",
    )
    add_figure(
        doc,
        ROOT / "ЛР4_сегментация" / "results" / "segmentation_examples.png",
        "Рисунок 3 – Примеры сегментации: исходное изображение, истинная маска, предсказание",
        width_cm=13,
    )
    add_body(
        doc,
        "Высокие метрики объясняются относительно простой структурой датасета "
        "(чёткие фигуры и ограниченное число классов). При этом пайплайн полностью "
        "соответствует реальной задаче сегментации: загрузка с диска, обучение U-Net "
        "и визуальный контроль результатов.",
    )

    add_heading_center(doc, "Тестирование")
    add_body(
        doc,
        "Тестирование выполнялось на валидационной выборке. Проверялись: сходимость "
        "обучения, рост mIoU, отсутствие грубых ошибок на визуализациях (сдвиг маски, "
        "перепутывание классов). Модель корректно выделяет круги, квадраты и треугольники "
        "даже при частичном перекрытии объектов.",
    )

    add_heading_center(doc, "Вывод")
    add_body(
        doc,
        "В лабораторной работе реализована модель сегментации изображений на основе U-Net. "
        "Данные загружались с диска в формате image/mask. Обучение показало высокую "
        "точность на учебном датасете Shapes, а визуализация подтвердила корректность "
        "предсказанных масок. Полученный опыт можно перенести на более сложные датасеты "
        "сегментации (например, Oxford-IIIT Pet или PASCAL VOC).",
    )

    out = OUT_DIR / "ЛР4_Сегментация.docx"
    doc.save(out)
    print("saved", out)


if __name__ == "__main__":
    build_lab1()
    build_lab2()
    build_lab3()
    build_lab4()
    print("ALL DONE ->", OUT_DIR)
