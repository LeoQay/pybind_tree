# Tree Regressor with special loss wrapped with pybind11

Реализация на языке C++ алгоритма решающего дерева с нестандартным функционалом качества.
Решение обернуто в динамическую библиотеку для языка python с помощью библиотеки pybind11.

[Постановка функционала и некоторые выкладки](forms.pdf).

__ВАЖНО__: для повышения производительности модуль внутри себя не копирует получаемые из python массивы (numpy array).
Поэтому в модуль необходимо передавать только "contigious" и "C style" массивы.

## Инструкция по сборке на linux

Лучше использовать python версии 3.10, так как запускалось с ней, но должно работать с любой версией 3.6+.

1. Установить библиотеку pybind11 с помощью pip
2. Запустить команду:
   
    python3 -m pybind11 --includes
   
    Она выведет директории с исходными файлами библиотеки pybind11.
    Вывод необходимо скопировать в Makefile, а именно приравнять переменной PYBIND11_INCLUDES.

    Пример вывода скрипта:

    -I/usr/include/python3.10 -I/home/leoqay/second/lib/python3.10/site-packages/pybind11/include

3. То же самое с командой:
    python3-config --extension-suffix

    Приравнять ее вывод переменной MODULE_NAME_SUFFIX в Makefile.

    Пример вывода скрипта:

    .cpython-310-x86_64-linux-gnu.so

4. Если все хорошо, то командой make в корне проекта скомпилируется необходимый модуль.
