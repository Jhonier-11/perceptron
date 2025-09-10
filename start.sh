#!/bin/bash

echo "========================================"
echo "   PERCEPTRON SIMPLE - INICIO RAPIDO"
echo "========================================"
echo

echo "[1/4] Verificando Python..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python no está instalado"
    exit 1
fi

echo
echo "[2/4] Instalando dependencias..."
pip3 install numpy pandas matplotlib openpyxl
if [ $? -ne 0 ]; then
    echo "ERROR: No se pudieron instalar las dependencias"
    exit 1
fi

echo
echo "[3/4] Configurando base de datos..."
python3 manage.py makemigrations
python3 manage.py migrate
if [ $? -ne 0 ]; then
    echo "ERROR: No se pudo configurar la base de datos"
    exit 1
fi

echo
echo "[4/4] Iniciando servidor..."
echo
echo "========================================"
echo "  Servidor iniciado en http://127.0.0.1:8000"
echo "  Presiona Ctrl+C para detener el servidor"
echo "========================================"
echo

python3 manage.py runserver
