@echo off
echo ========================================
echo    PERCEPTRON SIMPLE - INICIO RAPIDO
echo ========================================
echo.

echo [1/4] Verificando Python...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python no esta instalado
    pause
    exit /b 1
)

echo.
echo [2/4] Instalando dependencias...
pip install numpy pandas matplotlib openpyxl
if %errorlevel% neq 0 (
    echo ERROR: No se pudieron instalar las dependencias
    pause
    exit /b 1
)

echo.
echo [3/4] Configurando base de datos...
python manage.py makemigrations
python manage.py migrate
if %errorlevel% neq 0 (
    echo ERROR: No se pudo configurar la base de datos
    pause
    exit /b 1
)

echo.
echo [4/4] Iniciando servidor...
echo.
echo ========================================
echo   Servidor iniciado en http://127.0.0.1:8000
echo   Presiona Ctrl+C para detener el servidor
echo ========================================
echo.

python manage.py runserver
