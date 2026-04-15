@echo off

ECHO 正在配置环境，请稍候...

REM --- 唯一需要修改的地方 (如果你的 Miniconda 不在默认位置) ---
REM 默认 Miniconda 路径使用了 %USERPROFILE% 变量，通常无需修改
SET MINICONDA_PATH=%USERPROFILE%\miniconda3

REM --- 以下部分无需修改 ---

REM 初始化 Conda 环境
call %MINICONDA_PATH%\Scripts\activate.bat

REM 检查 Conda 是否初始化成功
if %errorlevel% neq 0 (
    ECHO.
    ECHO 错误: 无法在以下路径找到 Miniconda:
    ECHO %MINICONDA_PATH%
    ECHO 请编辑此 .bat 文件，并修正 MINICONDA_PATH 变量。
    ECHO.
    pause
    exit /b
)

REM 激活环境
call conda activate E:\mycondaenv\rlcard

REM 自动切换到此 .bat 文件所在的目录
cd /d "%~dp0"

ECHO 环境已激活，正在启动...
ECHO 当前目录: %cd%
ECHO.

cmd /k