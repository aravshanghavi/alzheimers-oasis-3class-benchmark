@ECHO OFF
CLS
ECHO.
ECHO =================================================================
ECHO      Starting Comprehensive Alzheimer's Experiment Pipeline
ECHO =================================================================
ECHO.

REM Create a main directory for all outputs if it doesn't exist
IF NOT EXIST "experiment_outputs" MKDIR "experiment_outputs"

REM --- This script is now populated with your exact filenames. ---

REM ==================================
REM === RUN 1: ResNet50 + Standard CE
REM ==================================
ECHO.
ECHO [1/8] Starting: ResNet50 with Standard CE Loss...
SET SCRIPT_NAME=resnet_standard_ce.py
SET MODEL_FOLDER=resnet50_standard_ce
IF NOT EXIST "experiment_outputs\%MODEL_FOLDER%" MKDIR "experiment_outputs\%MODEL_FOLDER%"
ECHO [INFO] Running %SCRIPT_NAME%... Log will be saved to experiment_outputs\%MODEL_FOLDER%\results.txt
python %SCRIPT_NAME% > experiment_outputs\%MODEL_FOLDER%\results.txt 2>&1
ECHO [INFO] Moving generated figures...
MOVE /Y *.png experiment_outputs\%MODEL_FOLDER%\ > NUL
MOVE /Y *.jpg experiment_outputs\%MODEL_FOLDER%\ > NUL
ECHO [SUCCESS] Run 1 complete.

REM ==================================
REM === RUN 2: ResNet50 + Weighted CE
REM ==================================
ECHO.
ECHO [2/8] Starting: ResNet50 with Weighted CE Loss...
SET SCRIPT_NAME=resnet_weighted_ce.py
SET MODEL_FOLDER=resnet50_weighted_ce
IF NOT EXIST "experiment_outputs\%MODEL_FOLDER%" MKDIR "experiment_outputs\%MODEL_FOLDER%"
ECHO [INFO] Running %SCRIPT_NAME%... Log will be saved to experiment_outputs\%MODEL_FOLDER%\results.txt
python %SCRIPT_NAME% > experiment_outputs\%MODEL_FOLDER%\results.txt 2>&1
ECHO [INFO] Moving generated figures...
MOVE /Y *.png experiment_outputs\%MODEL_FOLDER%\ > NUL
MOVE /Y *.jpg experiment_outputs\%MODEL_FOLDER%\ > NUL
ECHO [SUCCESS] Run 2 complete.

REM ==================================
REM === RUN 3: ResNet50 + CBFL
REM ==================================
ECHO.
ECHO [3/8] Starting: ResNet50 with Custom Balanced Focal Loss...
SET SCRIPT_NAME=resnet_cbfl.py
SET MODEL_FOLDER=resnet50_cbfl
IF NOT EXIST "experiment_outputs\%MODEL_FOLDER%" MKDIR "experiment_outputs\%MODEL_FOLDER%"
ECHO [INFO] Running %SCRIPT_NAME%... Log will be saved to experiment_outputs\%MODEL_FOLDER%\results.txt
python %SCRIPT_NAME% > experiment_outputs\%MODEL_FOLDER%\results.txt 2>&1
ECHO [INFO] Moving generated figures...
MOVE /Y *.png experiment_outputs\%MODEL_FOLDER%\ > NUL
MOVE /Y *.jpg experiment_outputs\%MODEL_FOLDER%\ > NUL
ECHO [SUCCESS] Run 3 complete.

REM ==================================
REM === RUN 4: EfficientNet-B0 + Standard CE
REM ==================================
ECHO.
ECHO [4/8] Starting: EfficientNet-B0 with Standard CE Loss...
SET SCRIPT_NAME=efficientnetb0_standard_ce.py
SET MODEL_FOLDER=efficientnet_b0_ce
IF NOT EXIST "experiment_outputs\%MODEL_FOLDER%" MKDIR "experiment_outputs\%MODEL_FOLDER%"
ECHO [INFO] Running %SCRIPT_NAME%... Log will be saved to experiment_outputs\%MODEL_FOLDER%\results.txt
python %SCRIPT_NAME% > experiment_outputs\%MODEL_FOLDER%\results.txt 2>&1
ECHO [INFO] Moving generated figures...
MOVE /Y *.png experiment_outputs\%MODEL_FOLDER%\ > NUL
MOVE /Y *.jpg experiment_outputs\%MODEL_FOLDER%\ > NUL
ECHO [SUCCESS] Run 4 complete.

REM ==================================
REM === RUN 5: MobileNetV3-Large + Standard CE
REM ==================================
ECHO.
ECHO [5/8] Starting: MobileNetV3-Large with Standard CE Loss...
SET SCRIPT_NAME=mobilenetv3_large_standard_ce.py
SET MODEL_FOLDER=mobilenet_v3_ce
IF NOT EXIST "experiment_outputs\%MODEL_FOLDER%" MKDIR "experiment_outputs\%MODEL_FOLDER%"
ECHO [INFO] Running %SCRIPT_NAME%... Log will be saved to experiment_outputs\%MODEL_FOLDER%\results.txt
python %SCRIPT_NAME% > experiment_outputs\%MODEL_FOLDER%\results.txt 2>&1
ECHO [INFO] Moving generated figures...
MOVE /Y *.png experiment_outputs\%MODEL_FOLDER%\ > NUL
MOVE /Y *.jpg experiment_outputs\%MODEL_FOLDER%\ > NUL
ECHO [SUCCESS] Run 5 complete.

REM ==================================
REM === RUN 6: ViT + Standard CE
REM ==================================
ECHO.
ECHO [6/8] Starting: Vision Transformer with Standard CE Loss...
SET SCRIPT_NAME=vit_standard_ce.py
SET MODEL_FOLDER=vit_standard_ce
IF NOT EXIST "experiment_outputs\%MODEL_FOLDER%" MKDIR "experiment_outputs\%MODEL_FOLDER%"
ECHO [INFO] Running %SCRIPT_NAME%... Log will be saved to experiment_outputs\%MODEL_FOLDER%\results.txt
python %SCRIPT_NAME% > experiment_outputs\%MODEL_FOLDER%\results.txt 2>&1
ECHO [INFO] Moving generated figures...
MOVE /Y *.png experiment_outputs\%MODEL_FOLDER%\ > NUL
MOVE /Y *.jpg experiment_outputs\%MODEL_FOLDER%\ > NUL
ECHO [SUCCESS] Run 6 complete.

REM ==================================
REM === RUN 7: ViT + Weighted CE
REM ==================================
ECHO.
ECHO [7/8] Starting: Vision Transformer with Weighted CE Loss...
SET SCRIPT_NAME=vit_weighted_ce.py
SET MODEL_FOLDER=vit_weighted_ce
IF NOT EXIST "experiment_outputs\%MODEL_FOLDER%" MKDIR "experiment_outputs\%MODEL_FOLDER%"
ECHO [INFO] Running %SCRIPT_NAME%... Log will be saved to experiment_outputs\%MODEL_FOLDER%\results.txt
python %SCRIPT_NAME% > experiment_outputs\%MODEL_FOLDER%\results.txt 2>&1
ECHO [INFO] Moving generated figures...
MOVE /Y *.png experiment_outputs\%MODEL_FOLDER%\ > NUL
MOVE /Y *.jpg experiment_outputs\%MODEL_FOLDER%\ > NUL
ECHO [SUCCESS] Run 7 complete.

REM ==================================
REM === RUN 8: ViT + CBFL
REM ==================================
ECHO.
ECHO [8/8] Starting: Vision Transformer with Custom Balanced Focal Loss...
SET SCRIPT_NAME=vit_cbfl.py
SET MODEL_FOLDER=vit_cbfl
IF NOT EXIST "experiment_outputs\%MODEL_FOLDER%" MKDIR "experiment_outputs\%MODEL_FOLDER%"
ECHO [INFO] Running %SCRIPT_NAME%... Log will be saved to experiment_outputs\%MODEL_FOLDER%\results.txt
python %SCRIPT_NAME% > experiment_outputs\%MODEL_FOLDER%\results.txt 2>&1
ECHO [INFO] Moving generated figures...
MOVE /Y *.png experiment_outputs\%MODEL_FOLDER%\ > NUL
MOVE /Y *.jpg experiment_outputs\%MODEL_FOLDER%\ > NUL
ECHO [SUCCESS] Run 8 complete.


ECHO.
ECHO =================================================================
ECHO  All experiment runs have completed.
ECHO  Check the 'experiment_outputs' directory for all results.
ECHO =================================================================
ECHO.