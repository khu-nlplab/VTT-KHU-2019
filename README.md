 Video Turing Test 2019
=====
Question Level classification (QLC)
-----------------------------
입력 질문에 대한 난이도를 측정하는 모델입니다.
[Heo et al.(AAAI Spring Symposium)](https://arxiv.org/abs/1904.00623)에 정의되어 있는 Criteria 중 2가지(Memory, Logical Complexity)의 관점에서 난이도를 측정합니다. <br><br>
__How to Use__
1. ``Level_Classification/requirement.sh``을 실행시켜주세요
> ./requirement.sh

2. 다음 링크의 미리 학습된 모델을 받아주세요<br>
모델을 다운로드 받은 후 model 디렉토리를 만들어 model 디렉토리 안에 다운 받은 모델을 넣어 주세요. <br>
question level classification pre-trained model download link : <br>
>[Google drive link](https://drive.google.com/drive/folders/1AsK-xYOwN8x_rw05AyA3ZiIISGj4N2I2?usp=sharing)

3. ``Level_Classification/LevelClassificationModel.py``를 import하고 해당 모듈에 선언된 함수를 사용하시면 되겠습니다.
> 현재 LevelClassificationModel.py에는 미리 예시로 3개의 문장(question, clip description, scene description)이 들어가 있습니다. 

4. ``output``으로 다음 두가지가 출력됩니다.
> Memory Level : 1 <br>
> Logic Level : 2

Contact : Su-Hwan Yun (yunsh3432@khu.ac.kr)

Multiple Answer Selection (MAS)
------------------
다중의 응답을 입력으로 받아 가장 적절한 응답을 선택하는 모델입니다. <br><br>
__How to Use__
1. 다음 링크의 미리 학습된 모델과 데이터를 받아서 ``Answer_Selection/output_dir`` 과 ``Answer_Selection/data`` 디렉토리로 압축해제 해주세요.<br>
Answer selection classification pre-trained model and data download link: <br>
>[Google drive link](https://drive.google.com/open?id=1U2OV0ZFi-BtjJC-78DeZ0UgPpic0tzKG)

2. ``Answer_Selection/run_answer_selection.sh``를 실행하면 output이 출력됩니다.(예시포함)
>./run_answer_selection.sh

3. ``output``으로 다음 두가지가 출력됩니다.
>Answer Confidence : [x.xxxx  x.xxxx] <br>
>Selected Answer : blah blah ~.

Contact : Gyu-Min Park (pgm1219@khu.ac.kr)
