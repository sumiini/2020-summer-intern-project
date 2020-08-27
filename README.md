# 주석화 도구
* [AnnotationTool](#annotationtool)
* [VerificationTool](#verificationtool)

# AnnotationTool

## 기존 Yolo_Mark 기능 + 신규 기능 다수 추가
- 'q' 클릭시 해당 위치 박스 복제
- 'g' 기능 삭제
- 'F1-F12' 키로 인덱스 10부터 21까지 변경 가능

   * +키로 +1, -키로 -1하며 조정 가능, 다만 최대 인덱스를 초과시 마스킹(0)으로 리턴, -로 0까지만 인덱스 감소 가능 
<br/>

- 'Backspace' 키로 실행 취소(UNDO) 기능 : 방금 생성한 박스 삭제 기능
- 'Page Up/ Down' 키로 마우스 커서 올릴 시 박스 이미지 확대 On/Off

  - 기본 상태 : **Off** / Page Up 키 : 활성화 / Page Down 키 : **_비_** 활성화 (확대 창이 떠있는 경우 확대 창 자동 종료)
<br/>

- Multi-labeling 기능 개선: 같은 위치에 2개 이상의 label이 존재시 겹치는 obj name들을 한개씩 수직 출력
<br/> 

- 박스 내 박스 선택 가능: 마우스 커서 올려놓기로 박스 내 박스 선택 가능

- 'Ctrl' + 마우스 스크롤 휠로 이미지 확대  가능
  - 확대 후 'Ctrl' 누른 채로 마우스 움직여서 화면 이동 가능

<br/>

- 'k' 키로 인덱스 숨기기 그리고 제일 얇은 선으로 자동교체

- '/' 키로 현재 화면 모든 박스 삭제

- 'delete' 키로 모든 인덱스 및 박스 숨기기



## 기존 기능 + 추가 기능 Keyboard Shortcuts (UPDATED)
Shortcut | Description | 
--- | --- |
<kbd>→</kbd> | Next image |
<kbd>←</kbd> | Previous image |
<kbd>r</kbd> | Delete selected box (mouse hovered) |
<kbd>p</kbd> | Copy previous mark |
<kbd>o</kbd> | Track objects |
<kbd>ESC</kbd> | Close application |
<kbd>n</kbd> | One object per image |
<kbd>0-9</kbd> | Object id |
<kbd>m</kbd> | Show coords |
<kbd>w</kbd> | Line width |
<kbd>k</kbd> | Hide object name, Change into the thinnest line width |
<kbd>h</kbd> | Help |
<kbd>/</kbd> | Clear all boxes on the current image |
<kbd>delete</kbd> | Hide all boxes & marks on the current image |
<kbd>F1-F12</kbd> | Change Index (10-21) |
<kbd>Backspace←</kbd> | Undo (Delete the box generated lately) |
<kbd>Page Up</kbd>,<kbd>Page Down</kbd> | Box enlargement ON / OFF |
<kbd> Ctrl +  Mouse Scroll</kbd> | Image enlargement |
<kbd> Enter</kbd> | Next image |
## How to compile


#### Requirements
- Windows
- CMake >= 3.12: https://cmake.org/download/
- OpenCV >= 4.3: https://opencv.org/releases/
- Make msvc solution
```shell
mkdir build
cd build
cmake -G "Visual Studio 15 2017 Win64" -DINCLUDE_DIR_OPENCV=[OPENCV INCLUDE DIR] -DLIBRARY_DIR_OPENCV=[OPENCV LIBRARY DIR]  ..
cmake --build . --config "Release"-j;
```

## How to use
- yolo_mark.exe 경로에 data(img, train.txt, obj.names)폴더, cudnn64_7.dll, opencv_worldxxx.dll, tbb.dll, cuda관련 dll 복붙 
```shell
.\bin\...\yolo_mark.exe [video file] .\data\img .\data\train.txt .\data\obj.names
```

---------

# VerificationTool
#### 라벨링 컨펌 및 검수를 위한 툴 

##### 주요 기능 설명
- 라벨링 표시 및 비율 맞추기

  - 검수툴 제작 상단 트랙바 2개 (1st: 해당 이미지 박스 개수 표시, 2nd: 총 이미지수, 현 이미지 인덱스 개수 표시)
  
  - 라벨링된 이미지 확인을 위해, 화면에서 보는 이미지는 일정 비율의 사이즈로 라벨링된 이미지만 보이게 한다.
- 단축키 적용

  - 숫자키로 인덱스 변환 가능

  - 라벨링되어있는 부분의 크기를 조절, 재조정된 라벨링 위치 값 저장.

- 이미지 미니맵 띄워서 현재 이미지 확인
