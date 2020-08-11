//****************************************************************************************
//****************************************************************************************
//---------------------------[[ 라벨링 검수를 위한 툴 ]]----------------------------------				 
//				 
//					이전에 라벨링을 한 이미지를 검수 하기 위해 사용 
//
//			- Usage: [path_to_images] [train.txt] [obj.names]
//
//			- 라벨링이 된 이미지를 확대하여 확인 후, 섬세한 크기 조절 가능
//			+ W -> 위 늘리기 , w -> 위 줄이기
//			+ A -> 왼쪽 늘리기 , a -> 왼쪽 줄이기
//			+ S -> 아래 늘리기 , s -> 아래 줄이기
//			+ D -> 오른족 늘리기 ,d -> 오른쪽 줄이기
//
//			- Object ID 변경 가능
//			+ 0 ~ 9 숫자 키로 Object ID 변경
//
//			- 방향키로 Image 변경, 현재 이미지에 있는 Label 변경
//			+ ↑ : Next Image 
//			+ ↓ : Previous Imgae
//			+ → : Next Label in present Image
//			+ ← : Previous Label in present Image
//****************************************************************************************
//****************************************************************************************

#define _CRC_SECURE_NO_WARNINGS
#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <stdarg.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <atomic>
#define percentage 1.5	// 라벨 주변을 보여줄 비율

using namespace cv;
using namespace std;


struct coord_t {
	cv::Rect_<float> abs_rect;
	int id;
};


int img_slider_max;				// 읽어온 이미지의 총갯수
int label_slider_max;			// 현재 이미지의 라벨 총갯수
int present_img_pos;			// 현재 이미지 번호
int present_label_pos;			// 현재 라벨 번호

string img_path[5000];			// img path
string txt_path;				// txt path
string Labeled = "Labeled";
string Labeled_Image = "Labeled Image";
string Original_Image = "Original Image";
Mat img;						// 원본이미지
Mat src;						// display 모습을 담는다.
Mat sub_img[1000];				// 각이미지의 sub img를 담는다.
Mat image_cloned;
static int sub_num;				// 한 이미지에 담겨있는 label의 총 갯수

vector <string> objname_vec;	// 오브젝트 이름을 담아두는 vector
vector <int> num_obj;			// 오브젝트 종류 수

char TrackbarName_label[50];			// label trackbar 이름
char TrackbarName_img[50];				// img trackbar 이름

vector<coord_t> Label_coord_in_img_vec;	// 한 이미지를 읽을때마다 라벨들의 값을 저장할곳.
vector<coord_t> Label_coord_in_sub_vec;	// sub img 내에서 라벨의 위치 - sub img에서는 라벨을 하나씩만 보여주기위해서
vector<coord_t> Rel_co_vec;				// rel의 크기를 조정하기 위한 vector

vector<int> label_N;					// 이미지에 라벨의 유무 검사

atomic<bool> exit_flag(false);			// switch 문에서 esc키를 이용한 종료 상태
atomic<bool> move_label_flag(false);	// switch 문에서 라벨을 다시 그려주어야 한다는 상태
atomic<bool> show_mark_class(true);		//switch 문에서 object id 보여주는 것을 결정하는 상태



// 미니맵창을 띄워 현재의 이미지를 보여준다.
static void  show_mini_map(Mat img, string str) {
	resize(img, img, Size(560, 400), 0, 0);
	imshow(str, img);
}

// 아랫쪽 트랙바 : sub img(labeled)를 선택함
static void on_trackbar_select(int, void*) {
	src = sub_img[present_label_pos];
	imshow(Labeled, src);
}

// 라벨순서에따라 색을 지정해서 네모칸을 그린다.
static void make_colored_label(int coord_id, Mat subimg[], Mat imgclone, int present_pos, coord_t i) {
	int r[255];
	int g[255];
	int b[255];

	r[coord_id] = 200 * (coord_id + 1) % 255;
	g[coord_id] = 150 * (coord_id + 1) % 255;
	b[coord_id] = 100 * (coord_id + 1) % 255;


	Scalar color_rect(r[coord_id], g[coord_id], b[coord_id]);
	for (int j = 0; j < num_obj.size(); j++) {
		if (coord_id == num_obj.at(j)) {
			rectangle(subimg[present_pos], Label_coord_in_sub_vec.at(present_pos).abs_rect, cv::Scalar(b[coord_id], g[coord_id], r[coord_id]), 1);
			rectangle(imgclone, Label_coord_in_img_vec.at(present_pos).abs_rect, cv::Scalar(b[coord_id], g[coord_id], r[coord_id]), 2);
			if (show_mark_class) {
				putText(imgclone, objname_vec.at(j), i.abs_rect.tl() + Point2f(2, 22), FONT_HERSHEY_SIMPLEX, 0.7, color_rect, 2);

			}
		}
	}
}

//상대 라벨위치에 따라 sub이미지에서의 라벨위치등을 알아낸다.
static coord_t manage_coord(float *relative_coord_0, float *relative_coord_1, float *relative_coord_2, float *relative_coord_3, Mat subimg[], coord_t coord, coord_t *Label_coord_in_sub, int present) {

	coord_t sub_img_coord;			// sub_img의 좌표가 담길 곳
	sub_img_coord.id = coord.id;

	// 라벨이 x축방향으로 img를 넘어가는것을 방지
	if (*relative_coord_0 - *relative_coord_2 / 2 < 0 || *relative_coord_0 + *relative_coord_2 / 2 > 1)
		*relative_coord_2 = min(*relative_coord_0, 1 - *relative_coord_0);

	// 라벨이 y축방향으로 img를 넘어가는것을 방지
	if (*relative_coord_1 - *relative_coord_3 / 2 < 0 || *relative_coord_1 + *relative_coord_3 / 2 > 1)
		*relative_coord_3 = min(*relative_coord_1, 1 - *relative_coord_1);

	// img에서 라벨이 쳐져있는 실제 범위
	coord.abs_rect.x = (*relative_coord_0 - *relative_coord_2 / 2) * (float)img.cols;
	coord.abs_rect.y = (*relative_coord_1 - *relative_coord_3 / 2) * (float)img.rows;
	coord.abs_rect.width = *relative_coord_2 *(float)img.cols;
	coord.abs_rect.height = *relative_coord_3 * (float)img.rows;

	// img에서 sub_img가 쳐져있는 범위
	{
		if (*relative_coord_0 - percentage * *relative_coord_2 / 2 >= 0)
			sub_img_coord.abs_rect.x = (*relative_coord_0 - percentage * *relative_coord_2 / 2)*(float)img.cols;
		else
			sub_img_coord.abs_rect.x = (float)0;

		if (*relative_coord_0 + percentage * *relative_coord_2 / 2 <= 1) {
			sub_img_coord.abs_rect.width = percentage * *relative_coord_2 * (float)img.cols;
			if (sub_img_coord.abs_rect.width >= img.cols)		// 비율확대시킨 박스가 원본이미지의 크기를 넘어갈 예외처리
				sub_img_coord.abs_rect.width = img.cols;
		}
		else
			sub_img_coord.abs_rect.width = img.cols - sub_img_coord.abs_rect.x;


		if (*relative_coord_1 - percentage * *relative_coord_3 / 2 >= 0) {
			sub_img_coord.abs_rect.y = (*relative_coord_1 - percentage * *relative_coord_3 / 2)*(float)img.rows;
		}
		else
			sub_img_coord.abs_rect.y = (float)0;

		if (*relative_coord_1 + percentage * *relative_coord_3 / 2 <= 1) {
			sub_img_coord.abs_rect.height = percentage * *relative_coord_3 * (float)img.rows;
			if (sub_img_coord.abs_rect.height >= img.rows)		// 비율확대시킨 박스가 원본이미지의 크기를 넘어갈 예외처리
				sub_img_coord.abs_rect.height = img.rows;
		}
		else
			sub_img_coord.abs_rect.height = img.rows - sub_img_coord.abs_rect.y;
	}

	Rect rect_130p(sub_img_coord.abs_rect.x, sub_img_coord.abs_rect.y, sub_img_coord.abs_rect.width, sub_img_coord.abs_rect.height);

	//sub img에서 라벨의 위치
	auto &i = coord;
	(*Label_coord_in_sub).abs_rect.x = i.abs_rect.x - sub_img_coord.abs_rect.x;
	(*Label_coord_in_sub).abs_rect.y = i.abs_rect.y - sub_img_coord.abs_rect.y;
	(*Label_coord_in_sub).abs_rect.width = i.abs_rect.width;
	(*Label_coord_in_sub).abs_rect.height = i.abs_rect.height;

	sub_img[present] = img(rect_130p).clone();	// 원본 이미지에 라벨주변의 직사각형 모양을 하나씩 sub img에 저장함.

	return coord;
}

// 윗쪽 트랙바 : img 를 선택함
static void on_trackbar_img(int, void*) {

	present_label_pos = 0;							// 새로운 이미지를 선택할때마다 첫번째 sub img를 갖고오게 함.
	txt_path = img_path[present_img_pos];
	txt_path.replace(txt_path.end() - 4, txt_path.end(), ".txt");	// img_path로부터 수정하여 txt_path만듦

	img = imread(img_path[present_img_pos]);		// 현재 이미지를 읽어옴 - 원본이미지
	Mat img_clone = img.clone();					// 트랙바 및 작업을 한 후에 라벨이 쳐져있게됨
	image_cloned = img_clone.clone();
	Mat img_label;									// label 되어있는 전체 양상

	ifstream readLabel;
	readLabel.open(txt_path);

	if (!readLabel.is_open()) {
		cout << "Label.txt is not exist : " << img_path[present_img_pos] << endl;
	}

	Label_coord_in_img_vec.clear();		// 이미지 하나를 읽을때마다 label값이 저장될 vector 초기화
	Label_coord_in_sub_vec.clear();
	Rel_co_vec.clear();
	label_N.clear();					// label이 있는 subimg인지 아닌지를 판별해줌

	std::vector<int> co_id;

	int k = 0;
	sub_num = 0;

	// 이미지에 있는 라벨들로 sub img와 label생성
	for (string line; getline(readLabel, line);) {
		std::stringstream ss(line);

		coord_t coord;				// 라벨 하나의 좌표가 들어감
		coord_t sub_img_coord;		// 라벨 하나의 130percent의 주변 이미지가 들어감
		coord_t Label_coord_in_sub;		// sub_img에서의 라벨의 위치
		coord_t rel_temp;			// Rel_co_vec에 넣을 값

		coord.id = -1;
		Label_coord_in_sub.id = -1;
		rel_temp.id = -1;

		ss >> coord.id;
		Label_coord_in_sub.id = coord.id;
		rel_temp.id = coord.id;

		if (coord.id < 0)
			continue;

		float relative_coord[4] = { -1, -1, -1, -1 };	// rel_center_x, rel_center_y, rel_width, rel_height

		for (size_t i = 0; i < 4; i++)
			if (!(ss >> relative_coord[i]))
				continue;

		for (size_t i = 0; i < 4; i++)
			if (relative_coord[i] < 0)
				continue;

		// 라벨의 실제 좌표를 계산해서 넣어준다.
		Label_coord_in_img_vec.push_back(manage_coord(&relative_coord[0], &relative_coord[1], &relative_coord[2], &relative_coord[3], sub_img, coord, &Label_coord_in_sub, k));
		Label_coord_in_sub_vec.push_back(Label_coord_in_sub);

		//수정하기 이전의 상대 라벨위치 넣어놓기
		rel_temp.abs_rect.x = relative_coord[0];
		rel_temp.abs_rect.y = relative_coord[1];
		rel_temp.abs_rect.width = relative_coord[2];
		rel_temp.abs_rect.height = relative_coord[3];

		Rel_co_vec.push_back(rel_temp);


		auto &i = Label_coord_in_img_vec.at(k);
		// 라벨 박스 그리기 - subimg, clone원본img 에
		make_colored_label(coord.id, sub_img, img_clone, k, i); //object 별로 라벨링 해줌

		label_N.push_back(0);

		sub_num++;
		k++;

	}

	readLabel.close();

	label_slider_max = sub_num;

	if (sub_num == 0) {
		sub_img[sub_num++] = Mat::zeros(Size(200, 200), CV_8UC3);		 //label 이 없다면 검은색 화면을 표시
		label_N.push_back(-1);
	}


	src = sub_img[present_label_pos];
	imshow(Labeled, src);

	sprintf(TrackbarName_label, "Labeled picture", sub_num);
	createTrackbar(TrackbarName_label, Labeled, &present_label_pos, sub_num - 1, on_trackbar_select);		// label trackbar
	on_trackbar_select(present_label_pos, 0);

	img_label = img_clone.clone();
	
	show_mini_map(img_label, Labeled_Image);		// 미니맵 보여주기 
}

// change 2 relative coord
static coord_t current_to_relative(coord_t current_coord) {
	current_coord.abs_rect.x = ((current_coord.abs_rect.x + (current_coord.abs_rect.x + current_coord.abs_rect.width)) / 2) / (float)img.cols;
	current_coord.abs_rect.y = ((current_coord.abs_rect.y + (current_coord.abs_rect.y + current_coord.abs_rect.height)) / 2) / (float)img.rows;
	current_coord.abs_rect.width = current_coord.abs_rect.width / (float)img.cols;
	current_coord.abs_rect.height = current_coord.abs_rect.height / (float)img.rows;

	return current_coord;
}

// img가 다음으로 넘어가면 수정된 라벨의 값을 저장한다.
static void save_coord() {
	ofstream writefile(txt_path.data());
	if (writefile.is_open()) {
		for (vector<coord_t>::iterator iter = Label_coord_in_img_vec.begin(); iter != Label_coord_in_img_vec.end(); iter++) {
			coord_t changed_coordvec = current_to_relative(*iter);

			cout.unsetf(ios::fixed);
			writefile << changed_coordvec.id << " ";
			writefile << fixed;
			writefile.precision(6);
			writefile << changed_coordvec.abs_rect.x << " " << changed_coordvec.abs_rect.y
				<< " " << changed_coordvec.abs_rect.width << " " << changed_coordvec.abs_rect.height << endl;
		}
	}

	if (present_img_pos == 0 || present_img_pos == img_slider_max)
		cout << "save : " << img_path[present_img_pos] << endl;
	else
		cout << "save : " << img_path[present_img_pos - 1] << endl;
}



int main(int argc, char *argv[]) {

	if (argc < 4) {
		cout << "Usage: [path_to_images] [train.txt] [obj.names] \n" << endl;
		return -1;
	}

	string images_path = string(argv[1]);			// path to images, train and synset
	string train_filename = string(argv[2]);		// file containing: list of images
	string synset_filename = string(argv[3]);		// file containing: object names


	ifstream readFile;
	readFile.open(train_filename);					//사전에 train.txt 가 만들어져 있어야한다.

	int index = 0;

	if (!readFile.is_open()) {
		cout << "Cannot read train.txt file." << endl;
		return -1;
	}
	while (!readFile.eof()) {
		getline(readFile, img_path[index]);
		index++;
	}
	img_slider_max = index - 1;


	namedWindow(Labeled, WINDOW_NORMAL); // Create Window


	ifstream objnameFile(synset_filename);	//obj 파일 받아옴
	int class_number = 0;
	string objname;
	while (!objnameFile.eof()) {
		getline(objnameFile, objname);
		class_number++;							//obj name 개수
		cout << "name :  " << objname << endl;
		objname_vec.push_back(objname);
	}

	for (int h = 0; h < class_number; h++) {
		num_obj.push_back(h);
	}

	sprintf(TrackbarName_img, "images %d", img_slider_max);
	createTrackbar(TrackbarName_img, Labeled, &present_img_pos, img_slider_max - 1, on_trackbar_img);	// img trackbar
	on_trackbar_img(present_img_pos, 0);


	do {
		setTrackbarPos(TrackbarName_label, Labeled, present_label_pos);	//labeled trackbar의 포지션을 set 시켜준다.
		setTrackbarPos(TrackbarName_img, Labeled, present_img_pos);		//img trackbar의 포지션을 set 시킨다.

		coord_t coord;				// 라벨 하나의 좌표가 들어감
		coord_t Lcoord;				// sub_img에서의 라벨의 위치

		coord.id = -1;
		Lcoord.id = coord.id;

		Mat img_clone = img.clone();


		// 키 입력을 받을 때 마다 스위치문에서 반복적으로 label의 크기조정
		if (move_label_flag) {
			move_label_flag = false;
			coord.id = Label_coord_in_img_vec.at(present_label_pos).id;

			Rel_co_vec.at(present_label_pos) = current_to_relative(Label_coord_in_img_vec.at(present_label_pos));

			Label_coord_in_img_vec.at(present_label_pos) =
				manage_coord(&Rel_co_vec.at(present_label_pos).abs_rect.x, &Rel_co_vec.at(present_label_pos).abs_rect.y,
					&Rel_co_vec.at(present_label_pos).abs_rect.width, &Rel_co_vec.at(present_label_pos).abs_rect.height,
					sub_img, coord, &Lcoord, present_label_pos);

			Label_coord_in_sub_vec.at(present_label_pos) = Lcoord;
			make_colored_label(coord.id, sub_img, img_clone, present_label_pos, Label_coord_in_img_vec.at(present_label_pos));


			on_trackbar_select(present_label_pos, 0);


			// 현재 이미지의 라벨을 모두 표시하고 보고 있는 라벨만 조정
			for (int i = 0; i < Label_coord_in_img_vec.size(); i++) {
				if (Label_coord_in_img_vec.at(i).abs_rect == Label_coord_in_img_vec.at(present_label_pos).abs_rect) {
					continue;
				}
				else {
					auto &n = Label_coord_in_img_vec.at(i);

					make_colored_label(Label_coord_in_img_vec.at(i).id, sub_img, img_clone, i, n);
				}
			}

			show_mini_map(img_clone, Labeled_Image);
		}


#ifndef CV_VERSION_EPOCH
		int pressed_key = cv::waitKeyEx(20); // OpenCV 3.x
#else
		int pressed_key = cv::waitKey(20);   // OpenCV 2.x
#endif
		if (pressed_key >= 0)
			for (int i = 0; i < 5; ++i)
				cv::waitKey(1);

		if (exit_flag)
			break;				 // exit after saving
		if (pressed_key == 27 || pressed_key == 1048603) {
			save_coord();
			cout << "bye" << endl;
			exit_flag = true;	 // break;  // ESC - save & exit
		}

		// 오브젝트 ID 변경하기 0~9 번 까지만 가능
		if (pressed_key >= '0' && pressed_key <= '9') {
			Label_coord_in_img_vec.at(present_label_pos).id = int(pressed_key - '0')+89; // 0 - 9
			move_label_flag = true;
		}

		switch (pressed_key) {
		case 'W':	//up increase
			if (label_N.at(present_label_pos) == -1)
				break;

			Label_coord_in_img_vec.at(present_label_pos).abs_rect.y -= 1.0;
			Label_coord_in_img_vec.at(present_label_pos).abs_rect.height += 1.0;

			if (Label_coord_in_img_vec.at(present_label_pos).abs_rect.y <= 0) {
				Label_coord_in_img_vec.at(present_label_pos).abs_rect.y += 1.0;
				Label_coord_in_img_vec.at(present_label_pos).abs_rect.height -= 1.0;
			}
			move_label_flag = true;
			break;

		case 'w':	//up decrease
			if (label_N.at(present_label_pos) == -1)
				break;

			Label_coord_in_img_vec.at(present_label_pos).abs_rect.y += 1.0;
			Label_coord_in_img_vec.at(present_label_pos).abs_rect.height -= 1.0;

			if (Label_coord_in_img_vec.at(present_label_pos).abs_rect.height <= 0.5) {
				Label_coord_in_img_vec.at(present_label_pos).abs_rect.y -= 1.0;
				Label_coord_in_img_vec.at(present_label_pos).abs_rect.height += 1.0;
			}
			move_label_flag = true;
			break;

		case 'S':	//down increase
			if (label_N.at(present_label_pos) == -1)
				break;

			Label_coord_in_img_vec.at(present_label_pos).abs_rect.height += 1.0;
			if (Label_coord_in_img_vec.at(present_label_pos).abs_rect.height + Label_coord_in_img_vec.at(present_label_pos).abs_rect.y >= img.rows) {
				Label_coord_in_img_vec.at(present_label_pos).abs_rect.height -= 1.0;
			}
			move_label_flag = true;
			break;

		case 's':	//down decrease
			if (label_N.at(present_label_pos) == -1)
				break;

			Label_coord_in_img_vec.at(present_label_pos).abs_rect.height -= 1.0;
			if (Label_coord_in_img_vec.at(present_label_pos).abs_rect.height <= 0) {
				Label_coord_in_img_vec.at(present_label_pos).abs_rect.height += 1.0;
			}
			move_label_flag = true;
			break;

		case 'A':	//left increase
			if (label_N.at(present_label_pos) == -1)
				break;

			Label_coord_in_img_vec.at(present_label_pos).abs_rect.x -= 1.0;
			Label_coord_in_img_vec.at(present_label_pos).abs_rect.width += 1.0;
			if (Label_coord_in_img_vec.at(present_label_pos).abs_rect.x < 0) {
				Label_coord_in_img_vec.at(present_label_pos).abs_rect.x += 1.0;
				Label_coord_in_img_vec.at(present_label_pos).abs_rect.width -= 1.0;
			}
			move_label_flag = true;
			break;

		case 'a':	//left decrease
			if (label_N.at(present_label_pos) == -1)
				break;

			Label_coord_in_img_vec.at(present_label_pos).abs_rect.x += 1.0;
			Label_coord_in_img_vec.at(present_label_pos).abs_rect.width -= 1.0;
			if (Label_coord_in_img_vec.at(present_label_pos).abs_rect.width < 0) {
				Label_coord_in_img_vec.at(present_label_pos).abs_rect.x -= 1.0;
				Label_coord_in_img_vec.at(present_label_pos).abs_rect.width += 0.0;
			}
			move_label_flag = true;
			break;

		case 'D':	//right increase
			if (label_N.at(present_label_pos) == -1)
				break;

			Label_coord_in_img_vec.at(present_label_pos).abs_rect.width += 1.0;
			if (Label_coord_in_img_vec.at(present_label_pos).abs_rect.width + Label_coord_in_img_vec.at(present_label_pos).abs_rect.x >= img.cols) {
				Label_coord_in_img_vec.at(present_label_pos).abs_rect.width -= 1.0;
			}
			move_label_flag = true;
			break;

		case 'd':	//right decrease
			if (label_N.at(present_label_pos) == -1)
				break;

			Label_coord_in_img_vec.at(present_label_pos).abs_rect.width -= 1.0;
			if (Label_coord_in_img_vec.at(present_label_pos).abs_rect.width <= 0.5) {
				Label_coord_in_img_vec.at(present_label_pos).abs_rect.width += 1.0;
			}

			move_label_flag = true;
			break;

		case 'k':
		case 1048683:
			show_mark_class = !show_mark_class;
			break;

		case 2490368:   // 위 - 다음 img
			++present_img_pos;
			if (present_img_pos >= img_slider_max || present_img_pos < 0) {				// trackbar가 img개수를 넘지 못하게 예외처리
				present_img_pos = min(max(0, present_img_pos), img_slider_max - 1);
			}
			save_coord();
			on_trackbar_img(present_img_pos, 0);
			cout << "present img : " << img_path[present_img_pos] << endl;
			break;

		case 2621440:   // 아래 - 이전 img
			--present_img_pos;
			if (present_img_pos >= img_slider_max || present_img_pos < 0) {
				present_img_pos = min(max(0, present_img_pos), img_slider_max - 1);
			}
			save_coord();
			on_trackbar_img(present_img_pos, 0);
			cout << "present img : " << img_path[present_img_pos] << endl;
			break;


		case 2424832: // <-
		case 65361:   // <-
			--present_label_pos;
			if (present_label_pos >= sub_num || present_label_pos < 0) {				// trackbar가 label개수를 넘지 못하게 예외처리
				present_label_pos = min(max(0, present_label_pos), sub_num - 1);
			}
			on_trackbar_select(present_label_pos, 0);
			if (label_N.at(present_label_pos) == -1)
				break;
			move_label_flag = true;
			break;

		case 2555904: // ->
		case 65363:   // ->
			++present_label_pos;
			if (present_label_pos >= sub_num || present_label_pos < 0) {
				present_label_pos = min(max(0, present_label_pos), sub_num - 1);
			}
			on_trackbar_select(present_label_pos, 0);

			if (label_N.at(present_label_pos) == -1)
				break;
			move_label_flag = true;
			break;

		default:;
		}
	} while (true);
	cout << "finished" << endl;
	return 0;
}