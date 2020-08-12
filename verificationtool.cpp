//****************************************************************************************
//****************************************************************************************
//---------------------------[[ 라벨링 검수를 위한 툴 ]]----------------------------------				 
//				 
//					이전에 라벨링을 한 이미지를 검수 하기 위해 사용 
//
//			- Usage: [path_to_images] [train.txt] [obj.names]
//
//			- 라벨 검수 기능 - Object ID 변경
//			+ 1 : index 91번 - re_label	(라벨링을 더해야할 이미지)
//			+ 2 : index 92번 - re_index	(인덱스 번호가 잘못되어있는 라벨)
//			+ 3 : index 93번 - re_size	(사이즈를 재조정해야할 라벨)
//			+ 4 : index 94번 - remove	(제거해야할 라벨링)
//				=> 검수 후 수정이 필요한 이미지는 img_re라는 디렉토리에
//				   텍스트 파일과 함꼐 저장된다.
//			
//			- 방향키로 Image 변경, 현재 이미지에 있는 Label 변경
//			+ ↑ : Next Image 
//			+ ↓ : Previous Imgae
//			+ → : Next Label in present Image
//			+ ← : Previous Label in present Image
//
//			- undo 기능
//			+ z : 다른이미지로 넘어가기전에(저장되기 이전에)
//				  잘못 검수한 라벨을 원래상태로 되돌릴 수 있다.
//				  단, 91번의 경우 다른이미지로 넘어갔다가 돌아와도 undo 기능 작동.
//
//****************************************************************************************
//****************************************************************************************


#define _CRC_SECURE_NO_WARNINGS
#include <algorithm>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
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

string img_path[10000];			// img path
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
vector<int> id4undo;					// 어떤 id로 undo할지 사전에 저장해두는 vector
vector <coord_t> sort_vec;				// multi index를 나란히 보기위해 피요한 sorting vector


atomic<bool> exit_flag(false);			// switch 문에서 esc키를 이용한 종료 상태
atomic<bool> show_mark_class(true);		// switch 문에서 object id 보여주는 것을 결정하는 상태
atomic<bool> change_index_flag(false);	// index의 수정사항이 있는상태 저장할때 
atomic<bool> directory(false);			// directory 를 옮겨서 해야하는지의 상태
atomic<bool> redirectory(false);		// directory 수정한것을 undo해서 directory를 다시 옮겨야할때 사용


// 좌표 순으로 sorting하기
bool compare_xy(const coord_t &a, const coord_t &b) {
	if (a.abs_rect.y < 0.5 && b.abs_rect.y < 0.5)
		return a.abs_rect.x < b.abs_rect.x;
	if (a.abs_rect.y > 0.5 && b.abs_rect.y > 0.5)
		return a.abs_rect.x < b.abs_rect.x;

	return a.abs_rect.y < b.abs_rect.y;
}

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

	int offset = coord_id * 30;
	int red = (offset + 40) % 255 * ((coord_id + 2) % 3);
	int green = (offset + 190) % 255 * ((coord_id + 1) % 3);
	int blue = (offset + 100) % 255 * ((coord_id + 0) % 3);

	r[coord_id] = red;
	g[coord_id] = green;
	b[coord_id] = blue;


	Scalar color_rect(r[coord_id], g[coord_id], b[coord_id]);
	for (int j = 0; j < num_obj.size(); j++) {
		if (coord_id == num_obj.at(j)) {

			rectangle(subimg[present_pos], Label_coord_in_sub_vec.at(present_pos).abs_rect, color_rect, 1); //하나
			rectangle(imgclone, Label_coord_in_img_vec.at(present_pos).abs_rect, color_rect, 2); // 여러
			if (present_pos == present_label_pos) {
				rectangle(imgclone, Label_coord_in_img_vec.at(present_pos).abs_rect, cv::Scalar(255, 0, 255), 2);
				putText(subimg[present_label_pos], objname_vec.at(j), (Label_coord_in_sub_vec.at(present_label_pos).abs_rect.tl() + Point2f(2 / percentage, 22 / percentage)), FONT_HERSHEY_SIMPLEX, 0.3, color_rect, 1.8);

			}

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
		*relative_coord_2 = min((*relative_coord_0) * 2, (1 - *relative_coord_0) * 2);

	// 라벨이 y축방향으로 img를 넘어가는것을 방지
	if (*relative_coord_1 - *relative_coord_3 / 2 < 0 || *relative_coord_1 + *relative_coord_3 / 2 > 1)
		*relative_coord_3 = min((*relative_coord_1) * 2, (1 - *relative_coord_1) * 2);

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
				sub_img_coord.abs_rect.width = (float)img.cols;
		}
		else
			sub_img_coord.abs_rect.width = (float)img.cols - (float)sub_img_coord.abs_rect.x;


		if (*relative_coord_1 - percentage * *relative_coord_3 / 2 >= 0) {
			sub_img_coord.abs_rect.y = (*relative_coord_1 - percentage * *relative_coord_3 / 2)*(float)img.rows;
		}
		else
			sub_img_coord.abs_rect.y = (float)0;

		if (*relative_coord_1 + percentage * *relative_coord_3 / 2 <= 1) {
			sub_img_coord.abs_rect.height = percentage * *relative_coord_3 * (float)img.rows;
			if (sub_img_coord.abs_rect.height >= img.rows)		// 비율확대시킨 박스가 원본이미지의 크기를 넘어갈 예외처리
				sub_img_coord.abs_rect.height = (float)img.rows;
		}
		else
			sub_img_coord.abs_rect.height = (float)img.rows - (float)sub_img_coord.abs_rect.y;
	}

	Rect rect_130p(sub_img_coord.abs_rect.x, sub_img_coord.abs_rect.y, sub_img_coord.abs_rect.width, sub_img_coord.abs_rect.height);

	//sub img에서 라벨의 위치
	auto &i = coord;
	(*Label_coord_in_sub).abs_rect.x = (float)i.abs_rect.x - (float)sub_img_coord.abs_rect.x;
	(*Label_coord_in_sub).abs_rect.y = (float)i.abs_rect.y - (float)sub_img_coord.abs_rect.y;
	(*Label_coord_in_sub).abs_rect.width = (float)i.abs_rect.width;
	(*Label_coord_in_sub).abs_rect.height = (float)i.abs_rect.height;

	sub_img[present] = img(rect_130p).clone();	// 원본 이미지에 라벨주변의 직사각형 모양을 하나씩 sub img에 저장함.

	return coord;
}

// change 2 relative coord
static coord_t current_to_relative(coord_t current_coord) {
	current_coord.abs_rect.x = ((current_coord.abs_rect.x + (current_coord.abs_rect.x + current_coord.abs_rect.width)) / 2) / (float)img.cols;
	current_coord.abs_rect.y = ((current_coord.abs_rect.y + (current_coord.abs_rect.y + current_coord.abs_rect.height)) / 2) / (float)img.rows;
	current_coord.abs_rect.width = current_coord.abs_rect.width / (float)img.cols;
	current_coord.abs_rect.height = current_coord.abs_rect.height / (float)img.rows;
	return current_coord;
}

string new_jpg_path;
string old_jpg_path;

// img가 다음으로 넘어가면 수정된 라벨의 값을 저장한다.
static void save_coord() {

	string new_txt_path;
	string old_txt_path;

	old_txt_path = txt_path;
	new_txt_path = txt_path;

	// directory를 옮길때
	if (directory == true) {

		string temp_txt;
		string temp;

		temp = txt_path;

		new_txt_path = txt_path.replace((int)(txt_path.find_last_of("\\")) + 1, (int)(txt_path.find_last_of("/")) - (int)(txt_path.find_last_of("\\")) - 1, "img_re");

		temp_txt = new_txt_path;

		old_jpg_path = temp.replace(temp.end() - 4, temp.end(), ".jpg");
		new_jpg_path = temp_txt.replace(temp_txt.end() - 4, temp_txt.end(), ".jpg");

		string mv = "move ";
		mv += old_jpg_path;
		mv += " ";
		mv += new_jpg_path;

		replace(mv.begin(), mv.end(), '/', '\\');

		cout << mv << endl;

		system(mv.c_str());
		remove(old_txt_path.c_str());
	}


	//옮긴 directory를 다시 원래대로 해놓을때 - z를 눌렀을때 아무것도 라벨이 없는경우만
	if (redirectory == true) {

		string temp_txt;
		string temp;

		temp = txt_path;
		//cout << temp << endl;

		new_txt_path = txt_path.replace((int)(txt_path.find_last_of("\\")) + 1, (int)(txt_path.find_last_of("/")) - (int)(txt_path.find_last_of("\\")) - 1, "img");

		temp_txt = new_txt_path;

		old_jpg_path = temp.replace(temp.end() - 4, temp.end(), ".jpg");
		new_jpg_path = temp_txt.replace(temp_txt.end() - 4, temp_txt.end(), ".jpg");

		string mv = "move ";
		mv += old_jpg_path;
		mv += " ";
		mv += new_jpg_path;

		replace(mv.begin(), mv.end(), '/', '\\');

		cout << mv << endl;

		system(mv.c_str());
		remove(old_txt_path.c_str());

	}


	ofstream writefile(new_txt_path.data());
	if (writefile.is_open()) {
		for (vector<coord_t>::iterator iter = Label_coord_in_img_vec.begin(); iter != Label_coord_in_img_vec.end(); iter++) {
			coord_t curcoordvec = current_to_relative(*iter);

			cout.unsetf(ios::fixed);
			writefile << curcoordvec.id << " ";
			writefile << fixed;
			writefile.precision(6);
			writefile << curcoordvec.abs_rect.x << " " << curcoordvec.abs_rect.y
				<< " " << curcoordvec.abs_rect.width << " " << curcoordvec.abs_rect.height << endl;
		}
	}
	writefile.close();
	cout << "saved" << endl;
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

	ifstream readLabel_f;
	readLabel_f.open(txt_path);

	if (!readLabel_f.is_open()) {
		cout << "Label.txt is not exist : " << img_path[present_img_pos] << endl;
		return;
	}

	sort_vec.clear();

	for (string line; getline(readLabel_f, line);) {
		coord_t temp;
		stringstream ss(line);

		ss >> temp.id;
		ss >> temp.abs_rect.x;
		ss >> temp.abs_rect.y;
		ss >> temp.abs_rect.width;
		ss >> temp.abs_rect.height;

		sort_vec.push_back(temp);
	}

	sort(sort_vec.begin(), sort_vec.end(), compare_xy);


	ofstream writefile(txt_path.data());

	if (writefile.is_open()) {
		for (vector<coord_t>::iterator iter = sort_vec.begin(); iter != sort_vec.end(); iter++) {
			cout.unsetf(ios::fixed);
			writefile << (*iter).id << " ";
			writefile << fixed;
			writefile.precision(6);
			writefile << (*iter).abs_rect.x << " " << (*iter).abs_rect.y
				<< " " << (*iter).abs_rect.width << " " << (*iter).abs_rect.height << endl;
		}
	}
	readLabel_f.close();
	writefile.close();

	ifstream readLabel;
	readLabel.open(txt_path);

	if (!readLabel.is_open()) {
		cout << "Label.txt is not exist : " << img_path[present_img_pos] << endl;
		return;
	}

	Label_coord_in_img_vec.clear();		// 이미지 하나를 읽을때마다 label값이 저장될 vector 초기화
	Label_coord_in_sub_vec.clear();
	Rel_co_vec.clear();
	label_N.clear();					// label이 있는 subimg인지 아닌지를 판별해줌
	id4undo.clear();

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

		id4undo.push_back(coord.id);

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
	show_mini_map(img_label, Labeled_Image);		// 미니맵 보여주기 - 하나씩만
}




int main(int argc, char *argv[]) {
	if (argc == 1) {
		argc = 5;
		char* str_exe = (char *)"./Project1.exe";
		char* str_img = (char *)"./data/img ";
		char* str_train = (char *)"./data/train.txt";
		char* str_obj = (char *)"./data/obj.names";
		argv[0] = str_exe; "./Project1.exe";
		argv[1] = str_img;
		argv[2] = str_train;
		argv[3] = str_obj;
	}
	if (argc < 4) {
		cout << "Usage: [path_to_images] [train.txt] [obj.names] \n" << endl;
		return -1;
	}


	//train.txt 만들기
	system("dir data\\img/oe/b > ./data/file_names.txt");

	FILE *rFile = NULL;
	FILE *wFile = NULL;

	//읽어올  file_names.txt파일의 위치
	char* rfile_addr = (char*)".\\data/file_names.txt";
	char* wfile_addr = (char*)".\\data/train.txt";


	rFile = fopen(rfile_addr, "r");
	wFile = fopen(wfile_addr, "w");

	if (rFile != NULL) {
		char strTemp[128];
		char tmp4save[128];
		char *pStr;
		char *parse;
		char img_add_org[200] = ".\\data\\img/";
		char img_add[200] = ".\\data\\img/";

		while (!feof(rFile)) {
			strcpy(img_add, img_add_org);
			pStr = fgets(strTemp, sizeof(strTemp), rFile);
			strcpy(tmp4save, strTemp);

			if (pStr == NULL)
				goto Finish;

			parse = strtok(pStr, ".");
			parse = strtok(NULL, "");

			if (!strcmp(parse, "txt\n"))
				goto Finish;

			fputs(strcat(img_add, tmp4save), wFile);
		}
	}
	else
		printf("Not exist the file.");
Finish:

	fclose(rFile);
	fclose(wFile);

	// 수정되어야할 파일들이 저장될곳 생성
	system("mkdir .\\data\\img_re");

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

	// obj name
	while (!objnameFile.eof()) {
		getline(objnameFile, objname);

		if (objname != "")
			cout << "name " << class_number << " : " << objname << endl;

		objname_vec.push_back(objname);
		num_obj.push_back(class_number++);
	}

	sprintf(TrackbarName_img, "images %d", img_slider_max);
	createTrackbar(TrackbarName_img, Labeled, &present_img_pos, img_slider_max - 1, on_trackbar_img);	// img trackbar
	on_trackbar_img(present_img_pos, 0);

	do {
		coord_t coord;				// 라벨 하나의 좌표가 들어감
		coord_t Lcoord;				// sub_img에서의 라벨의 위치

		coord.id = -1;
		Lcoord.id = coord.id;

		Mat img_clone = img.clone();


		// 키 입력을 받을 때 마다 스위치문에서 반복적으로 label의 크기조정
		if (change_index_flag) {

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

			change_index_flag = false;
		}

		setTrackbarPos(TrackbarName_label, Labeled, present_label_pos);	//labeled trackbar의 포지션을 set 시켜준다.
		setTrackbarPos(TrackbarName_img, Labeled, present_img_pos);		//img trackbar의 포지션을 set 시킨다.


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

		coord_t temp;
		int exitoutloop = 0;
		int is_last = 0;

		switch (pressed_key) {

			// 라벨이 덜쳐져있는 img
		case '1':
		{
			if (label_N.at(present_label_pos) == -1) {

				label_N.pop_back();
				sub_num--;
			}
			for (vector<coord_t>::iterator iter = Label_coord_in_img_vec.begin(); iter != Label_coord_in_img_vec.end(); iter++) {
				if ((*iter).id == 91) {
					cout << "라벨을 다시라는 표시는 1번만 가능" << endl;
					exitoutloop = -1;
					break;
				}
			}
			if (exitoutloop == -1)
				break;

			temp.id = 91;
			temp.abs_rect.x = 1;
			temp.abs_rect.y = 1;
			temp.abs_rect.width = img.cols - 1;
			temp.abs_rect.height = img.rows - 1;

			Label_coord_in_img_vec.push_back(temp);

			Rel_co_vec.push_back(temp);
			Label_coord_in_sub_vec.push_back(temp);
			label_N.push_back(0);

			sub_num++;

			present_label_pos = sub_num - 1;

			createTrackbar(TrackbarName_label, Labeled, &present_label_pos, sub_num - 1, on_trackbar_select);

			change_index_flag = true;
			break;
		}

		// reindex
		case '2':
		{
			if (label_N.at(present_label_pos) == -1)
				break;
			if (Label_coord_in_img_vec.at(present_label_pos).id == 91)
				break;

			Label_coord_in_img_vec.at(present_label_pos).id = 92;
			change_index_flag = true;
			break;
		}

		// resize label box
		case '3':
		{
			if (label_N.at(present_label_pos) == -1)
				break;
			if (Label_coord_in_img_vec.at(present_label_pos).id == 91)
				break;

			Label_coord_in_img_vec.at(present_label_pos).id = 93;
			change_index_flag = true;
			break;
		}

		// remove label
		case '4':
		{
			if (label_N.at(present_label_pos) == -1)
				break;
			if (Label_coord_in_img_vec.at(present_label_pos).id == 91)
				break;

			Label_coord_in_img_vec.at(present_label_pos).id = 94;
			change_index_flag = true;

			break;
		}

		// undo labeling before next img
		case 'z':
		{
			if (label_N.at(present_label_pos) == -1)
				break;

			if (Label_coord_in_img_vec.at(present_label_pos).id == 91) {

				if (sub_num == 1) {

					Label_coord_in_img_vec.pop_back();

					Rel_co_vec.pop_back();

					Label_coord_in_sub_vec.pop_back();

					sub_img[sub_num - 1] = Mat::zeros(Size(200, 200), CV_8UC3);

					label_N.at(present_label_pos) = -1;
					on_trackbar_select(present_img_pos, 0);

					redirectory = true;

					break;
				}

				Label_coord_in_img_vec.pop_back();

				Rel_co_vec.pop_back();

				Label_coord_in_sub_vec.pop_back();

				label_N.pop_back();

				sub_num--;

				present_label_pos = sub_num - 1;

				createTrackbar(TrackbarName_label, Labeled, &present_label_pos, sub_num - 1, on_trackbar_select);

				change_index_flag = true;
				break;
			}

			Label_coord_in_img_vec.at(present_label_pos).id = id4undo.at(present_label_pos);
			change_index_flag = true;

			break;
		}

		// index표시
		case 'k':
		case 1048683:
			show_mark_class = !show_mark_class;
			change_index_flag = true;
			break;

			// 위 - 다음 img
		case 2490368:
		{
			++present_img_pos;
			if (present_img_pos >= img_slider_max || present_img_pos < 0) {				// trackbar가 img개수를 넘지 못하게 예외처리
				present_img_pos = min(max(0, present_img_pos), img_slider_max - 1);
				is_last = 1;
			}

			for (vector<coord_t>::iterator iter = Label_coord_in_img_vec.begin(); iter != Label_coord_in_img_vec.end(); iter++) {
				if ((*iter).id >= 91) {
					directory = true;
					break;
				}
			}
			save_coord();

			if (directory == true || redirectory == true) {
				if (is_last == 1)
					img_path[present_img_pos] = new_jpg_path;
				else
					img_path[present_img_pos - 1] = new_jpg_path;
			}

			redirectory = false;
			directory = false;

			on_trackbar_img(present_img_pos, 0);
			cout << "present img : " << img_path[present_img_pos] << endl;
			break;
		}

		// 아래 - 이전 img
		case 2621440:
		{
			--present_img_pos;
			if (present_img_pos >= img_slider_max || present_img_pos < 0) {
				present_img_pos = min(max(0, present_img_pos), img_slider_max - 1);
				is_last = 1;
			}

			for (vector<coord_t>::iterator iter = Label_coord_in_img_vec.begin(); iter != Label_coord_in_img_vec.end(); iter++) {
				if ((*iter).id >= 91) {
					directory = true;
					break;
				}
			}

			save_coord();

			if (directory == true || redirectory == true) {
				if (is_last == 1)
					img_path[present_img_pos] = new_jpg_path;
				else
					img_path[present_img_pos + 1] = new_jpg_path;
			}

			redirectory = false;
			directory = false;

			on_trackbar_img(present_img_pos, 0);
			cout << "present img : " << img_path[present_img_pos] << endl;
			break;
		}

		// 이전 sub_img
		case 2424832: // <-
		case 65361:   // <-
		{
			if (label_N.at(present_label_pos) == -1)
				break;
			--present_label_pos;
			if (present_label_pos >= sub_num || present_label_pos < 0) {				// trackbar가 label개수를 넘지 못하게 예외처리
				present_label_pos = min(max(0, present_label_pos), sub_num - 1);
			}

			on_trackbar_select(present_label_pos, 0);
			change_index_flag = true;
			break;
		}

		// 다음 sub_img
		case 2555904: // ->
		case 65363:   // ->
		{
			if (label_N.at(present_label_pos) == -1)
				break;

			++present_label_pos;
			if (present_label_pos >= sub_num || present_label_pos < 0) {
				present_label_pos = min(max(0, present_label_pos), sub_num - 1);
			}

			on_trackbar_select(present_label_pos, 0);
			change_index_flag = true;
			break;
		}
		default:;
		}
	} while (true);
	cout << "finished" << endl;
	return 0;
}