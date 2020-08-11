//****************************************************************************************
//****************************************************************************************
//---------------------------[[ �󺧸� �˼��� ���� �� ]]----------------------------------				 
//				 
//					������ �󺧸��� �� �̹����� �˼� �ϱ� ���� ��� 
//
//			- Usage: [path_to_images] [train.txt] [obj.names]
//
//			- �󺧸��� �� �̹����� Ȯ���Ͽ� Ȯ�� ��, ������ ũ�� ���� ����
//			+ W -> �� �ø��� , w -> �� ���̱�
//			+ A -> ���� �ø��� , a -> ���� ���̱�
//			+ S -> �Ʒ� �ø��� , s -> �Ʒ� ���̱�
//			+ D -> ������ �ø��� ,d -> ������ ���̱�
//
//			- Object ID ���� ����
//			+ 0 ~ 9 ���� Ű�� Object ID ����
//
//			- ����Ű�� Image ����, ���� �̹����� �ִ� Label ����
//			+ �� : Next Image 
//			+ �� : Previous Imgae
//			+ �� : Next Label in present Image
//			+ �� : Previous Label in present Image
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
#define percentage 1.5	// �� �ֺ��� ������ ����

using namespace cv;
using namespace std;


struct coord_t {
	cv::Rect_<float> abs_rect;
	int id;
};


int img_slider_max;				// �о�� �̹����� �Ѱ���
int label_slider_max;			// ���� �̹����� �� �Ѱ���
int present_img_pos;			// ���� �̹��� ��ȣ
int present_label_pos;			// ���� �� ��ȣ

string img_path[5000];			// img path
string txt_path;				// txt path
string Labeled = "Labeled";
string Labeled_Image = "Labeled Image";
string Original_Image = "Original Image";
Mat img;						// �����̹���
Mat src;						// display ����� ��´�.
Mat sub_img[1000];				// ���̹����� sub img�� ��´�.
Mat image_cloned;
static int sub_num;				// �� �̹����� ����ִ� label�� �� ����

vector <string> objname_vec;	// ������Ʈ �̸��� ��Ƶδ� vector
vector <int> num_obj;			// ������Ʈ ���� ��

char TrackbarName_label[50];			// label trackbar �̸�
char TrackbarName_img[50];				// img trackbar �̸�

vector<coord_t> Label_coord_in_img_vec;	// �� �̹����� ���������� �󺧵��� ���� �����Ұ�.
vector<coord_t> Label_coord_in_sub_vec;	// sub img ������ ���� ��ġ - sub img������ ���� �ϳ����� �����ֱ����ؼ�
vector<coord_t> Rel_co_vec;				// rel�� ũ�⸦ �����ϱ� ���� vector

vector<int> label_N;					// �̹����� ���� ���� �˻�

atomic<bool> exit_flag(false);			// switch ������ escŰ�� �̿��� ���� ����
atomic<bool> move_label_flag(false);	// switch ������ ���� �ٽ� �׷��־�� �Ѵٴ� ����
atomic<bool> show_mark_class(true);		//switch ������ object id �����ִ� ���� �����ϴ� ����



// �̴ϸ�â�� ��� ������ �̹����� �����ش�.
static void  show_mini_map(Mat img, string str) {
	resize(img, img, Size(560, 400), 0, 0);
	imshow(str, img);
}

// �Ʒ��� Ʈ���� : sub img(labeled)�� ������
static void on_trackbar_select(int, void*) {
	src = sub_img[present_label_pos];
	imshow(Labeled, src);
}

// �󺧼��������� ���� �����ؼ� �׸�ĭ�� �׸���.
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

//��� ����ġ�� ���� sub�̹��������� ����ġ���� �˾Ƴ���.
static coord_t manage_coord(float *relative_coord_0, float *relative_coord_1, float *relative_coord_2, float *relative_coord_3, Mat subimg[], coord_t coord, coord_t *Label_coord_in_sub, int present) {

	coord_t sub_img_coord;			// sub_img�� ��ǥ�� ��� ��
	sub_img_coord.id = coord.id;

	// ���� x��������� img�� �Ѿ�°��� ����
	if (*relative_coord_0 - *relative_coord_2 / 2 < 0 || *relative_coord_0 + *relative_coord_2 / 2 > 1)
		*relative_coord_2 = min(*relative_coord_0, 1 - *relative_coord_0);

	// ���� y��������� img�� �Ѿ�°��� ����
	if (*relative_coord_1 - *relative_coord_3 / 2 < 0 || *relative_coord_1 + *relative_coord_3 / 2 > 1)
		*relative_coord_3 = min(*relative_coord_1, 1 - *relative_coord_1);

	// img���� ���� �����ִ� ���� ����
	coord.abs_rect.x = (*relative_coord_0 - *relative_coord_2 / 2) * (float)img.cols;
	coord.abs_rect.y = (*relative_coord_1 - *relative_coord_3 / 2) * (float)img.rows;
	coord.abs_rect.width = *relative_coord_2 *(float)img.cols;
	coord.abs_rect.height = *relative_coord_3 * (float)img.rows;

	// img���� sub_img�� �����ִ� ����
	{
		if (*relative_coord_0 - percentage * *relative_coord_2 / 2 >= 0)
			sub_img_coord.abs_rect.x = (*relative_coord_0 - percentage * *relative_coord_2 / 2)*(float)img.cols;
		else
			sub_img_coord.abs_rect.x = (float)0;

		if (*relative_coord_0 + percentage * *relative_coord_2 / 2 <= 1) {
			sub_img_coord.abs_rect.width = percentage * *relative_coord_2 * (float)img.cols;
			if (sub_img_coord.abs_rect.width >= img.cols)		// ����Ȯ���Ų �ڽ��� �����̹����� ũ�⸦ �Ѿ ����ó��
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
			if (sub_img_coord.abs_rect.height >= img.rows)		// ����Ȯ���Ų �ڽ��� �����̹����� ũ�⸦ �Ѿ ����ó��
				sub_img_coord.abs_rect.height = img.rows;
		}
		else
			sub_img_coord.abs_rect.height = img.rows - sub_img_coord.abs_rect.y;
	}

	Rect rect_130p(sub_img_coord.abs_rect.x, sub_img_coord.abs_rect.y, sub_img_coord.abs_rect.width, sub_img_coord.abs_rect.height);

	//sub img���� ���� ��ġ
	auto &i = coord;
	(*Label_coord_in_sub).abs_rect.x = i.abs_rect.x - sub_img_coord.abs_rect.x;
	(*Label_coord_in_sub).abs_rect.y = i.abs_rect.y - sub_img_coord.abs_rect.y;
	(*Label_coord_in_sub).abs_rect.width = i.abs_rect.width;
	(*Label_coord_in_sub).abs_rect.height = i.abs_rect.height;

	sub_img[present] = img(rect_130p).clone();	// ���� �̹����� ���ֺ��� ���簢�� ����� �ϳ��� sub img�� ������.

	return coord;
}

// ���� Ʈ���� : img �� ������
static void on_trackbar_img(int, void*) {

	present_label_pos = 0;							// ���ο� �̹����� �����Ҷ����� ù��° sub img�� ������� ��.
	txt_path = img_path[present_img_pos];
	txt_path.replace(txt_path.end() - 4, txt_path.end(), ".txt");	// img_path�κ��� �����Ͽ� txt_path����

	img = imread(img_path[present_img_pos]);		// ���� �̹����� �о�� - �����̹���
	Mat img_clone = img.clone();					// Ʈ���� �� �۾��� �� �Ŀ� ���� �����ְԵ�
	image_cloned = img_clone.clone();
	Mat img_label;									// label �Ǿ��ִ� ��ü ���

	ifstream readLabel;
	readLabel.open(txt_path);

	if (!readLabel.is_open()) {
		cout << "Label.txt is not exist : " << img_path[present_img_pos] << endl;
	}

	Label_coord_in_img_vec.clear();		// �̹��� �ϳ��� ���������� label���� ����� vector �ʱ�ȭ
	Label_coord_in_sub_vec.clear();
	Rel_co_vec.clear();
	label_N.clear();					// label�� �ִ� subimg���� �ƴ����� �Ǻ�����

	std::vector<int> co_id;

	int k = 0;
	sub_num = 0;

	// �̹����� �ִ� �󺧵�� sub img�� label����
	for (string line; getline(readLabel, line);) {
		std::stringstream ss(line);

		coord_t coord;				// �� �ϳ��� ��ǥ�� ��
		coord_t sub_img_coord;		// �� �ϳ��� 130percent�� �ֺ� �̹����� ��
		coord_t Label_coord_in_sub;		// sub_img������ ���� ��ġ
		coord_t rel_temp;			// Rel_co_vec�� ���� ��

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

		// ���� ���� ��ǥ�� ����ؼ� �־��ش�.
		Label_coord_in_img_vec.push_back(manage_coord(&relative_coord[0], &relative_coord[1], &relative_coord[2], &relative_coord[3], sub_img, coord, &Label_coord_in_sub, k));
		Label_coord_in_sub_vec.push_back(Label_coord_in_sub);

		//�����ϱ� ������ ��� ����ġ �־����
		rel_temp.abs_rect.x = relative_coord[0];
		rel_temp.abs_rect.y = relative_coord[1];
		rel_temp.abs_rect.width = relative_coord[2];
		rel_temp.abs_rect.height = relative_coord[3];

		Rel_co_vec.push_back(rel_temp);


		auto &i = Label_coord_in_img_vec.at(k);
		// �� �ڽ� �׸��� - subimg, clone����img ��
		make_colored_label(coord.id, sub_img, img_clone, k, i); //object ���� �󺧸� ����

		label_N.push_back(0);

		sub_num++;
		k++;

	}

	readLabel.close();

	label_slider_max = sub_num;

	if (sub_num == 0) {
		sub_img[sub_num++] = Mat::zeros(Size(200, 200), CV_8UC3);		 //label �� ���ٸ� ������ ȭ���� ǥ��
		label_N.push_back(-1);
	}


	src = sub_img[present_label_pos];
	imshow(Labeled, src);

	sprintf(TrackbarName_label, "Labeled picture", sub_num);
	createTrackbar(TrackbarName_label, Labeled, &present_label_pos, sub_num - 1, on_trackbar_select);		// label trackbar
	on_trackbar_select(present_label_pos, 0);

	img_label = img_clone.clone();
	
	show_mini_map(img_label, Labeled_Image);		// �̴ϸ� �����ֱ� 
}

// change 2 relative coord
static coord_t current_to_relative(coord_t current_coord) {
	current_coord.abs_rect.x = ((current_coord.abs_rect.x + (current_coord.abs_rect.x + current_coord.abs_rect.width)) / 2) / (float)img.cols;
	current_coord.abs_rect.y = ((current_coord.abs_rect.y + (current_coord.abs_rect.y + current_coord.abs_rect.height)) / 2) / (float)img.rows;
	current_coord.abs_rect.width = current_coord.abs_rect.width / (float)img.cols;
	current_coord.abs_rect.height = current_coord.abs_rect.height / (float)img.rows;

	return current_coord;
}

// img�� �������� �Ѿ�� ������ ���� ���� �����Ѵ�.
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
	readFile.open(train_filename);					//������ train.txt �� ������� �־���Ѵ�.

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


	ifstream objnameFile(synset_filename);	//obj ���� �޾ƿ�
	int class_number = 0;
	string objname;
	while (!objnameFile.eof()) {
		getline(objnameFile, objname);
		class_number++;							//obj name ����
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
		setTrackbarPos(TrackbarName_label, Labeled, present_label_pos);	//labeled trackbar�� �������� set �����ش�.
		setTrackbarPos(TrackbarName_img, Labeled, present_img_pos);		//img trackbar�� �������� set ��Ų��.

		coord_t coord;				// �� �ϳ��� ��ǥ�� ��
		coord_t Lcoord;				// sub_img������ ���� ��ġ

		coord.id = -1;
		Lcoord.id = coord.id;

		Mat img_clone = img.clone();


		// Ű �Է��� ���� �� ���� ����ġ������ �ݺ������� label�� ũ������
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


			// ���� �̹����� ���� ��� ǥ���ϰ� ���� �ִ� �󺧸� ����
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

		// ������Ʈ ID �����ϱ� 0~9 �� ������ ����
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

		case 2490368:   // �� - ���� img
			++present_img_pos;
			if (present_img_pos >= img_slider_max || present_img_pos < 0) {				// trackbar�� img������ ���� ���ϰ� ����ó��
				present_img_pos = min(max(0, present_img_pos), img_slider_max - 1);
			}
			save_coord();
			on_trackbar_img(present_img_pos, 0);
			cout << "present img : " << img_path[present_img_pos] << endl;
			break;

		case 2621440:   // �Ʒ� - ���� img
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
			if (present_label_pos >= sub_num || present_label_pos < 0) {				// trackbar�� label������ ���� ���ϰ� ����ó��
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