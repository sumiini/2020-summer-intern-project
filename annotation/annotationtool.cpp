#include <algorithm> // std::unique
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream> // std::ofstream
#include <future>  // C++11: async(); feature<>;
#include <iomanip>
#include <iostream>
#include <locale>
#include <numeric>
#include <vector>
#include <windows.h>
#include <opencv2/core/version.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp> // C++
//#include <opencv2/optflow.hpp>
#include <opencv2/video/tracking.hpp>

#ifdef _DEBUG
#define LIB_SUFFIX "d.lib"
#else
#define LIB_SUFFIX ".lib"
#endif // DEBUG

#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio.hpp"
#define OPENCV_VERSION                                                         \
  CVAUX_STR(CV_VERSION_MAJOR)                                                  \
  "" CVAUX_STR(CV_VERSION_MINOR) "" CVAUX_STR(CV_VERSION_REVISION)
#pragma comment(lib, "opencv_world" OPENCV_VERSION LIB_SUFFIX)
#else
#define OPENCV_VERSION                                                         \
  CVAUX_STR(CV_VERSION_EPOCH)                                                  \
  "" CVAUX_STR(CV_VERSION_MAJOR) "" CVAUX_STR(CV_VERSION_MINOR)
#pragma comment(lib, "opencv_core" OPENCV_VERSION LIB_SUFFIX)
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION LIB_SUFFIX)
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION LIB_SUFFIX)
#endif

#ifndef CV_FILLED
#define CV_FILLED cv::FILLED
#endif

using namespace cv;

// label coordinates
struct coord_t {
	cv::Rect_<float> abs_rect;
	int id;
	int multi_label;
};

class Tracker_optflow {
public:
	const int flow_error;

	Tracker_optflow(int win_size = 15, int max_level = 3, int iterations = 8000,
		int _flow_error = -1)
		: flow_error((_flow_error > 0) ? _flow_error : (win_size * 4)) {
		sync_PyrLKOpticalFlow = cv::SparsePyrLKOpticalFlow::create();
		sync_PyrLKOpticalFlow->setWinSize(
			cv::Size(win_size, win_size));             // 9, 15, 21, 31
		sync_PyrLKOpticalFlow->setMaxLevel(max_level); // +- 3 pt
	}

	// just to avoid extra allocations
	cv::Mat dst_grey;
	cv::Mat prev_pts_flow, cur_pts_flow;
	cv::Mat status, err;

	cv::Mat src_grey; // used in both functions
	cv::Ptr<cv::SparsePyrLKOpticalFlow> sync_PyrLKOpticalFlow;

	std::vector<coord_t> cur_bbox_vec;
	std::vector<bool> good_bbox_vec_flags;

	void update_cur_bbox_vec(std::vector<coord_t> _cur_bbox_vec) {
		cur_bbox_vec = _cur_bbox_vec;
		good_bbox_vec_flags = std::vector<bool>(cur_bbox_vec.size(), true);
		cv::Mat prev_pts, cur_pts_flow;

		for (auto &i : cur_bbox_vec) {
			float x_center = (i.abs_rect.x + i.abs_rect.width / 2.0F);
			float y_center = (i.abs_rect.y + i.abs_rect.height / 2.0F);
			prev_pts.push_back(cv::Point2f(x_center, y_center));
		}

		if (prev_pts.rows == 0)
			prev_pts_flow = cv::Mat();
		else
			cv::transpose(prev_pts, prev_pts_flow);
	}

	void update_tracking_flow(cv::Mat new_src_mat,
		std::vector<coord_t> _cur_bbox_vec) {
		if (new_src_mat.channels() == 1) {
			src_grey = new_src_mat.clone();
		}
		else if (new_src_mat.channels() == 3) {
			cv::cvtColor(new_src_mat, src_grey, cv::COLOR_BGR2GRAY, 1);
		}
		else if (new_src_mat.channels() == 4) {
			cv::cvtColor(new_src_mat, src_grey, cv::COLOR_BGRA2GRAY, 1);
		}
		else {
			std::cerr
				<< " Warning: new_src_mat.channels() is not: 1, 3 or 4. It is = "
				<< new_src_mat.channels() << " \n";
			return;
		}
		update_cur_bbox_vec(_cur_bbox_vec);
	}

	std::vector<coord_t> tracking_flow(cv::Mat new_dst_mat,
		bool check_error = true) {
		if (sync_PyrLKOpticalFlow.empty()) {
			std::cout << "sync_PyrLKOpticalFlow isn't initialized \n";
			return cur_bbox_vec;
		}

		cv::cvtColor(new_dst_mat, dst_grey, cv::COLOR_BGR2GRAY, 1);

		if (src_grey.rows != dst_grey.rows || src_grey.cols != dst_grey.cols) {
			src_grey = dst_grey.clone();
			// std::cerr << " Warning: src_grey.rows != dst_grey.rows || src_grey.cols
			// != dst_grey.cols \n";
			return cur_bbox_vec;
		}

		if (prev_pts_flow.cols < 1) {
			return cur_bbox_vec;
		}

		////sync_PyrLKOpticalFlow_gpu.sparse(src_grey_gpu, dst_grey_gpu,
		/// prev_pts_flow_gpu, cur_pts_flow_gpu, status_gpu, &err_gpu);    //
		/// OpenCV 2.4.x
		sync_PyrLKOpticalFlow->calc(src_grey, dst_grey, prev_pts_flow, cur_pts_flow,
			status, err); // OpenCV 3.x

		dst_grey.copyTo(src_grey);

		std::vector<coord_t> result_bbox_vec;

		if (err.rows == cur_bbox_vec.size() && status.rows == cur_bbox_vec.size()) {
			for (size_t i = 0; i < cur_bbox_vec.size(); ++i) {
				cv::Point2f cur_key_pt = cur_pts_flow.at<cv::Point2f>(0, i);
				cv::Point2f prev_key_pt = prev_pts_flow.at<cv::Point2f>(0, i);

				float moved_x = cur_key_pt.x - prev_key_pt.x;
				float moved_y = cur_key_pt.y - prev_key_pt.y;

				if (abs(moved_x) < 100 && abs(moved_y) < 100 && good_bbox_vec_flags[i])
					if (err.at<float>(0, i) < flow_error &&
						status.at<unsigned char>(0, i) != 0 &&
						((float)cur_bbox_vec[i].abs_rect.x + moved_x) > 0 &&
						((float)cur_bbox_vec[i].abs_rect.y + moved_y) > 0) {
						cur_bbox_vec[i].abs_rect.x += moved_x; // +0.5;
						cur_bbox_vec[i].abs_rect.y += moved_y; // +0.5;
						result_bbox_vec.push_back(cur_bbox_vec[i]);
					}
					else
						good_bbox_vec_flags[i] = false;
				else
					good_bbox_vec_flags[i] = false;

				// if(!check_error && !good_bbox_vec_flags[i])
				// result_bbox_vec.push_back(cur_bbox_vec[i]);
			}
		}

		prev_pts_flow = cur_pts_flow.clone();

		return result_bbox_vec;
	}
};
std::atomic<bool> isbr(false);
std::atomic<bool> right_button_click;
std::atomic<bool> left_button_down;
std::atomic<int> move_rect_id;
std::atomic<int> move_edge_id;
std::atomic<bool> move_rect;
std::atomic<bool> move_edge;
std::atomic<bool> clear_marks;
std::atomic<bool> copy_previous_marks(false);
std::atomic<bool> tracker_copy_previous_marks(false);
std::atomic<bool> copy_current_box(false);
std::atomic<int> change_class_id = 0;
std::atomic<int> zoomRec = 720;
std::atomic<int> mousex;
std::atomic<int> mousey;
std::atomic<int> change_image = 0;
std::atomic<bool> mini_window(false);

std::atomic<bool> show_help;
std::atomic<bool> exit_flag(false);

std::atomic<int> mark_line_width(2);
std::atomic<int> current_mark_line_width;// default mark line width is 2 pixels.
const int MAX_MARK_LINE_WIDTH = 3;
std::atomic<bool> show_mark_class(true);
std::atomic<bool> show_box(true);
std::atomic<bool> show_mark_box(true);

std::atomic<bool> delete_selected(false);

std::atomic<bool> resize_u_inc(false);
std::atomic<bool> resize_d_inc(false);
std::atomic<bool> resize_l_inc(false);
std::atomic<bool> resize_r_inc(false);
std::atomic<bool> resize_u_dec(false);
std::atomic<bool> resize_d_dec(false);
std::atomic<bool> resize_l_dec(false);
std::atomic<bool> resize_r_dec(false);
std::atomic<bool> zooming(false);
std::atomic<int> imgHOrig, imgWOrig;
int topPad, botPad, leftPad, rightPad;
float scaleWidth, scaleHeight;
int scrollHeightPad = 0;
int scrollWidthPad = 0;
float mouseScroll = 0;
bool mousePanning = false;


std::atomic<int> x_start, y_start;
std::atomic<int> x_end, y_end;
std::atomic<int> x_size, y_size;
std::atomic<bool> draw_select, selected, undo;

std::atomic<int> add_id_img;
Rect prev_img_rect(0, 0, 50, 100);
Rect next_img_rect(1280 - 50, 0, 50, 100);

cv::Mat resizeKeepAspectRatio(const cv::Mat &input, const cv::Size &dstSize, const cv::Scalar &bgcolor)
{
	cv::Mat output;

	double h1 = dstSize.width * (input.rows / (double)input.cols);
	double w2 = dstSize.height * (input.cols / (double)input.rows);

	if (h1 <= dstSize.height) {
		cv::resize(input, output, cv::Size(dstSize.width, h1));
	}
	else {
		cv::resize(input, output, cv::Size(w2, dstSize.height));
	}

	topPad = (dstSize.height - output.rows) / 2;
	botPad = (dstSize.height - output.rows + 1) / 2;
	leftPad = (dstSize.width - output.cols) / 2;
	rightPad = (dstSize.width - output.cols + 1) / 2;

	cv::copyMakeBorder(output, output, topPad, botPad, leftPad, rightPad, cv::BORDER_CONSTANT, bgcolor);

	// multiply by zoom in scale and crop
	//int imgH = input.rows + input.rows * mouseScroll;
	//int imgW = input.cols + input.cols * mouseScroll;
	int maxX0 = dstSize.width - ((float)output.cols * (1.0 - mouseScroll));
	int maxY0 = dstSize.height - ((float)output.rows * (1.0 - mouseScroll));
	scrollWidthPad = scrollWidthPad <= maxX0 ? scrollWidthPad : maxX0;
	scrollHeightPad = scrollHeightPad <= maxY0 ? scrollHeightPad : maxY0;
	int x0 = scrollWidthPad;
	int y0 = scrollHeightPad;

	output = output(Rect(x0, y0, output.cols - (float)output.cols * mouseScroll, output.rows - (float)output.rows * mouseScroll));
	int oldWidth = output.cols;
	int oldHeight = output.rows;
	cv::resize(output, output, cv::Size(dstSize.width, dstSize.height));
	scaleHeight = (float)output.rows / oldHeight;
	scaleWidth = (float)output.cols / oldWidth;
	return output;
}

void callback_mouse_click(int event, int x, int y, int flags, void *user_data) {
	mousex = x;
	mousey = y;

	if (flags&cv::EVENT_FLAG_CTRLKEY) {
		/*if (event == cv::EVENT_MOUSEWHEEL) {
			if (getMouseWheelDelta(flags) > 0) {
				std::cout << "cv::EVENT_MOUSEWHEEL_UP \n";
				std::cout << getMouseWheelDelta(flags);
				change_image = 1;
			}
			else {
				std::cout << "cv::EVENT_MOUSEWHEEL_DOWN \n";
				change_image = 2;
			}
		}*/
		if (getMouseWheelDelta(flags) > 0 && mouseScroll < 0.9) {
			mouseScroll += (float)0.1;
			zooming = true;
			std::min(mouseScroll, 0.9f);
			//std::cout << "Zooming in " << mouseScroll << "\n";
		}
		else if (getMouseWheelDelta(flags) < 0 && mouseScroll > 0.1) {
			mouseScroll -= (float)0.1;
			zooming = true;
			std::max(mouseScroll, 0.0f);
			if (event == cv::EVENT_RBUTTONDOWN) mousePanning = true;
			//std::cout << "Zooming out " << mouseScroll << "\n";
		}
		if (event == cv::EVENT_RBUTTONDOWN) mousePanning = true;
		if (event == cv::EVENT_RBUTTONUP) mousePanning = false;
		if (event == cv::EVENT_MOUSEMOVE) {
			if (mousePanning) {
				scrollHeightPad += (y - y_start);
				scrollWidthPad += (x - x_start);
				// clamp min to above or equal to zero
				scrollHeightPad = max(scrollHeightPad, 0);
				scrollWidthPad = max(scrollWidthPad, 0);
				// clamp scroll max:
				//scrollHeightPad = min(scrollHeightPad, original_img_size.height - current_img_size.height);
				//scrollWidthPad = min(scrollWidthPad, original_img_size.width - current_img_size.width);
				x_start = x;
				y_start = y;
			}
			//else {
			//	if (mouseScroll >= 0 && mouseScroll < 0.1) {
			//		//std::cout << "fully zoomed out!!, to zoom in please change scroll direction, ";
			//	}
			//	else {
			//		//std::cout << "fully zoomed in!!, to zoom out please change scroll direction, ";
			//	}
			//	//std::cout << "event = " << event << ", mouse delta: " << getMouseWheelDelta(flags) << " mouse scroll: " << mouseScroll << "\n";
			//}
		}
	}
	if (event == cv::EVENT_LBUTTONDBLCLK) {
		std::cout << "cv::EVENT_LBUTTONDBLCLK \n";
	}
	else if (event == cv::EVENT_LBUTTONDOWN) {
		left_button_down = true;
		if (!isbr)
			draw_select = true;
		selected = false;
		x_start = x;
		y_start = y;

		if (prev_img_rect.contains(Point2i(x, y)))
			add_id_img = -1;
		else if (next_img_rect.contains(Point2i(x, y)))
			add_id_img = 1;
		else
			add_id_img = 0;
		std::cout << "cv::EVENT_LBUTTONDOWN \n";
	}
	else if (event == cv::EVENT_LBUTTONUP) {
		left_button_down = false;
		move_edge = true;
		x_size = abs(x - x_start);
		y_size = abs(y - y_start);
		x_end = max(x, 0);
		y_end = max(y, 0);
		draw_select = false;
		if (!isbr)
			selected = true;
		else if (isbr)
			isbr = false;
		std::cout << "cv::EVENT_LBUTTONUP \n";
	}
	else if (event == cv::EVENT_RBUTTONDOWN) {
		right_button_click = true;

		x_start = x;
		y_start = y;

		std::cout << "cv::EVENT_RBUTTONDOWN \n";
	}
	else if (event == cv::EVENT_RBUTTONUP) {
		right_button_click = false;
		move_rect = true;
		//mousePanning = false;
	}
	if (event == cv::EVENT_RBUTTONDBLCLK) {

		std::cout << "cv::EVENT_RBUTTONDBLCLK \n";
	}
	else if (event == cv::EVENT_MOUSEMOVE) {
		x_end = max(x, 0);
		y_end = max(y, 0);
		change_class_id = 0;
	}
}


class comma : public std::numpunct<char> {
public:
	comma() : std::numpunct<char>() {}

protected:
	char do_decimal_point() const { return '.'; }
};

bool cmp(const coord_t & a, const coord_t & b)
{
	if (a.id > b.id) return true; // 제일 먼저 f를 기준으로 오름차순 정렬
	else if (a.id == b.id) // 만약에 f가 같다면
	{
		if (a.multi_label > b.multi_label) return true; // t를 기준으로 오름차순 정렬
	}

	// 각 경우에 대하여 else를 고려할 필요가 없다.
	return false;
}


int main(int argc, char *argv[]) {

	try {
		std::locale loccomma(std::locale::classic(), new comma);
		std::locale::global(loccomma);

		std::string images_path = "./";

		if (argc >= 2) {
			images_path = std::string(argv[1]); // path to images, train and synset
		}
		else {
			std::cout << "Usage: [path_to_images] [train.txt] [obj.names] \n";
			return 0;
		}

		std::string train_filename = images_path + "train.txt";
		std::string synset_filename = images_path + "obj.names";

		if (argc >= 3) {
			train_filename = std::string(argv[2]); // file containing: list of images
		}

		if (argc >= 4) {
			synset_filename = std::string(argv[3]); // file containing: object names
		}

		// optical flow tracker
		Tracker_optflow tracker_optflow;
		cv::Mat optflow_img;

		// capture frames from video file - 1 frame per 3 seconds of video
		if (argc >= 4 && (train_filename == "cap_video" ||
			train_filename == "cap_video_backward")) {
			const std::string videofile = synset_filename;
			cv::VideoCapture cap(videofile);
#ifndef CV_VERSION_EPOCH // OpenCV 3.x
			const int fps = cap.get(cv::CAP_PROP_FPS);
#else // OpenCV 2.x
			const int fps = cap.get(CV_CAP_PROP_FPS);
#endif
			int frame_counter = 0, image_counter = 0;
			int backward = (train_filename == "cap_video_backward") ? 1 : 0;
			if (backward)
				image_counter = 99999999; // 99M
			float save_each_frames = 50;
			if (argc >= 5)
				save_each_frames = std::stoul(std::string(argv[4]));

			int pos_filename = 0;
			if ((1 + videofile.find_last_of("\\")) < videofile.length())
				pos_filename = 1 + videofile.find_last_of("\\");
			if ((1 + videofile.find_last_of("/")) < videofile.length())
				pos_filename =
				std::max(pos_filename, 1 + (int)videofile.find_last_of("/"));
			std::string const filename = videofile.substr(pos_filename);
			std::string const filename_without_ext =
				filename.substr(0, filename.find_last_of("."));

			for (cv::Mat frame; cap >> frame, cap.isOpened() && !frame.empty();) {
				cv::imshow("video cap to frames", frame);
#ifndef CV_VERSION_EPOCH
				int pressed_key = cv::waitKeyEx(20); // OpenCV 3.x
#else
				int pressed_key = cv::waitKey(20); // OpenCV 2.x
#endif
				if (pressed_key == 27 || pressed_key == 1048603)
					break; // ESC - exit (OpenCV 2.x / 3.x)
				if (frame_counter++ >=
					save_each_frames) { // save frame for each 3 second
					frame_counter = 0;
					std::stringstream image_counter_ss;
					image_counter_ss << std::setw(8) << std::setfill('0')
						<< image_counter;
					if (backward)
						image_counter--;
					else
						image_counter++;
					std::string img_name = images_path + "/" + filename_without_ext +
						"_" + image_counter_ss.str() + ".jpg";
					std::cout << "saved " << img_name << std::endl;
					cv::imwrite(img_name, frame);
				}
			}
			exit(0);
		}

		bool show_mouse_coords = false;
		std::vector<std::string> filenames_in_folder;
		// glob(images_path, filenames_in_folder); // void glob(String pattern,
		// std::vector<String>& result, bool recursive = false);
		cv::String images_path_cv = images_path;
		std::vector<cv::String> filenames_in_folder_cv;
		glob(images_path_cv,
			filenames_in_folder_cv); // void glob(String pattern,
									 // std::vector<String>& result, bool recursive
									 // = false);
		for (auto &i : filenames_in_folder_cv)
			filenames_in_folder.push_back(i);

		std::vector<std::string> jpg_filenames_path;
		std::vector<std::string> jpg_filenames;
		std::vector<std::string> jpg_filenames_without_ext;
		std::vector<std::string> image_ext;
		std::vector<std::string> txt_filenames;
		std::vector<std::string> jpg_in_train;
		std::vector<std::string> synset_txt;

		// image-paths to txt-paths
		for (auto &i : filenames_in_folder) {
			int pos_filename = 0;
			if ((1 + i.find_last_of("\\")) < i.length())
				pos_filename = 1 + i.find_last_of("\\");
			if ((1 + i.find_last_of("/")) < i.length())
				pos_filename = std::max(pos_filename, 1 + (int)i.find_last_of("/"));

			std::string const filename = i.substr(pos_filename);
			std::string const ext = i.substr(i.find_last_of(".") + 1);
			std::string const filename_without_ext =
				filename.substr(0, filename.find_last_of("."));

			if (ext == "jpg" || ext == "JPG" || ext == "jpeg" || ext == "JPEG" ||
				ext == "bmp" || ext == "BMP" || ext == "png" || ext == "PNG" ||
				ext == "ppm" || ext == "PPM") {
				jpg_filenames_without_ext.push_back(filename_without_ext);
				image_ext.push_back(ext);
				jpg_filenames.push_back(filename);
				jpg_filenames_path.push_back(i);
			}
			if (ext == "txt") {
				txt_filenames.push_back(filename_without_ext);
			}
		}
		std::sort(jpg_filenames.begin(), jpg_filenames.end());
		std::sort(jpg_filenames_path.begin(), jpg_filenames_path.end());
		std::sort(txt_filenames.begin(), txt_filenames.end());

		if (jpg_filenames.size() == 0) {
			std::cout << "Error: Image files not found by path: " << images_path
				<< std::endl;
			return 0;
		}

		// check whether there are files with the same names (but different
		// extensions)
		{
			auto sorted_names_without_ext = jpg_filenames_without_ext;
			std::sort(sorted_names_without_ext.begin(),
				sorted_names_without_ext.end());
			for (size_t i = 1; i < sorted_names_without_ext.size(); ++i) {
				if (sorted_names_without_ext[i - 1] == sorted_names_without_ext[i]) {
					std::cout << "Error: Can't create " << sorted_names_without_ext[i]
						<< ".txt file for several images with different extensions "
						"but with the same filename: "
						<< sorted_names_without_ext[i] << std::endl;
					// print duplicate images
					for (size_t k = 0; k < jpg_filenames_without_ext.size(); ++k) {
						if (jpg_filenames_without_ext[k] == sorted_names_without_ext[i]) {
							std::cout << jpg_filenames_without_ext[k] << "." << image_ext[k]
								<< std::endl;
						}
					}
					return 0;
				}
			}
		}

		// intersect jpg & txt
		std::vector<std::string> intersect_filenames(jpg_filenames.size());
		std::vector<std::string> difference_filenames(jpg_filenames.size());
		std::vector<std::string> intersect_ext;
		std::vector<std::string> difference_ext;

		auto dif_it_end = std::set_difference(
			jpg_filenames_without_ext.begin(), jpg_filenames_without_ext.end(),
			txt_filenames.begin(), txt_filenames.end(),
			difference_filenames.begin());
		difference_filenames.resize(dif_it_end - difference_filenames.begin());

		auto inter_it_end = std::set_intersection(
			jpg_filenames_without_ext.begin(), jpg_filenames_without_ext.end(),
			txt_filenames.begin(), txt_filenames.end(),
			intersect_filenames.begin());
		intersect_filenames.resize(inter_it_end - intersect_filenames.begin());

		// get intersect extensions for intersect_filenames
		for (auto &i : intersect_filenames) {
			size_t ext_index = find(jpg_filenames_without_ext.begin(),
				jpg_filenames_without_ext.end(), i) -
				jpg_filenames_without_ext.begin();
			intersect_ext.push_back(image_ext[ext_index]);
		}

		// get difference extensions for intersect_filenames
		for (auto &i : difference_filenames) {
			size_t ext_index = find(jpg_filenames_without_ext.begin(),
				jpg_filenames_without_ext.end(), i) -
				jpg_filenames_without_ext.begin();
			difference_ext.push_back(image_ext[ext_index]);
		}

		txt_filenames.clear();
		for (auto &i : intersect_filenames) {
			txt_filenames.push_back(i + ".txt");
		}

		int image_list_count = max(1, (int)jpg_filenames_path.size() - 1);

		// store train.txt
		std::ofstream ofs_train(train_filename, std::ios::out | std::ios::trunc);
		if (!ofs_train.is_open()) {
			throw(std::runtime_error("Can't open file: " + train_filename));
		}

		for (size_t i = 0; i < intersect_filenames.size(); ++i) {
			ofs_train << images_path << "/" << intersect_filenames[i] << "."
				<< intersect_ext[i] << std::endl;
		}
		ofs_train.flush();
		std::cout << "File opened for output: " << train_filename << std::endl;

		// load synset.txt
		{
			std::ifstream ifs(synset_filename);
			if (!ifs.is_open()) {
				throw(std::runtime_error("Can't open file: " + synset_filename));
			}

			for (std::string line; getline(ifs, line);)
				synset_txt.push_back(line);
		}
		std::cout << "File loaded: " << synset_filename << std::endl;

		Mat preview(Size(100, 100), CV_8UC3);
		Mat full_image(Size(1280, 720), CV_8UC3);
		std::vector<Mat>zoom_out_image;
		Mat full_image_store(Size(1280, 720), CV_8UC3);
		Mat frame(Size(full_image.cols, full_image.rows + preview.rows), CV_8UC3);

		Rect full_rect_dst(Point2i(0, preview.rows),
			Size(frame.cols, frame.rows - preview.rows));
		Mat full_image_roi = frame(full_rect_dst);


		size_t const preview_number = frame.cols / preview.cols;

		// labels on the current image
		std::vector<coord_t> current_coord_vec;
		Size current_img_size;
		Size original_img_size;




		std::vector<Mat> previewImagesCache;
		float scaleFactor = 1.0;
		float resizeFactorImage = 0.0;

		std::string const window_name = "Marking images";
		namedWindow(window_name, WINDOW_NORMAL);
		resizeWindow(window_name, 1280, 720);
		imshow(window_name, frame);
		moveWindow(window_name, 0, 0);
		setMouseCallback(window_name, callback_mouse_click);

		bool next_by_click = false;
		bool marks_changed = false;

		int old_trackbar_value = -1, trackbar_value = 0;
		std::string const trackbar_name = "image num";
		int tb_res = createTrackbar(trackbar_name, window_name, &trackbar_value,
			image_list_count);

		int old_current_obj_id = -1, current_obj_id = 0;
		std::string const trackbar_name_2 = "object id";
		int const max_object_id = (synset_txt.size() > 0) ? synset_txt.size() : 20;
		int const min_object_id = (synset_txt.size() < 0) ? synset_txt.size() : 0;
		int tb_res_2 = createTrackbar(trackbar_name_2, window_name, &current_obj_id,
			max_object_id);
		Mat originalImage;
		int originalImaleLoadedId = -1;
		do {
			// trackbar_value = min(max(0, trackbar_value),
			// (int)jpg_filenames_path.size() - 1);

			// selected new image
			if (old_trackbar_value != trackbar_value || exit_flag || zooming || mousePanning) {
				trackbar_value =
					min(max(0, trackbar_value), (int)jpg_filenames_path.size() - 1);
				setTrackbarPos(trackbar_name, window_name, trackbar_value);
				frame(Rect(0, 0, frame.cols, preview.rows)) = Scalar::all(0);
				move_rect_id = -1;
				move_edge_id = -1;
				// save current coords
				if (old_trackbar_value >=
					0 && !zooming && !mousePanning) // && current_coord_vec.size() > 0) // Yolo v2 can processes
					   // background-image without objects
				{
					try {
						std::string const jpg_filename = jpg_filenames[old_trackbar_value];
						std::string const filename_without_ext =
							jpg_filename.substr(0, jpg_filename.find_last_of("."));
						std::string const txt_filename = filename_without_ext + ".txt";
						std::string const txt_filename_path =
							images_path + "/" + txt_filename;

						std::cout << "txt_filename_path = " << txt_filename_path
							<< std::endl;

						std::ofstream ofs(txt_filename_path,
							std::ios::out | std::ios::trunc);
						ofs << std::fixed;

						// store coords to [image name].txt
						for (auto &i : current_coord_vec) {
							float const relative_center_x =
								(float)(i.abs_rect.x + i.abs_rect.width / 2) /
								full_image_roi.cols;
							float const relative_center_y =
								(float)(i.abs_rect.y + i.abs_rect.height / 2) /
								full_image_roi.rows;
							float const relative_width =
								(float)i.abs_rect.width / full_image_roi.cols;
							float const relative_height =
								(float)i.abs_rect.height / full_image_roi.rows;

							if (relative_width <= 0)
								continue;
							if (relative_height <= 0)
								continue;
							if (relative_center_x <= 0)
								continue;
							if (relative_center_y <= 0)
								continue;

							ofs << i.id << " " << relative_center_x << " "
								<< relative_center_y << " " << relative_width << " "
								<< relative_height << std::endl;
						}

						// store [path/image name.jpg] to train.txt
						auto it =
							std::find(difference_filenames.begin(),
								difference_filenames.end(), filename_without_ext);
						if (it != difference_filenames.end()) {
							ofs_train << images_path << "/" << jpg_filename << std::endl;
							ofs_train.flush();

							size_t new_size = std::remove(difference_filenames.begin(),
								difference_filenames.end(),
								filename_without_ext) -
								difference_filenames.begin();
							difference_filenames.resize(new_size);
						}
					}
					catch (...) {
						std::cout << " Exception when try to write txt-file \n";
					}
				}
				bool loadPreviewImagesToCache = false;
				// show preview images
				for (size_t i = 0; i < preview_number &&
					(i + trackbar_value) < jpg_filenames_path.size();
					++i) {
					Mat img = imread(jpg_filenames_path[trackbar_value + i]);
					// check if the image has been loaded successful to prevent crash
					if (originalImaleLoadedId != trackbar_value && !zooming && !mousePanning) {
						originalImage = imread(jpg_filenames_path[trackbar_value + i]);
						originalImaleLoadedId = trackbar_value;
						if (img.cols == 0) {
							continue;
						}
						original_img_size = originalImage.size();
						resize(img, preview, preview.size());
						previewImagesCache.clear();
						loadPreviewImagesToCache = true;
						previewImagesCache.push_back(preview.clone());
					}
					else {
						if (loadPreviewImagesToCache) {
							Mat previewImage = imread(jpg_filenames_path[trackbar_value + i]);
							// check if the image has been loaded successful to prevent crash 
							if (previewImage.cols == 0)
							{
								continue;
							}
							resize(previewImage, preview, preview.size());
							previewImagesCache.push_back(preview.clone());
						}
						else {
							if (i < previewImagesCache.size()) {
								preview = previewImagesCache.at(i);
								//resize(previewImage, preview, preview.size());
							}
						}
					}

					int const x_shift = i * preview.cols + prev_img_rect.width;
					Rect rect_dst(Point2i(x_shift, 0), preview.size());
					Mat dst_roi = frame(rect_dst);
					preview.copyTo(dst_roi);
					// rectangle(frame, rect_dst, Scalar(200, 150, 200), 2);
					putText(dst_roi, jpg_filenames[trackbar_value + i], Point2i(0, 10),
						FONT_HERSHEY_COMPLEX_SMALL, 0.5, Scalar::all(255));

					if (i == 0) {
						imgHOrig = originalImage.rows;
						imgWOrig = originalImage.cols;

						optflow_img = img;
						//resize(img, full_image, full_rect_dst.size());
						full_image = resizeKeepAspectRatio(img, full_rect_dst.size(), cv::Scalar(0, 0, 0));
						full_image.copyTo(full_image_roi);
						current_img_size = img.size();
						scaleFactor = (float)original_img_size.width / current_img_size.width;
						resizeFactorImage = (float)full_rect_dst.width / original_img_size.width;

						// pad to cursor location:
						int const x_inside = std::min((int)x_end, full_image_roi.cols);
						int const y_inside = std::min(std::max(0, y_end - (int)prev_img_rect.height), full_image_roi.rows);
						float const relative_center_x = ((float)(x_inside) / full_image_roi.cols);
						float const relative_center_y = ((float)(y_inside) / full_image_roi.rows);

						// read bounding boxes from file
						if (!zooming && !mousePanning) {

							try {
								std::string const jpg_filename = jpg_filenames[trackbar_value];
								std::string const txt_filename =
									jpg_filename.substr(0, jpg_filename.find_last_of(".")) +
									".txt";
								// std::cout << (images_path + "/" + txt_filename) << std::endl;
								std::ifstream ifs(images_path + "/" + txt_filename);
								if (copy_previous_marks)
									copy_previous_marks = false;
								else if (tracker_copy_previous_marks) {
									tracker_copy_previous_marks = false;
									current_coord_vec = tracker_optflow.tracking_flow(img, false);
								}
								else
									current_coord_vec.clear();

								for (std::string line; getline(ifs, line);) {
									std::stringstream ss(line);
									coord_t coord;
									coord.id = -1;
									ss >> coord.id;
									if (coord.id < 0)
										continue;
									float relative_coord[4] = {
										-1, -1, -1,
										-1 }; // rel_center_x, rel_center_y, rel_width, rel_height
									for (size_t i = 0; i < 4; i++)
										if (!(ss >> relative_coord[i]))
											continue;
									for (size_t i = 0; i < 4; i++)
										if (relative_coord[i] < 0)
											continue;
									coord.abs_rect.x = (relative_coord[0] - relative_coord[2] / 2) *
										(float)full_image_roi.cols;
									coord.abs_rect.y = (relative_coord[1] - relative_coord[3] / 2) *									(float)full_image_roi.rows;
									coord.abs_rect.width =
										relative_coord[2] * (float)full_image_roi.cols;
									coord.abs_rect.height =
										relative_coord[3] * (float)full_image_roi.rows;

									current_coord_vec.push_back(coord);
									sort(current_coord_vec.begin(), current_coord_vec.end(), cmp);
								}


								int count_multi_label = 0;
								for (int i = 0; i < current_coord_vec.size() - 1; i++) {
									auto &p = current_coord_vec.at(i);
									for (int j = i + 1; j < current_coord_vec.size(); j++) {
										auto &n = current_coord_vec.at(j);
										if (p.abs_rect == n.abs_rect)
											count_multi_label++;
									}
									p.multi_label = count_multi_label;
									count_multi_label = 0;
								}
							}
							catch (...) {
								std::cout << " Exception when try to read txt-file \n";
							}
						}
					}
					std::string const jpg_filename = jpg_filenames[trackbar_value + i];
					std::string const filename_without_ext =
						jpg_filename.substr(0, jpg_filename.find_last_of("."));
					// green check-mark on the preview image if there is a lebel txt-file
					// for this image
					if (!std::binary_search(difference_filenames.begin(),
						difference_filenames.end(),
						filename_without_ext)) {
						line(dst_roi, Point2i(80, 88), Point2i(85, 93), Scalar(20, 70, 20),
							5);
						line(dst_roi, Point2i(85, 93), Point2i(93, 85), Scalar(20, 70, 20),
							5);

						line(dst_roi, Point2i(80, 88), Point2i(85, 93),
							Scalar(50, 200, 100), 2);
						line(dst_roi, Point2i(85, 93), Point2i(93, 85),
							Scalar(50, 200, 100), 2);
					}
				}
				if (!zooming && !mousePanning) {
					std::cout << " trackbar_value = " << trackbar_value << std::endl;
				}
				old_trackbar_value = trackbar_value;

				marks_changed = false;
				zooming = false;
				rectangle(frame, prev_img_rect, Scalar(100, 100, 100), CV_FILLED);
				rectangle(frame, next_img_rect, Scalar(100, 100, 100), CV_FILLED);
			}

			trackbar_value =
				min(max(0, trackbar_value), (int)jpg_filenames_path.size() - 1);

			// highlight prev img
			for (size_t i = 0; i < preview_number &&
				(i + trackbar_value) < jpg_filenames_path.size();
				++i) {
				int const x_shift = i * preview.cols + prev_img_rect.width;
				Rect rect_dst(Point2i(x_shift, 0),
					Size(preview.cols - 2, preview.rows));
				Scalar color(100, 70, 100);
				if (i == 0)
					color = Scalar(250, 120, 150);
				if (y_end < preview.rows &&
					i == (x_end - prev_img_rect.width) / preview.cols)
					color = Scalar(250, 200, 200);
				rectangle(frame, rect_dst, color, 2);
			}

			if (undo) {
				undo = false;
				if (current_coord_vec.size() > 0) {
					full_image.copyTo(full_image_roi);
					current_coord_vec.pop_back();
				}
			}

			// marking is completed (left mouse button is OFF)
			if (selected) {
				selected = false;
				full_image.copyTo(full_image_roi);

				if (y_end < preview.rows && x_end > prev_img_rect.width &&
					x_end < (full_image.cols - prev_img_rect.width) &&
					y_start < preview.rows) {
					int const i = (x_end - prev_img_rect.width) / preview.cols;
					trackbar_value += i;
				}
				else if (y_end >= preview.rows) {
					if (next_by_click) {
						++trackbar_value;
						current_coord_vec.clear();
					}

					if (x_size >= 10 || y_size >= 10) {
						Rect selected_rect(
							Point2i((int)max(min(x_start, x_end), std::atomic<int>(0)),
							(int)max(min(y_start, y_end), std::atomic<int>(0))),
							Size(x_size, y_size));
						selected_rect.width =
							selected_rect.x + selected_rect.width >= frame.cols
							? frame.cols - selected_rect.x - 1
							: selected_rect.width;
						selected_rect.height =
							selected_rect.y + selected_rect.height >= frame.rows
							? frame.rows - selected_rect.y - 1
							: selected_rect.height;
						selected_rect &= full_rect_dst;
						selected_rect.y -= (int)prev_img_rect.height;

						coord_t coord;
						coord.abs_rect = selected_rect;
						coord.id = current_obj_id;
						current_coord_vec.push_back(coord);
						sort(current_coord_vec.begin(), current_coord_vec.end(), cmp);
						marks_changed = true;
					}
				}
			}

			std::string current_synset_name;
			if (current_obj_id < synset_txt.size())
				current_synset_name = "   - " + synset_txt[current_obj_id];

			// show X and Y coords of mouse
			if (show_mouse_coords) {
				full_image.copyTo(full_image_roi);
				int const x_inside = std::min((int)x_end, full_image_roi.cols);
				int const y_inside =
					std::min(std::max(0, y_end - (int)prev_img_rect.height),
						full_image_roi.rows);
				float const relative_center_x = (float)(x_inside) / full_image_roi.cols;
				float const relative_center_y = (float)(y_inside) / full_image_roi.rows;
				int const abs_x = relative_center_x * current_img_size.width;
				int const abs_y = relative_center_y * current_img_size.height;
				char buff[100];
				snprintf(buff, 100, "Abs: %d x %d    Rel: %.3f x %.3f", abs_x, abs_y,
					relative_center_x, relative_center_y);
				// putText(full_image_roi, buff, Point2i(800, 20), FONT_HERSHEY_SIMPLEX,
				// 0.7, Scalar(50, 10, 10), 3);
				putText(full_image_roi, buff, Point2i(800, 20), FONT_HERSHEY_SIMPLEX,
					0.7, Scalar(100, 50, 50), 2);
				putText(full_image_roi, buff, Point2i(800, 20), FONT_HERSHEY_SIMPLEX,
					0.7, Scalar(220, 120, 120), 1);
			}
			else {
				full_image.copyTo(full_image_roi);
				// std::string text = "Show mouse coordinates - press M";
				// putText(full_image_roi, text, Point2i(800, 20), FONT_HERSHEY_SIMPLEX,
				// 0.7, Scalar(100, 50, 50), 2); putText(full_image_roi, text,
				// Point2i(800, 20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(220, 120, 120),
				// 1);
			}

			// marking is in progress (left mouse button is ON)
			if (draw_select) {
				if (add_id_img != 0)
					trackbar_value += add_id_img;
				if (y_start >= preview.rows) {
					// full_image.copyTo(full_image_roi);
					Rect selected_rect(
						Point2i(max(0, (int)min(x_start, x_end)),
							max(preview.rows, (int)min(y_start, y_end))),
						Point2i(max(x_start, x_end), max(y_start, y_end)));
					rectangle(frame, selected_rect, Scalar(150, 200, 150));

					if (show_mark_class) {
						putText(frame, std::to_string(current_obj_id) + current_synset_name,
							selected_rect.tl() + Point2i(2, 22), FONT_HERSHEY_SIMPLEX,
							0.8, Scalar(150, 200, 150), 2);
					}
				}
			}

			// Draw crosshair 화살표
			{
				const int offset = preview.rows; // Vertical offset

				// Only draw crosshair, if mouse is over image area
				if (y_end >= offset) {
					const bool bit_high = true;
					const bool bit_low = false;
					const int mouse_offset = 25;
					const int ver_min = draw_select ? std::min(x_end - mouse_offset,
						x_start - mouse_offset)
						: x_end - mouse_offset;
					const int ver_max = draw_select ? std::max(x_end + mouse_offset,
						x_start + mouse_offset)
						: x_end + mouse_offset;
					const int hor_min = draw_select ? std::min(y_end - mouse_offset,
						y_start - mouse_offset)
						: y_end - mouse_offset;
					const int hor_max = draw_select ? std::max(y_end + mouse_offset,
						y_start + mouse_offset)
						: y_end + mouse_offset;

					// Draw crosshair onto empty canvas (draws high bits on
					// low-bit-canvas)
					cv::Mat crosshair_mask(frame.size(), CV_8UC1, cv::Scalar(bit_low));
					cv::line(crosshair_mask, cv::Point(0, y_end),
						cv::Point(ver_min, y_end),
						cv::Scalar(bit_high)); // Horizontal, left to mouse
					cv::line(crosshair_mask, cv::Point(ver_max, y_end),
						cv::Point(crosshair_mask.size().width, y_end),
						cv::Scalar(bit_high)); // Horizontal, mouse to right
					cv::line(crosshair_mask, cv::Point(x_end, offset),
						cv::Point(x_end, std::max(offset, hor_min)),
						cv::Scalar(bit_high)); // Vertical, top to mouse
					cv::line(crosshair_mask, cv::Point(x_end, hor_max),
						cv::Point(x_end, crosshair_mask.size().height),
						cv::Scalar(bit_high)); // Vertical, mouse to bottom

			   // Draw crosshair onto frame copy
					cv::Mat crosshair_frame(frame.size(), frame.type());
					frame.copyTo(crosshair_frame);
					cv::bitwise_not(crosshair_frame, crosshair_frame, crosshair_mask);

					// Fade-in frame copy with crosshair into original frame (for alpha)
					const double alpha = 0.7;
					cv::addWeighted(crosshair_frame, alpha, frame, 1 - alpha, 0.0, frame);
				}
			}

			// remove all labels from this image
			if (clear_marks == true) {
				clear_marks = false;
				marks_changed = true;
				full_image.copyTo(full_image_roi);

				current_coord_vec.clear();
			}

			if (old_current_obj_id != current_obj_id) {
				full_image.copyTo(full_image_roi);
				old_current_obj_id = current_obj_id;
				setTrackbarPos(trackbar_name_2, window_name, current_obj_id);
			}

			int selected_id = -1;
			// draw all labels
			// for (auto &i : current_coord_vec)

			int count = 0;
			std::vector<float> tmp;
			std::vector<int> num;
			for (size_t k = 0; k < current_coord_vec.size(); ++k) {
				auto &i = current_coord_vec.at(k);
				// selected rect
				if ((i.abs_rect.x + 0.9*i.abs_rect.width) < x_start && (i.abs_rect.x + 1.1*i.abs_rect.width) >= x_start &&
					(i.abs_rect.y + 0.9*i.abs_rect.height + preview.rows) < y_start &&
					(i.abs_rect.y + 1.1*i.abs_rect.height + preview.rows) >= y_start && left_button_down) {

					isbr = true;
					draw_select = false;
					selected = false;
					if (selected_id < 0)
						selected_id = k;

				}
				else if (i.abs_rect.x < x_end && (i.abs_rect.x + i.abs_rect.width) > x_end &&
					(i.abs_rect.y + preview.rows) < y_end &&
					(i.abs_rect.y + i.abs_rect.height + preview.rows) > y_end) {
					if (selected_id < 0)
						selected_id = k;
				}
				if (i.abs_rect.x < mousex && (i.abs_rect.x + i.abs_rect.width) > mousex && //box in box
					(i.abs_rect.y + preview.rows) < y_end &&
					(i.abs_rect.y + i.abs_rect.height + preview.rows) > mousey) {
					selected_id = k;
					tmp.push_back((mousey - i.abs_rect.y + i.abs_rect.height + preview.rows) + (mousex - i.abs_rect.x));
					num.push_back(k);
					if (tmp.size() > 1) {
						selected_id = num[min_element(tmp.begin(), tmp.end()) - tmp.begin()];
					}
				}
			}
			//std::cout << *min_element(tmp.begin(), tmp.end()) << std::endl;
			//if (change_image == 1) { //zoom in
			//	change_image = 0;
			//	zoom_out_image.push_back(full_image);
			//	full_image_store = zoom_out_image[0];
			//	Rect bounds(0, 0, full_image.cols, full_image.rows);

			//	int width = zoomRec, height = zoomRec;
			//	if (mousex >= 0 && mousex <= 1280 && mousey >= 100 && mousey <= 820) {
			//		int ptoX = mousex - (zoomRec / 2), ptoY = mousey - (zoomRec / 2);

			//		/*Verifica que el ROI este dentro de la la imagen*/
			//		if ((mousex + (zoomRec / 2)) > full_image.size().width)
			//			width = width - ((mousex + (zoomRec / 2)) - full_image.size().width);

			//		if ((mousey + (zoomRec / 2)) > full_image.size().height)
			//			height = height - ((mousey + (zoomRec / 2)) - full_image.size().height);

			//		if ((mousex - (zoomRec / 2)) < 0)
			//			ptoX = 0;

			//		if ((mousey - (zoomRec / 2)) < 0)
			//			ptoY = 0;


			//		Rect roi = Rect(ptoX, ptoY, width, height);
			//		Mat imagen_roi = full_image(roi & bounds);
			//		resize(imagen_roi, imagen_roi, Size(full_image.size().width, full_image.size().height), 0, 0, cv::INTER_AREA);
			//		full_image = imagen_roi;
			//	}
			//}
			//if (change_image == 2) { //zoom out
			//	change_image = 0;

			//	if (zoom_out_image.empty() || zoom_out_image.size() < 2) {
			//		full_image = full_image_store;
			//	}
			//	else {
			//		full_image = zoom_out_image[zoom_out_image.size() - 1];
			//		zoom_out_image.pop_back();
			//	}

			//}
			if (selected_id >= 0) {

				// remove selected rect
				if (delete_selected) {
					delete_selected = false;
					current_coord_vec.erase(current_coord_vec.begin() + selected_id);
				}

				// resize selected rect
				auto &current_coord_vec_selected =
					current_coord_vec[selected_id].abs_rect;
				if (resize_u_inc) {
					resize_u_inc = false;
					if (current_coord_vec_selected.y >= 0) {
						current_coord_vec_selected.height += 1.0;
						current_coord_vec_selected.y -= 1.0;
					}
				}
				if (resize_u_dec) {
					resize_u_dec = false;
					if (current_coord_vec_selected.height > 10) {
						current_coord_vec_selected.height -= 1.0;
						current_coord_vec_selected.y += 1.0;
					}
				}
				if (resize_l_inc) {
					resize_l_inc = false;
					if (current_coord_vec_selected.x >= 0) {
						current_coord_vec_selected.width += 1.0;
						current_coord_vec_selected.x -= 1.0;
					}
				}
				if (resize_l_dec) {
					resize_l_dec = false;
					if (current_coord_vec_selected.width > 10) {
						current_coord_vec_selected.width -= 1.0;
						current_coord_vec_selected.x += 1.0;
					}
				}
				if (resize_d_inc) {
					resize_d_inc = false;
					current_coord_vec_selected.height += 1.0;
				}
				if (resize_d_dec) {
					resize_d_dec = false;
					if (current_coord_vec_selected.height > 10)
						current_coord_vec_selected.height -= 1.0;
				}
				if (resize_r_inc) {
					resize_r_inc = false;
					current_coord_vec_selected.width += 1.0;
				}
				if (resize_r_dec) {
					resize_r_dec = false;
					if (current_coord_vec_selected.width > 10)
						current_coord_vec_selected.width -= 1.0;
				}
				if (copy_current_box) {
					copy_current_box = false;
					coord_t new_box;
					new_box.id = current_coord_vec[selected_id].id;
					new_box.abs_rect = current_coord_vec_selected;
					current_coord_vec.push_back(new_box);
					sort(current_coord_vec.begin(), current_coord_vec.end(), cmp);
				}
				if (change_class_id == 1) {
					change_class_id = 0;
					//current_coord_vec[selected_id].id += 1;
					current_coord_vec[selected_id].id = current_obj_id;
					if (current_coord_vec[selected_id].id >= max_object_id)
						current_coord_vec[selected_id].id = 0;
				}
				if (change_class_id == 2) {
					change_class_id = 0;
					current_coord_vec[selected_id].id += 1;
					//current_coord_vec[selected_id].id = current_obj_id;
					if (current_coord_vec[selected_id].id >= max_object_id)
						current_coord_vec[selected_id].id = 0;
				}
				if (change_class_id == 3) {
					change_class_id = 0;
					current_coord_vec[selected_id].id -= 1;
					//current_coord_vec[selected_id].id = current_obj_id;
					if (current_coord_vec[selected_id].id <= min_object_id)
						current_coord_vec[selected_id].id = 0;
				}

				current_coord_vec_selected.x =
					current_coord_vec_selected.x < 0 ? 0 : current_coord_vec_selected.x;
				current_coord_vec_selected.y =
					current_coord_vec_selected.y < 0 ? 0 : current_coord_vec_selected.y;
				current_coord_vec_selected.width =
					current_coord_vec_selected.x + current_coord_vec_selected.width >=
					frame.cols
					? frame.cols - current_coord_vec_selected.x - 1
					: current_coord_vec_selected.width;
				current_coord_vec_selected.height =
					current_coord_vec_selected.y + preview.rows +
					current_coord_vec_selected.height >=
					frame.rows
					? frame.rows - current_coord_vec_selected.y - preview.rows - 1
					: current_coord_vec_selected.height;
				auto o = frame(
					Rect(static_cast<int>(current_coord_vec_selected.x),
						static_cast<int>(current_coord_vec_selected.y + preview.rows),
						static_cast<int>(current_coord_vec_selected.width),
						static_cast<int>(current_coord_vec_selected.height)));
				if (mini_window) {
					namedWindow(window_name + " object", WINDOW_NORMAL);
					moveWindow(window_name + " object", 0, 0);
					imshow(window_name + " object", o);
					double multiply_ratio = min(o.cols / 360.0, o.rows / 360.0);
					resizeWindow(window_name + " object", o.cols / multiply_ratio,
						o.rows / multiply_ratio);
				}
			}
			else {
				if (mini_window) {
					destroyWindow(window_name + " object");
				}
			}
			selected_id = -1;
			tmp.clear();
			num.clear();
			for (size_t k = 0; k < current_coord_vec.size(); ++k) {
				auto &i = current_coord_vec.at(k);
				std::string synset_name;
				if (i.id < synset_txt.size())
					synset_name = " - " + synset_txt[i.id];

				int offset = i.id * 30;
				int red = (offset + 40) % 255 * ((i.id + 2) % 3);
				int green = (offset + 190) % 255 * ((i.id + 1) % 3);
				int blue = (offset + 100) % 255 * ((i.id + 0) % 3);
				Scalar color_rect(red, green,
					blue); // Scalar color_rect(100, 200, 100);

  // selected rect
				//if (i.abs_rect.x < x_end && (i.abs_rect.x + i.abs_rect.width) > x_end &&
				//	(i.abs_rect.y + preview.rows) < y_end &&
				//	(i.abs_rect.y + i.abs_rect.height + preview.rows) > y_end) {
				//	if (selected_id < 0) {
				//		color_rect = Scalar(100, 200, 300);
				//		selected_id = k;
				//		rectangle(full_image_roi, i.abs_rect, color_rect, mark_line_width);
				//	}
				//}
				if (i.abs_rect.x < mousex && (i.abs_rect.x + i.abs_rect.width) > mousex && //box in box
					(i.abs_rect.y + preview.rows) < y_end &&
					(i.abs_rect.y + i.abs_rect.height + preview.rows) > mousey) {
					selected_id = k;
					tmp.push_back((mousey - i.abs_rect.y + i.abs_rect.height + preview.rows) + (mousex - i.abs_rect.x));
					num.push_back(k);
					if (tmp.size() >= 1) {
						selected_id = num[min_element(tmp.begin(), tmp.end()) - tmp.begin()];
						//if(tmp.size()==1) selected_id = num[0];
						color_rect = Scalar(100, 200, 300);
						//selected_id = k;
						rectangle(full_image_roi, i.abs_rect, color_rect, mark_line_width);
					}

				}
				if (show_mark_class&&i.id != 0) {
					if (i.multi_label != 0) {
						putText(full_image_roi, std::to_string(i.id) + synset_name,
							i.abs_rect.tl() + Point2f(2, 22 * 1.5*(0.5 + i.multi_label)), FONT_HERSHEY_SIMPLEX, 0.8, // �����ϴ°�
							color_rect, 2);
					}
					else {
						putText(full_image_roi, std::to_string(i.id) + synset_name,
							i.abs_rect.tl() + Point2f(2, 22), FONT_HERSHEY_SIMPLEX, 0.8,
							color_rect, 2);
					}
				}

				if (i.id == 0) {
					cv::Mat roi = full_image_roi(i.abs_rect);
					cv::Mat rec(roi.size(), CV_8UC3, cv::Scalar(255, 255, 255));
					cv::addWeighted(rec, 0.5, roi, 0.5, 0.0, roi);
					if (show_box) rectangle(full_image_roi, i.abs_rect, cv::Scalar(0, 0, 0),
						mark_line_width);
				}
				else
					if (show_box) rectangle(full_image_roi, i.abs_rect, color_rect, mark_line_width);
			}

			// show moving rect
			if (right_button_click == true && mousePanning == false) {
				if (move_rect_id < 0 && mousePanning == false)
					move_rect_id = selected_id;

				int x_delta = x_end - x_start;
				int y_delta = y_end - y_start;
				auto rect = current_coord_vec[move_rect_id].abs_rect;
				rect.x += x_delta;
				rect.y += y_delta;

				Scalar color_rect = Scalar(300, 200, 100);
				rectangle(full_image_roi, rect, color_rect, mark_line_width);
			}

			// complete moving label rect
			if (move_rect && move_rect_id >= 0) {
				int x_delta = x_end - x_start;
				int y_delta = y_end - y_start;
				current_coord_vec[move_rect_id].abs_rect.x += x_delta;
				current_coord_vec[move_rect_id].abs_rect.y += y_delta;
				move_rect = false;
				move_rect_id = -1;
			}

			if (isbr&&left_button_down) {
				if (move_edge_id < 0)
					move_edge_id = selected_id;

				int x_delta = x_end - current_coord_vec[move_edge_id].abs_rect.x;
				int y_delta = y_end - current_coord_vec[move_edge_id].abs_rect.y;
				auto rect = current_coord_vec[move_edge_id].abs_rect;
				if (x_delta >= 0 && y_delta >= 0) {
					rect.width = x_delta;
					rect.height = y_delta - preview.rows;
				}
				Scalar color_rect = Scalar(300, 200, 100);
				rectangle(full_image_roi, rect, color_rect, mark_line_width);


			}
			if (move_edge&&move_edge_id >= 0) {
				int x_delta = x_end - current_coord_vec[move_edge_id].abs_rect.x;
				int y_delta = y_end - current_coord_vec[move_edge_id].abs_rect.y;
				if (x_delta >= 0 && y_delta >= 0) {
					current_coord_vec[move_edge_id].abs_rect.width = x_delta;
					current_coord_vec[move_edge_id].abs_rect.height = y_delta - preview.rows;
				}
				move_edge = false;
				move_edge_id = -1;
			}

			if (next_by_click) {
				putText(full_image_roi, "Mode: 1 mark per image (next by click)",
					Point2i(850, 20), FONT_HERSHEY_SIMPLEX, 0.6,
					Scalar(50, 170, 100), 2);
			}

			{
				std::string const obj_str =
					"Object id: " + std::to_string(current_obj_id) +
					current_synset_name;
				if (show_box) {
					putText(full_image_roi, obj_str, Point2i(0, 21), FONT_HERSHEY_DUPLEX,
						0.8, Scalar(10, 50, 10), 3);
					putText(full_image_roi, obj_str, Point2i(0, 21), FONT_HERSHEY_DUPLEX,
						0.8, Scalar(20, 120, 60), 2);
					putText(full_image_roi, obj_str, Point2i(0, 21), FONT_HERSHEY_DUPLEX,
						0.8, Scalar(50, 200, 100), 1);
				}
			}

			if (show_help) {
				putText(full_image_roi,
					"<- prev_img    -> next_img    c - clear_marks    n - "
					"one_object_per_img    0-9 - obj_id    m - show coords    ESC "
					"- exit",
					Point2i(0, 45), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(50, 10, 10),
					2);
				putText(
					full_image_roi,
					"w - line width   k - hide obj_name   p - copy previous   o - "
					"track objects   r - delete selected   R-mouse - move box", //   h -
																				//   disable
																				//   help",
					Point2i(0, 80), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(50, 10, 10), 2);
			}
			else {
				if (show_box) {
					putText(full_image_roi, "h - show help", Point2i(0, 45),
						FONT_HERSHEY_SIMPLEX, 0.6, Scalar(50, 10, 10), 2);
				}
			}

			// arrows
			{
				Scalar prev_arrow_color(200, 150, 100);
				Scalar next_arrow_color = prev_arrow_color;
				if (prev_img_rect.contains(Point2i(x_end, y_end)))
					prev_arrow_color = Scalar(220, 190, 170);
				if (next_img_rect.contains(Point2i(x_end, y_end)))
					next_arrow_color = Scalar(220, 190, 170);

				std::vector<Point> prev_triangle_pts = { Point(5, 50), Point(40, 90),  //◁
														Point(40, 10), Point(5, 50) };
				Mat prev_roi = frame(prev_img_rect);
				line(prev_roi, prev_triangle_pts[0], prev_triangle_pts[1],
					prev_arrow_color, 5);
				line(prev_roi, prev_triangle_pts[1], prev_triangle_pts[2],
					prev_arrow_color, 5);
				line(prev_roi, prev_triangle_pts[2], prev_triangle_pts[3],
					prev_arrow_color, 5);
				line(prev_roi, prev_triangle_pts[3], prev_triangle_pts[0],
					prev_arrow_color, 5);

				std::vector<Point> next_triangle_pts = { Point(10, 10), Point(10, 90), // ▷
														Point(45, 50), Point(10, 10) };
				Mat next_roi = frame(next_img_rect);
				line(next_roi, next_triangle_pts[0], next_triangle_pts[1],
					next_arrow_color, 5);
				line(next_roi, next_triangle_pts[1], next_triangle_pts[2],
					next_arrow_color, 5);
				line(next_roi, next_triangle_pts[2], next_triangle_pts[3],
					next_arrow_color, 5);
				line(next_roi, next_triangle_pts[3], next_triangle_pts[0],
					next_arrow_color, 5);
			}

			imshow(window_name, frame);
			if (show_mark_box) {
				mark_line_width = (int)current_mark_line_width;
			}
			if (!show_mark_box) {
				mark_line_width = 0;
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
				break; // exit after saving
			if (pressed_key == 27 || pressed_key == 1048603)
				exit_flag = true; // break;  // ESC - save & exit

			if (pressed_key >= '0' && pressed_key <= '9') {
				current_obj_id = pressed_key - '0'; // 0 - 9
				change_class_id = 1;
			}

			if (pressed_key >= 1048624 && pressed_key <= 1048633) {
				current_obj_id = pressed_key - 1048624; // 0 - 9
				change_class_id = 1;
			}
			if (pressed_key >= 7340032 && pressed_key <= 8060928) {
				switch (pressed_key) {
				case 7340032:
					current_obj_id = 10;
					break;
				case 7405568:
					current_obj_id = 11;
					break;
				case 7471104:
					current_obj_id = 12;
					break;
				case 7536640:
					current_obj_id = 13;
					break;
				case 7602176:
					current_obj_id = 14;
					break;
				case 7667712:
					current_obj_id = 15;
					break;
				case 7733248:
					current_obj_id = 16;
					break;
				case 7798784:
					current_obj_id = 17;
					break;
				case 7864320:
					current_obj_id = 18;
					break;
				case 7929856:
					current_obj_id = 19;
					break;
				case 7995392:
					current_obj_id = 20;
					break;
				case 8060928:
					current_obj_id = 21;
					break;
				}
				change_class_id = 1;
			}

			switch (pressed_key) {
				//   case 'z':		// z
			case 8:	// backspace �ڷΰ���
				undo = true;
				break;

			case 'a':
				resize_u_inc = true;
				break;
			case 's':
				resize_u_dec = true;
				break;
			case 'z':
				resize_d_inc = true;
				break;
			case 'x':
				resize_d_dec = true;
				break;
			case 'd':
				resize_l_inc = true;
				break;
			case 'f':
				resize_l_dec = true;
				break;
			case 'c':
				resize_r_inc = true;
				break;
			case 'v':
				resize_r_dec = true;
				break;
			case 'p':     // p
			case 1048688: // p
				copy_previous_marks = 1;
				++trackbar_value;
				break;

			case 'q':
				copy_current_box = 1;
				break;
			case '+':
				change_class_id = 2;
				break;
			case '-':
				change_class_id = 3;
				break;
			case 'o':     // o
			case 1048687: // o
				tracker_copy_previous_marks = 1;
				++trackbar_value;
				break;

			case 32:      // SPACE
			case 1048608: // SPACE
				++trackbar_value;
				break;

			case 13: // enter
				++trackbar_value;
				break;

			case 2424832: // <-
			case 65361:   // <-
			case 91:      // [
				--trackbar_value;
				break;
			case 2555904: // ->
			case 65363:   // ->
			case 93:      // ]
				++trackbar_value;
				break;
			case '/':     // /
			case 1048675: // /
				clear_marks = true;
				break;
			case 'm':     // m
			case 1048685: // m
				show_mouse_coords = !show_mouse_coords;
				full_image.copyTo(full_image_roi);
				break;
			case 'n':     // n
			case 1048686: // n
				next_by_click = !next_by_click;
				full_image.copyTo(full_image_roi);
				break;
			case 'w':     // w
			case 1048695: // w
				mark_line_width = mark_line_width % MAX_MARK_LINE_WIDTH + 1;
				current_mark_line_width = (int)mark_line_width;
				break;
			case 'h':     // h
			case 1048680: // h
				show_help = !show_help;
				break;
			case 'k':
			case 1048683:
				show_mark_class = !show_mark_class;
				show_mark_box = !show_mark_box;
				break;
			case 'r':     // r
			case 1048690: // r
				delete_selected = true;
				break;
			case 2162688: // page on
				mini_window = true;
				break;
			case 2228224: //page down
				mini_window = false;
				destroyWindow(window_name + " object");
				break;
			case 3014656: //delete
				show_box = !show_box;
				show_mark_class = !show_mark_class;
				break;
			default:;
			}

			if (tracker_copy_previous_marks)
				tracker_optflow.update_tracking_flow(optflow_img, current_coord_vec);

			// if (pressed_key >= 0) std::cout << "pressed_key = " << (int)pressed_key
			// << std::endl;

		} while (true);

	}
	catch (std::exception &e) {
		std::cout << "exception: " << e.what() << std::endl;
	}
	catch (...) {
		std::cout << "unknown exception \n";
	}

	return 0;
}
