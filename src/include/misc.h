#include <opencv2/core/core.hpp>
#include <iostream>

#define BLACK 0
#define GREY 128
#define WHITE 255

inline void print_ocv_version()
{
    std::cout << "Project using OpenCV " << CV_VERSION_MAJOR << "." << CV_VERSION_MINOR << "." << CV_VERSION_REVISION << std::endl;
}

/* the function 'bulid_image_histogram' compute the intensity histogram af a given image.
 * INPUT  -> IgInPut : the input image.
 * OUTPUT -> array : the intensity histogram of the given image.
 */
void build_image_histogra(cv::Mat IgInPut, int * array) {
	/* if the image is continuous */
	if(IgInPut.isContinuous()) {
		unsigned char * pIgData = IgInPut.ptr(0);
		unsigned char * pIgDataEnd = pIgData + IgInPut.cols * IgInPut.rows;

		/* use the intensity of the image as index of the image
		 * histogram and increase the value in that position.
		 */
		while(pIgData < pIgDataEnd) {
			array[*pIgData]++;
			pIgData++;
		}

	} else {/* if the image isn't continuous */
		for(int i =0; i<IgInPut.rows; ++i) {
			unsigned char * pIgData = IgInPut.ptr(i);
			unsigned char * pIgDataEnd = pIgData + IgInPut.cols;

			/* use the intensity of the image as index of the image
			 * histogram and increase the value in that position.
			 */
			while(pIgData < pIgDataEnd) {
				array[*pIgData]++;
				pIgData++;
			}
		}
	}
}

/* the function 'otsu_global_thresholding_method' compute the
 * threshold with otsu global thresholding method.
 * INPUT  -> image_histogram : an image histogram.
 * OUTPUT -> threshold : a threshold computed with otsu global thresholding method.
 */
int otsu_global_thresholding_method(int *image_histogram) {
	int n =0, t =0;
	double variance =0;
	double cumulative_sum =0;
	double cumulative_mean =0;
	double global_intensity_mean =0;
	double between_class_variance =0;
	double normalized_histogram[256];

	for(int i =0; i<256; ++i) {
		normalized_histogram[i] = image_histogram[i];
		n += image_histogram[i];
	}

	/* normalize histogram */
	for(int i =0; i<256; ++i) {
		normalized_histogram[i] /= n;
		global_intensity_mean += i * normalized_histogram[i];
	}

	/* compute the threshold */
	for(int i =0; i<256;++i) {
		cumulative_mean += i * normalized_histogram[i];
		cumulative_sum += normalized_histogram[i];

		/* if the cumulative sum is between 0 and 1 compute the between class variance */
		if(0<cumulative_sum && cumulative_sum <1) {
			variance = pow((global_intensity_mean * cumulative_sum - cumulative_mean),2) / (cumulative_sum * (1 - cumulative_sum));
			
			if (variance>between_class_variance) {
				between_class_variance = variance;
				t = i;
			}
		}
	}

	return t;
}

/* the function 'apply_threshold' apply a threshold to a given image.
 * INPUT  -> IgInPut : the given image.
 *        -> threshold : the threshold.
 * OUTPUT -> IgOutPut : image to which the threshold has been applied.
 */
void apply_threshold(cv::Mat IgInPut, cv::Mat IgOutPut, int threshold) {
	for(int i =0; i<IgInPut.rows; ++i) {
		unsigned char * pIgInPut = IgInPut.ptr(i);
		unsigned char * pIgInPutEnd = pIgInPut + IgInPut.cols;

		unsigned char * pIgOutPut = IgOutPut.ptr(i);

		while(pIgInPut<pIgInPutEnd) {
			*pIgOutPut = (threshold < *pIgInPut) * WHITE;
			pIgInPut++;
			pIgOutPut++;
		}
	}
}

/* the function 'apply_otsu' apply otsu threshold at input image.
 * INPUT  -> IgInPut : the original image.
 * OUTPUT -> IgInPut : image with otsu applied.
 */
cv::Mat apply_otsu(cv::Mat IgInPut) {
	int threshold;

	/* create the array for the image histogram and initialize it at 0 */
	int image_histogram[256] ={};

	cv::cvtColor(IgInPut, IgInPut, cv::COLOR_RGB2GRAY);

	/* compute the image histogram of 'Ia' */
	build_image_histogra(IgInPut,image_histogram);

	/* compute the otsu */
	threshold = otsu_global_thresholding_method(image_histogram);
	
	/* apply threshold */
	apply_threshold(IgInPut,IgInPut,threshold);

	return IgInPut;
}


/* the function 'mix_image' marge together ImageWhite and ImageBlack with respect the ImageBlackWhite.
 * INPUT  -> ImageWhite : the image that will be applied with respect to white.
 *        -> ImageBlack : the image that will be applied with respect to black.
 		  -> ImageBlackWhite : threshold image.
 * OUTPUT -> ImageMix : the merged image.
 */
void mix_image(cv::Mat ImageWhite, cv::Mat ImageBlack, cv::Mat ImageBlackWhite, cv::Mat ImageMix) {
	for(int i =0; i<ImageBlack.rows; ++i) {
		unsigned char * pIgWh = ImageWhite.ptr(i);

		unsigned char * pIgBl = ImageBlack.ptr(i);
		unsigned char * pIgBlEnd = pIgBl + ImageBlack.cols;

		unsigned char * pIgMi = ImageMix.ptr(i);

		unsigned char * pIgBlWh = ImageBlackWhite.ptr(i);

		while(pIgBl < pIgBlEnd) {
			/* if the value in '*pIgBlWh' is 0 (black) put in
			 * '*pIgMi' the value in '*pIgBl' otherwise '*pIgWh'
			 */
			*pIgMi = ((*pIgBlWh)==BLACK) ? *pIgBl : *pIgWh;
			pIgWh++;
			pIgBl++;
			pIgMi++;
			pIgBlWh++;
		}
	}
}

/* the function 'add_span' adds a border of 'span_size' pixel to each side of the input image
 * INPUT  -> IgInPut : the original image.
 *        -> span_size : the size of the border.
 * OUTPUT -> IgSpn : the image with the border
 */
cv::Mat add_span(cv::Mat IgInPut, int span_size) {

	/* create the output image with the correct shape */
	cv::Mat IgSpn = cv::Mat(IgInPut.rows+span_size*2,IgInPut.cols+span_size*2,CV_8U);

	IgSpn = 0;

	for(int i =0; i<IgInPut.rows; ++i) {
		unsigned char * pIgInPut = IgInPut.ptr(i);
		unsigned char * pIgInPutEnd = pIgInPut + IgInPut.cols;

		unsigned char * pIgSpn = IgSpn.ptr(i+span_size) + span_size;

		/* copy the original image in the center of the new one */
		while(pIgInPut<pIgInPutEnd) {
			*pIgSpn = *pIgInPut;
			pIgInPut++;
			pIgSpn++;
		}
	}

	return IgSpn;
}

/* the function 'find_blob' searches for all the points that make up the blob starting from x and y.
 * INPUT  -> IgInPut : the original image.
 *        -> x : abscissa of the starting point of the blob.
 *		  -> y : ordinate of the starting point of the blob.
 * OUTPUT -> Blob : an object containing all the blob information. See also class comment.
 */
std::vector<cv::Point2i> find_blob(cv::Mat IgInPut, int x, int y) {
	int b0x = x, b0y = y;
	int bx = x, by = y;
	int clock_pos = 0;

	/* clock phase, all possible points around the center */
	int clock_bx[8] = {-1,-1,-1,0,1,1,1,0};
	int clock_by[8] = {-1,0,1,1,1,0,-1,-1};
	/* start clock positions */
	int array[8] = {6,0,0,2,2,4,4,6};
	std::vector<cv::Point2i> blob;

	IgInPut.at<unsigned char>(b0x, b0y) = GREY;

	blob.push_back(cv::Point2i(b0y-1,b0x-1));

	do {
		/* try all eight clock phase */
		for(int i=0;i<8;i++) {
			/* if it isn't black then it is a part of a blob */
			if(IgInPut.at<unsigned char>(bx+clock_bx[clock_pos],by+clock_by[clock_pos])!=0) {
				/* update position */
				bx = bx + clock_bx[clock_pos];
				by = by + clock_by[clock_pos];

				if(IgInPut.at<unsigned char>(bx,by)==WHITE)
					/* add coordinates to the blob */
					blob.push_back(cv::Point2i(by-1,bx-1));
				
				/* mark the position with the color GREY */
				IgInPut.at<unsigned char>(bx, by) = GREY;

				/* update the starting point of the next research */
				clock_pos = array[clock_pos];
				break;
				
			} else
				/* try next clock phase */
				clock_pos = (clock_pos + 1) % 8;
		}
	/* loop until it reach the starting point */
	} while(b0x != bx || b0y != by);

	return blob;
}

/* the function 'find_contours' takes an image as input and finds all the contours.
 * INPUT  -> IgInPut : the original image.
 * INPUT  -> lower_bound : the lower bound of the blobs perimeter.
 * INPUT  -> upper_bound : the upper bound of the blobs perimeter.
 * INPUT  -> draw : true draw the boundary, false no (default: false).
 * OUTPUT -> IgOutPut : image with highlighted contours.
 */
std::vector<std::vector<cv::Point2i> >find_contours(cv::Mat IgInPut,cv::Mat IgOutPut,int lower_bound=0, int upper_bound=0, bool draw=false) {

	std::vector<cv::Point2i> b;
	std::vector<std::vector<cv::Point2i> > blobs;
	cv::Vec3b color(0,0,255);

	int threshold;
	bool flag = true;

	/* create the array for the image histogram and initialize it at 0 */
	int image_histogram[256] ={};

	cv::Mat IgGry = IgInPut.clone();
	cv::cvtColor(IgInPut, IgGry, cv::COLOR_RGB2GRAY);

	/* compute the image histogram of 'IgGry' */
	build_image_histogra(IgGry,image_histogram);

	/* compute the otsu of 'IgGry' */
	threshold = otsu_global_thresholding_method(image_histogram);

	/* apply the threshold to 'IgGry' */
	apply_threshold(IgGry,IgGry,128);
	
	/* add the span to 'IgGry' */
	cv::Mat IgSpn = add_span(IgGry,1);

	/* start to search the blob */
	for(int i=1;i<IgSpn.rows-1;i++) {
		for(int j=1;j<IgSpn.cols-1;j++) {

			/* if it's white it is a starting point of a blob */
			if(flag && IgSpn.at<unsigned char>(i,j)==WHITE) {
				/* find the blob from the starting point 'i' and 'j' */
				b = find_blob(IgSpn,i,j);
				/* keeps only the blobs with a perimeter between 100 and 200 pixels */
				if(b.size() >= lower_bound && (upper_bound==0 || b.size()<=upper_bound))
					blobs.push_back(b);
			}

			/* if it is not white or black then it is a pixel of a blob already
			 * found so I do not have to look for it anymore, 'flag = false'.
			 */
			if(IgSpn.at<unsigned char>(i,j)!=BLACK && IgSpn.at<unsigned char>(i,j)!=WHITE)
				flag = false;

			/* if it finds black it is out of a blob, then it can start the search for blobs again. */
			if(IgSpn.at<unsigned char>(i,j)==BLACK)
				flag = true;
			}
	}

	/* for each blob it adds a red border around it in the image. */
	if(draw) {
		for(int i=0;i<blobs.size();i++)
			for(int j=0;j<blobs[i].size();j++)
				IgOutPut.at<cv::Vec3b>(blobs[i][j]) = color;
	}
	
	return blobs;
}

/* the function 'compute_rectangle' find the coordinates of start and and of the square that includes the selected card.
 * INPUT  -> vertices : the vertices of the card.
 * OUTPUT  -> rectangle : the coordinates of the square.
 */
void compute_rectangle(std::vector<cv::Point2f> &vertices, std::vector<int> &rectangle) {
	if(vertices.size()==0)
		throw "input parameter must non empty.";

	int width_min = vertices[0].x, width_max = 0, height_min = vertices[0].y, height_max = 0;

	// find the maximim and minimum coordinates of the card
	for(int i=0;i<vertices.size();i++) {
		width_min = (width_min > vertices[i].x)? vertices[i].x : width_min;
		width_max = (width_max < vertices[i].x)? vertices[i].x : width_max;
		height_min = (height_min > vertices[i].y)? vertices[i].y : height_min;
		height_max = (height_max < vertices[i].y)? vertices[i].y : height_max;
	}

	rectangle.push_back(height_min);
	rectangle.push_back(height_max);
	rectangle.push_back(width_min);
	rectangle.push_back(width_max);
}

/* the function 'increase_to' increase the size of an image from (0,0) to (columns,rows).
 * INPUT  -> IgInPut : the image to increase.
 * INPUT  -> columns : max column.
 * INPUT  -> rows : max row.
 * OUTPUT  -> IgInc : the increased image.
 */
cv::Mat increase_to(cv::Mat IgInPut, int columns, int rows) {

    /* create the output image with the correct shape */
    std::cout << IgInPut.type() << std::endl;
    cv::Mat IgInc = cv::Mat(columns,rows,CV_8U,cv::Scalar(0));

    for(int i =0; i<IgInPut.rows; ++i) {
        unsigned char * pIgInPut = IgInPut.ptr(i);
        unsigned char * pIgInPutEnd = pIgInPut + IgInPut.cols;

        unsigned char * pIgInc = IgInc.ptr(i);

        /* copy the original image in the top left corner of the increase image */
        while(pIgInPut<pIgInPutEnd) {
            *pIgInc = *pIgInPut;
            pIgInPut++;
            pIgInc++;
        }
    }

    return IgInc;
}

/* the function 'intersection' take as input two pairs of point for two line and find the intersection between them.
 * INPUT  -> o1 : first point line 1
 * INPUT  -> p1 : second point line 1
 * INPUT  -> o2 : first point line 2
 * INPUT  -> p2 : second point line 2
 * OUTPUT ->  r : the intersection coordinate.
 * OUTPUT  : a boolean that specify if the intersection exists.
 */
bool intersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2, cv::Point2f &r) {
    cv::Point2f x = o2 - o1;
    cv::Point2f d1 = p1 - o1;
    cv::Point2f d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (std::abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;

    return true;
}