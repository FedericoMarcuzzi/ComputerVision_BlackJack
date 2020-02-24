#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "include/misc.h"

#include <boost/lexical_cast.hpp>


#define CARD_WIDTH 458
#define CARD_HEIGHT 640


/* the 'CardManager' class deals with providing the value of the cards and maintaining their status in the game
 * the 'CardManager' have the following function:
 * get_state(int,int) -> int : return the state for a card: 0 = in deck, 1 = in field, 2 = removed.
 * get_value(int) -> int : return the value for a card.
 * mark_seen(int) -> void : set that a card it was seen in the last frame.
 * mark_not_seen[int] -> void : set that a card it wasn't seen in the last frame.
 */
class CardManager {
public:

    CardManager(): cons_frame_views(), cons_frame_not_views(), state() {
    }

    // 0 = in deck, 1 = in field, 2 = removed.
    int get_state(int i) {
        return state[i];
    }

    // return the value of card i
    int get_value(int i) {
        return value[i%13];
    }

    // set that the card i it was seen in the last frame
    void mark_seen(int i) {
        if(state[i]!=2) {
            cons_frame_views[i]++;
            cons_frame_not_views[i] = 0;
            // if the card was seen for more than 5 consecutively frame, then the card is marked as on the table.
            if (cons_frame_views[i] > 5)
                state[i] = 1;
        }
    }

    // set that the card i it wasn't seen in the last frame
    void mark_not_seen(int i) {
        if(state[i]==1) {
            cons_frame_not_views[i]++;
            // if the card wasn't seen for more than 118 consecutively frame then the card is marked as removed.
            if(cons_frame_not_views[i]>118)
                state[i] = 2;
        }
        else
            cons_frame_views[i] = 0;
    }

private:
    // TODO warning inizializzazione
    // consecutive number of frames in which the card was viewed
    int cons_frame_views[52];
    // consecutive number of frames in which the card has not been viewed
    int cons_frame_not_views[52];
    // 0 = in deck, 1 = in field, 2 = removed.
    int state[52];
    // cards value
    const int value[13] = {11,2,3,4,5,6,7,8,9,10,10,10,10};
};


/* the 'find_figure' function extract the blob from the image by removing the borders that surround it.
 * INPUT  -> IgInPut : input image.
 * INPUT  -> blobs : the blobs of the image.
 * OUTPUT -> IgBlWh : only the image of the blob
 */
cv::Mat find_figure(cv::Mat IgInPut, std::vector<std::vector<cv::Point> > blobs) {
    // if there are no blobs, the ostu of the entire image is returned.
    if(blobs.empty())
        return apply_otsu(IgInPut);

    // search the biggest blob
    std::vector<cv::Point> blob = blobs[0];
    for(int i=0;i<blobs.size();i++)
        if(blobs[i].size()>blob.size())
            blob = blobs[i];

    // removes the space around the blob
    int width_min =IgInPut.cols, width_max =0, height_min =IgInPut.rows, height_max =0;

    for(int i=0;i<blob.size();i++) {
        width_min = (width_min > blob[i].x)? blob[i].x : width_min;
        width_max = (width_max < blob[i].x)? blob[i].x : width_max;
        height_min = (height_min > blob[i].y)? blob[i].y : height_min;
        height_max = (height_max < blob[i].y)? blob[i].y : height_max;
    }

    cv::Mat IgBlWh = apply_otsu(IgInPut);

    return IgBlWh(cv::Range(height_min,height_max),cv::Range(width_min,width_max));
}


/* the 'compute_differences' function compute the differences between the two figure.
 * The difference is calculated based on the number of different pixels between the two images.
 * INPUT  -> IgOne : one of the figures.
 * INPUT  -> IgTwo : one of the figures.
 * OUTPUT  : the differences between the two figure.
 */
int compute_differences(cv::Mat IgOne, cv::Mat IgTwo) {
    int rows_ig1 = IgOne.rows, cols_ig1 = IgOne.cols, rows_ig2 = IgTwo.rows, cols_ig2 = IgTwo.cols;
    int rows_min = (rows_ig1 < rows_ig2)? rows_ig1 : rows_ig2, cols_min = (cols_ig1 < cols_ig2)? cols_ig1 : cols_ig2;

    IgOne = IgOne(cv::Range(0,rows_min), cv::Range(0,cols_min));
    IgTwo = IgTwo(cv::Range(0,rows_min), cv::Range(0,cols_min));
    cv::Mat IgDiff;

    // compute the difference fo each pixel, 0 if have the same value 255 otherwise.
    bitwise_xor(IgOne,IgTwo,IgDiff);

    return (int) cv::sum(IgDiff)[0]/255;
}


/* the 'is_black' function return true if the image contain red part.
 * INPUT  -> IgInPut : the input image.
 * OUTPUT  : boolean true o false (there is at last one red part in the image).
 */
bool is_black(cv::Mat &IgInPut) {
    cv::Mat hsv;
    cv::Mat1b mask1, mask2;

    // change the color space from BGR to HSV
    cv::cvtColor(IgInPut, hsv, cv::COLOR_BGR2HSV);

    // detect red object (compute threshold based on the range of pixel values)
    cv::inRange(hsv, cv::Scalar(0, 70, 50), cv::Scalar(10, 255, 255), mask1);
    cv::inRange(hsv, cv::Scalar(170, 70, 50), cv::Scalar(180, 255, 255), mask2);

    // compute the or between the two mask (the result matrix contains one if only if a red object has been identified)
    cv::Mat1b result = mask1 | mask2;

    // true if == 0, false otherwise
    return (cv::sum(result)[0] == 0);
}


/* the 'is_horizontal' function identifies the orientation of the card.
 * if the number of vertices passed are not 4 throw an exception.
 * INPUT  -> vertices : the 4 vertices of the card.
 * OUTPUT  : boolean true o false (true is horizontal, false is vertical).
 */
bool is_horizontal(std::vector<cv::Point2f> &vertices) {
    if(vertices.size()!=4)
        throw std::runtime_error("The number of vertices must be equal to 4");

    // compute distances between vertices 0 and 1 and 0 and 3.
    double dist1to2 = std::sqrt(std::pow(vertices[0].x - vertices[1].x, 2) + std::pow(vertices[0].y - vertices[1].y, 2));
    double dist4to1 = std::sqrt(std::pow(vertices[3].x - vertices[0].x, 2) + std::pow(vertices[3].y - vertices[0].y, 2));

    return (dist1to2 < dist4to1);
}


/* the 'warp_card' function change perspective of the card.
 * if the number of vertices passed are not 4 throw an exception.
 * INPUT  -> IgInPut : the frame of the table.
 * INPUT  -> card_vertices : the 4 vertices of the card.
 * OUTPUT -> IgOutPut : the warped card.
 */
void warp_card(cv::Mat &IgInPut, std::vector<cv::Point2f> &card_vertices, cv::Mat &IgOutPut) {
    std::vector<cv::Point2f> card_new_vertices;

    // identifies the orientation of the card and assigns the position of the new vertices.
    if(is_horizontal(card_vertices)) {
        card_new_vertices.push_back(cv::Point(0,0));
        card_new_vertices.push_back(cv::Point(CARD_WIDTH-1,0));
        card_new_vertices.push_back(cv::Point(CARD_WIDTH-1,CARD_HEIGHT-1));
        card_new_vertices.push_back(cv::Point(0,CARD_HEIGHT-1));
    } else {
        card_new_vertices.push_back(cv::Point(CARD_WIDTH-1,0));
        card_new_vertices.push_back(cv::Point(CARD_WIDTH-1,CARD_HEIGHT-1));
        card_new_vertices.push_back(cv::Point(0,CARD_HEIGHT-1));
        card_new_vertices.push_back(cv::Point(0,0));
    }

    // find the perspective transformation (the 3x3 homography matrix)
    cv::Mat perspective_transformation  = cv::findHomography(card_vertices,card_new_vertices);
    // warp the perspective
    cv::warpPerspective(IgInPut,IgOutPut,perspective_transformation,cv::Size(CARD_WIDTH,CARD_HEIGHT));
}


/* the 'draw_square' function draw a colored squared in a given image.
 * INPUT  -> IgInPut : the image on which the square will be drawn.
 * INPUT  -> column_start : starting column of the square.
 * INPUT  -> rows_start :  starting row of the square.
 * INPUT  -> column_end : column of the end of the square.
 * INPUT  -> rows_end : row of the end of the square.
 * INPUT  -> color : the square color.
 * INPUT  -> alpha : transparency (default 0.3).
 * OUTPUT -> IgInPut : the input image with the drawn square.
 */
void draw_square(cv::Mat &IgInPut, int column_start, int rows_start, int column_end, int rows_end, cv::Scalar const &color, double alpha=0.3) {
    cv::Mat roi = IgInPut(cv::Range(rows_start, rows_end),cv::Range(column_start, column_end));
    cv::Mat shape(roi.size(), CV_8UC3, color);
    cv::addWeighted(shape, alpha, roi, 1.0 - alpha , 0.0, roi);
}


/* the 'MatchingMethod' function compute the template matching (with CV_TM_CCORR_NORMED method) between the warped card and the template of the cards.
 * INPUT  -> IgTmplt : the warped card.
 * INPUT  -> IgCard : the template image.
 * OUTPUT -> int : the index of the matched card.
 */
int MatchingMethod(cv::Mat IgTmplt, cv::Mat IgCard) {
    // create the result matrix
    cv::Mat result(IgTmplt.cols - IgCard.cols + 1,IgTmplt.rows - IgCard.rows + 1,CV_8UC3);

    // do the matching and normalize
    cv::matchTemplate(IgTmplt,IgCard,result,CV_TM_CCORR_NORMED);
    normalize(result,result,0,1,cv::NORM_MINMAX,-1,cv::Mat());

    // localizing the best match with minMaxLoc
    double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

    // return the index of the card (from 0 to 15)
    return (maxLoc.y + IgTmplt.rows/32)/(IgTmplt.rows/16);
}

int main(int argc, char* argv[])
{
    // cards attributes
    const int num_seed =4;
    const int num_value =13;
    const int num_card =52;
    CardManager cards;
    cv::Mat img_seed[4];

    // cards states
    int card_deck;
    int card_removed;
    int card_field;
    int last =0;

    // game statistics
    double less, equal, greater;
    int total_rank;

    // fps variable
    int fps;
    double time1, time2;

    // fps statistics
    double freq = cv::getTickFrequency();

    // graphic parameters
    const double screen_proportion = 9;
    int cw = (int) round(CARD_WIDTH/screen_proportion);
    int ch = (int) round(CARD_HEIGHT/screen_proportion);
    const cv::Scalar color[] = {cv::Scalar(255,0,0),cv::Scalar(0,255,0),cv::Scalar(0,0,255),cv::Scalar(255,255,255)};

    // video frame
    cv::Mat frame;

    // input video pointer.
    cv::VideoCapture cap;

    if(!cap.open("test_video_1.mp4")) {
        std::cerr << "error in the opening of input file.";
        return -1;
    }

    // load cards template
    cv::Mat template_image = cv::imread("cards.jpg");
    if (template_image.rows==0)
        throw std::runtime_error("cards template image not found");


    // resize cards state for displaying
    cv::Mat game_state_empty = template_image(cv::Range(0,CARD_HEIGHT*num_value),cv::Range(0,CARD_WIDTH*num_seed));
    cv::resize(game_state_empty, game_state_empty, cv::Size((int) round(game_state_empty.cols/screen_proportion),(int) round(game_state_empty.rows/screen_proportion)));


    // resize the template to make the match faster
    int size = 11;
    cv::Mat all_card_image;
    cv::resize(template_image, all_card_image, cv::Size(template_image.cols/size, template_image.rows/size));


    // the template image is split by seed.
    cv::Mat spades_card_image = all_card_image(cv::Range(0,CARD_HEIGHT*16/size),cv::Range(0,CARD_WIDTH/size));
    cv::Mat clubs_card_image = all_card_image(cv::Range(0,CARD_HEIGHT*16/size),cv::Range(CARD_WIDTH/size,CARD_WIDTH*2/size));
    cv::Mat diamonds_card_image = all_card_image(cv::Range(0,CARD_HEIGHT*16/size),cv::Range(CARD_WIDTH*2/size,CARD_WIDTH*3/size));
    cv::Mat hearts_card_image = all_card_image(cv::Range(0,CARD_HEIGHT*16/size),cv::Range(CARD_WIDTH*3/size,CARD_WIDTH*4/size));
    cv::Mat match_card_list[] = {spades_card_image,clubs_card_image,diamonds_card_image,hearts_card_image};


    // retrieves seed image
    for(int i=0;i<num_seed;i++) {
        cv::Mat seed = ~template_image(cv::Range(106,167), cv::Range(CARD_WIDTH * i + 27, CARD_WIDTH * i + 85));
        std::vector<std::vector<cv::Point> > blobs = find_contours(seed,seed,100,1000);
        img_seed[i] = find_figure(seed,blobs);
    }


    // main loop
    while(cap.isOpened()) {
        cap >> frame;

        if (frame.rows==0)
            break;

        // sample for computing fps
        time1 = cv::getTickCount();

        // find the card in the field
        std::vector<std::vector<cv::Point> > cards_contours = find_contours(frame,frame,500,1500,false);

        // marks the card seen in this frame
        bool card_see_in_this_frame[num_card] = {};

        // presentation frame (the frame will contain all the information about the state of the game)
        cv::Mat frame_to_show = frame.clone();

        // continue if any shapes were found in the frame
        for (int i = 0; i < cards_contours.size(); i++) {
            // find the vertices of the card  /**-> start
            std::vector<cv::Point2f> card_vertices;

            double peri = cv::arcLength(cards_contours[i], true);
            cv::approxPolyDP(cards_contours[i], card_vertices, 0.01 * peri, true);
            // end <-**/

            // fist check if is a card (a card have 4 vertices)
            if (card_vertices.size() == 4) {
                // warp the retrieved card
                cv::Mat warped_card(CARD_HEIGHT, CARD_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
                warp_card(frame, card_vertices, warped_card);

                // extract seed
                cv::Mat part_seed = warped_card(cv::Range(100, 180), cv::Range(15, 73));
                cv::Mat seed = ~part_seed;
                std::vector<std::vector<cv::Point> > blobs = find_contours(seed, seed);
                seed = find_figure(seed, blobs);

                // check if a seed shape was found
                if (seed.cols != 0) {
                    int min_diff;
                    int seed_index;

                    /* check if the card is red or black, secondarily find the seed.
                     * This reduces four times the time needed to find the match.
                     */
                    if (is_black(part_seed)) {
                        min_diff = compute_differences(img_seed[0], seed);
                        seed_index = (min_diff > compute_differences(img_seed[1], seed)) ? 1 : 0;
                    } else {
                        min_diff = compute_differences(img_seed[2], seed);
                        seed_index = (min_diff > compute_differences(img_seed[3], seed)) ? 3 : 2;
                    }

                    // the 97/99 value allows the match to have a bit of slack
                    cv::resize(warped_card, warped_card, cv::Size(CARD_WIDTH/size*(97./99.),CARD_HEIGHT/size*(97./99.)));

                    // compute index retrieved card
                    int card_index = MatchingMethod(match_card_list[seed_index], warped_card);
                    card_index = ((card_index > num_value) ? card_index % 8 : card_index % num_value) + seed_index * num_value;

                    // marks cards
                    card_see_in_this_frame[card_index] = true;

                    // draws perimeter
                    for (int j = 0; j < cards_contours[i].size(); j++)
                        cv::circle(frame_to_show, cards_contours[i][j], 1, cv::Vec3b(120, 255, 180), -1);

                    // draws vertices
                    for(int i=0;i<card_vertices.size();i++) {
                        cv::circle(frame_to_show, card_vertices[i], 10, color[i], -1);
                        cv::putText(frame_to_show, boost::lexical_cast<std::string>(i),cv::Point2f(card_vertices[i].x-6,card_vertices[i].y+7),cv::FONT_HERSHEY_SIMPLEX,0.7,cv::Scalar(68, 147, 255),1,cv::LINE_AA);

                    }
                }
            }
        }


        // check state of cards  /**-> start
        cv::Mat game_state = game_state_empty.clone();
        less = 0, equal = 0, greater = 0, total_rank = 0, card_removed = 0, card_field = 0;

        for (int i = 0; i < num_card; i++) {
            // check card seen in this frame
            if (card_see_in_this_frame[i])
                cards.mark_seen(i);
            else
                cards.mark_not_seen(i);

            int color_index = 0;

            // draw the field state
            if (cards.get_state(i) == 1) {
                card_field++;
                total_rank += cards.get_value(i);
                color_index = 1;
            } else if (cards.get_state(i) == 2) {
                card_removed++;
                color_index = 2;
            }

            // draw the square on the game state image
            draw_square(game_state, cw * (i / num_value), ch * (i % num_value), cw * (i / num_value + 1), ch * (i % num_value + 1), color[color_index]);
        } //  end <-**/


        // check the next possible card to drawn  /**-> start
        card_deck = num_card - card_field - card_removed;
        int gap = 21 - total_rank;

        for (int i = 0; i < num_card; i++) {
            if (cards.get_state(i) == 0) {
                if (cards.get_value(i) > gap)
                    greater++;
                else if (cards.get_value(i) == gap)
                    equal++;
                else
                    less++;
            }
        } //  end <-**/


        // compute probability
        double p_equal = (card_deck) ? equal / card_deck : (total_rank == 21) ? 1 : 0;
        double p_less = (card_deck) ? less / card_deck : (total_rank < 21) ? 1 : 0;
        double p_grtr = (card_deck) ? greater / card_deck : (total_rank > 21) ? 1 : 0;


        // print probability  /**-> start
        if (total_rank > last) {
            last = total_rank;
            std::cout << "\n\nCARD STATE" << std::endl;
            std::cout << "total rank: " << total_rank << std::endl;
            std::cout << "dack card: " << card_deck << std::endl;
            std::cout << "table card: " << card_field << std::endl;
            std::cout << "removed card: " << card_removed << std::endl;

            std::cout << "\nPROBABILITY" << std::endl;
            std::cout << "probability of winning: " << p_equal << std::endl;
            std::cout << "probability less than 21: " << p_less << std::endl;
            std::cout << "probability greater than 21: " << p_grtr << std::endl;
        } else if (total_rank < last)
            last = 0;
        //  end <-**/


        // display probability  /**-> start
        std::string string_win = "win:  " + boost::lexical_cast<std::string>(round(p_equal * 100)) + '%';
        std::string string_less = "less: " + boost::lexical_cast<std::string>(round(p_less * 100)) + '%';
        std::string string_greater = "grtr: " + boost::lexical_cast<std::string>(round(p_grtr * 100)) + '%';

        cv::putText(frame_to_show, string_win, cv::Point2f(10, frame.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(frame_to_show, string_win, cv::Point2f(10, frame.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(50, 200, 200), 2, cv::LINE_AA);

        cv::putText(frame_to_show, string_less, cv::Point2f(10, frame.rows - 60), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(frame_to_show, string_less, cv::Point2f(10, frame.rows - 60), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(50, 200, 200), 2, cv::LINE_AA);

        cv::putText(frame_to_show, string_greater, cv::Point2f(10, frame.rows - 100), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(frame_to_show, string_greater, cv::Point2f(10, frame.rows - 100), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(50, 200, 200), 2, cv::LINE_AA);
        //  end <-**/


        // display game state  /**-> start
        std::string deck_string = "deck: " + boost::lexical_cast<std::string>(card_deck);
        std::string table_string = "table: " + boost::lexical_cast<std::string>(card_field);
        std::string removed_string = "rmv: " + boost::lexical_cast<std::string>(card_removed);
        std::string total_rank_string = "tr: " + boost::lexical_cast<std::string>(total_rank);

        cv::putText(frame_to_show, deck_string, cv::Point2f(10, frame.rows - 220), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(frame_to_show, deck_string, cv::Point2f(10, frame.rows - 220), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(50, 255, 50), 2, cv::LINE_AA);

        cv::putText(frame_to_show, table_string, cv::Point2f(10, frame.rows - 180), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(frame_to_show, table_string, cv::Point2f(10, frame.rows - 180), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(50, 255, 50), 2, cv::LINE_AA);

        cv::putText(frame_to_show, removed_string, cv::Point2f(10, frame.rows - 140), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(frame_to_show, removed_string, cv::Point2f(10, frame.rows - 140), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(50, 255, 50), 2, cv::LINE_AA);

        cv::putText(frame_to_show, total_rank_string, cv::Point2f(10, frame.rows - 260), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(frame_to_show, total_rank_string, cv::Point2f(10, frame.rows - 260), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(50, 255, 50), 2, cv::LINE_AA);
        //  end <-**/


        // sample for computing fps
        time2 = cv::getTickCount();


        // display number of frame per second  /**-> start
        fps = (int) round(freq / (time2 - time1));
        std::string fps_string = "FPS: " + boost::lexical_cast<std::string>(fps);

        cv::putText(frame_to_show, fps_string, cv::Point2f(10, 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 3,
                    cv::LINE_AA);
        cv::putText(frame_to_show, fps_string, cv::Point2f(10, 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(200, 50, 200),
                    2, cv::LINE_AA);
        //  end <-**/


        // display card state and field
        cv::imshow("card_state", game_state);
        cv::imshow("field", frame_to_show);


        // pause and quite  /**-> start
        int key_pressed = cv::waitKey(1);

        if (key_pressed=='p' || key_pressed=='P' || key_pressed==32)
            do {
                key_pressed = cv::waitKey(0);
            } while(key_pressed!='p' && key_pressed!='P' && key_pressed!=32 && key_pressed!='q' && key_pressed!='Q');
        if (key_pressed == 'q' || key_pressed=='Q')
            break;
        //  end <-**/
    }

    return 0;
}