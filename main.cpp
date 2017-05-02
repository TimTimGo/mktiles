#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <vector>
#include <fstream>
#include <cmath>

using namespace cv;
using namespace std;

class Palette
/// describes a palette of available colors
{
public:
  struct ColorSpec{
    string id;
    string name;
    struct Color{
      uint16_t r;
      uint16_t g;
      uint16_t b;
    } color;
    struct Availability{
      union{
        uint8_t indexed[4];
        struct{
          uint8_t plate1x1;
          uint8_t tile1x1;
          uint8_t round1x1;
          uint8_t round2x2;
        }named;
      };
    } availability;
  };
private:
  vector<ColorSpec> availableColors;
public:
  Palette(string paletteFile)
  /// construct palette from csv file
  {
    ifstream in(paletteFile);
    ColorSpec create;
    // skip header line
    in.ignore(numeric_limits<streamsize>::max(), '\n');
    while(getline(in, create.id,',') &&
          getline(in, create.name,',') &&
          in.ignore(numeric_limits<streamsize>::max(), '"') &&
          in >> create.color.r && in.ignore(numeric_limits<streamsize>::max(), ',') &&
          in >> create.color.g && in.ignore(numeric_limits<streamsize>::max(), ',') &&
          in >> create.color.b &&
          in.ignore(numeric_limits<streamsize>::max(), '"') &&
          in.ignore(numeric_limits<streamsize>::max(), ',') &&
          in.ignore(numeric_limits<streamsize>::max(), ',') && // hex value is ignored later
          in >> create.availability.named.plate1x1 &&
          in.ignore(numeric_limits<streamsize>::max(), ',') &&
          in >> create.availability.named.tile1x1 &&
          in.ignore(numeric_limits<streamsize>::max(), ',') &&
          in >> create.availability.named.round1x1 &&
          in.ignore(numeric_limits<streamsize>::max(), ',') &&
          in >> create.availability.named.round2x2 &&
          in.ignore(numeric_limits<streamsize>::max(), '\n'))
      availableColors.push_back(create);
  }

  ColorSpec* getSpecFromPalette(ColorSpec::Color c){
    double minDist = std::numeric_limits<double>::max();
    auto* minSpec = &availableColors[0];
    for(auto& spec : availableColors){
      double dist = pow(c.r-(double)spec.color.r,2) + pow(c.g-(double)spec.color.g,2) + pow(c.b-(double)spec.color.b,2);
      if(dist < minDist){
        minDist=dist;
        minSpec=&spec;
      }
    }
    return minSpec;
  }

};

void groupByMask(Mat image, Mat mask, uint64_t groups, Palette& palette){
  // accept only char type matrices
  CV_Assert(image.depth() == CV_8U);
  // accept only images with 3 channels
  CV_Assert(image.channels() == 3);
  uint64_t tileHeight = mask.rows;
  uint64_t tileWidth = mask.cols;

  // iterate over tiles of size mask
  for (int r = 0; r < image.rows; r += tileHeight)
    for (int c = 0; c < image.cols; c += tileWidth){
      Mat tile = image(Range(r, std::min(r + tileHeight, static_cast<uint64_t>(image.rows))),
                       Range(c, std::min(c + tileWidth, static_cast<uint64_t>(image.cols))));//no data copying here
      // create group averages for the tile
      uint64_t count[groups*3];
      uint64_t sum[groups*3];
      char avg[groups*3];
      for(int i = 0; i < 3*groups; i++){
        count[i] = 0;
        sum[i] = 0;
        avg[i] = 0;
      }
      for(MatIterator_<Vec3b> it = tile.begin<Vec3b>(), end = tile.end<Vec3b>(); it != end; ++it){
        const auto pos = it.pos();
        const auto group = mask.at<char>(pos);
        // add current position's value to aggregate of group
        for(int i = 0; i < 3; i++){
          count[group*3+i] += 1;
          sum[group*3+i] += (*it)[i];
        }
      }
      //create group values
      for(int i = 0; i < 3*groups; i+=3){
        // find closest color
        auto spec = palette.getSpecFromPalette({
            static_cast<uint16_t>(sum[i+2]/std::max(count[i+2], 1ull)),
            static_cast<uint16_t>(sum[i+1]/std::max(count[i+1], 1ull)),
            static_cast<uint16_t>(sum[i]/std::max(count[i], 1ull))});
        avg[i] = spec->color.b;
        avg[i+1] = spec->color.g;
        avg[i+2] = spec->color.r;
      }
      //write group values into image
      for(MatIterator_<Vec3b> it = tile.begin<Vec3b>(), end = tile.end<Vec3b>(); it != end; ++it){
        auto pos = it.pos();
        auto group = mask.at<char>(pos);
        // lookup average and write it back to image
        for(int i = 0; i < 3; i++){
          (*it)[i] = avg[group*3+i];
        }
      }
    }
}

Mat circleMask(int32_t circlesLongSide, int32_t imageLongSide){
  int32_t sideLength = imageLongSide / circlesLongSide;
  Mat mask = Mat::zeros(sideLength,sideLength, CV_8UC1);
  rectangle(mask, {Point(0,0),Point(sideLength/2,sideLength/2)}, 0, -1);
  rectangle(mask, {Point(sideLength/2,0),Point(sideLength,sideLength/2)}, 1, -1);
  rectangle(mask, {Point(0,sideLength/2),Point(sideLength/2,sideLength)}, 2, -1);
  rectangle(mask, {Point(sideLength/2,sideLength/2),Point(sideLength,sideLength)}, 3, -1);
  circle(mask, {sideLength/2,sideLength/2}, sideLength/2, 4, -1);
  circle(mask, {sideLength/2,sideLength/2}, sideLength/5, 5, -1);
  return mask;
}


int main(int argc, char* argv[]){
  if (argc < 3){
    std::cout << "usage: " << argv[0] << " <paletteFile> <inputImage> [<outputImage>]\n";
    return -1;
  }
  Palette palette(argv[1]);
  Mat image;
  image = imread(argv[2], 1);
  if (!image.data){
    printf("No image data \n");
    return -1;
  }
  auto mask = circleMask(100, max(image.rows, image.cols));
  groupByMask(image, mask, 6, palette);
  namedWindow("Display Image", WINDOW_AUTOSIZE);
  imshow("Display Image", image);
  if(argc >=4)
    imwrite(argv[3], image);
  waitKey(0);
  return 0;
}
