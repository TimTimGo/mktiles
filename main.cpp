#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <vector>
#include <fstream>
#include <cmath>
#include <highgui.h>

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
    struct ColorLab{
      float l;
      float a;
      float b;
      ColorLab() = default;
      ColorLab(Vec3f v):l(v[0]),a(v[1]),b(v[2]){}
      operator Vec3f()const{
        return Vec3f(l,a,b);
      }
    } colorLab;
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
  vector<ColorSpec> availableColors;
  Palette() = default;
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
    Mat rgb(availableColors.size(),1, CV_8UC3);
    int i = 0;
    for(auto& spec : availableColors){
      rgb.at<Vec3b>(i,0)[0] = spec.color.b;
      rgb.at<Vec3b>(i,0)[1] = spec.color.g;
      rgb.at<Vec3b>(i,0)[2] = spec.color.r;
      i++;
    }
    Mat rgbFloat;
    rgb.convertTo(rgbFloat, CV_32FC3, 1./255);
    Mat lab;
    cvtColor(rgbFloat, lab, COLOR_BGR2Lab);
    i = 0;
    for(auto& spec : availableColors){
      spec.colorLab.l = lab.at<Vec3f>(i,0)[0];
      spec.colorLab.a = lab.at<Vec3f>(i,0)[1];
      spec.colorLab.b = lab.at<Vec3f>(i,0)[2];
      i++;
    }
  }

  ColorSpec* getSpecFromPalette(ColorSpec::ColorLab c, int partId);

};


struct PaintState{
  Mat original;
  Mat image;
  int sigma = 200, threshold = 500, amount = 100, luminanceFactor = 500, circlesLongSide = 48;
  Palette palette;
  bool showMosaic = true;
  bool writeLDrawFile = false;
  bool writePartList = false;
  string ldrawFileName;
  string partList;
}state;


Palette::ColorSpec* Palette::getSpecFromPalette(ColorSpec::ColorLab c, int partId){
  double minDist = std::numeric_limits<double>::max();
  ColorSpec* minSpec = nullptr;
  for(auto& spec : availableColors){
    double dist = (state.luminanceFactor/500.0) * pow(c.l-spec.colorLab.l,2) + pow(c.a-spec.colorLab.a,2) + pow(c.b-spec.colorLab.b,2);
    if(spec.availability.indexed[partId] == '+' && dist < minDist){
      minDist=dist;
      minSpec=&spec;
    }
  }
  assert(minSpec != nullptr);
  return minSpec;
}

struct MaskConfig{
  Mat mask;
  std::vector<uint8_t> groupIdToPart;
  uint32_t nrGroups;
  uint32_t nrOfParts;
};

MaskConfig circleMask(int32_t circlesLongSide, int32_t imageLongSide){
  int32_t sideLength = imageLongSide / circlesLongSide;
  MaskConfig mc;
  Mat& mask = mc.mask;
  mask = Mat::zeros(sideLength,sideLength, CV_8UC1);
  rectangle(mask, {Point(0,0),                      Point(sideLength/2,sideLength/2)}, 0, -1);
  rectangle(mask, {Point(sideLength/2,0),           Point(sideLength,sideLength/2)}, 1, -1);
  rectangle(mask, {Point(0,sideLength/2),           Point(sideLength/2,sideLength)}, 2, -1);
  rectangle(mask, {Point(sideLength/2,sideLength/2),Point(sideLength,sideLength)}, 3, -1);
  circle(mask, {sideLength/2,sideLength/2}, sideLength/2, 4, -1);
  circle(mask, {sideLength/2,sideLength/2}, sideLength/5, 5, -1);
  mc.groupIdToPart = {1,1,1,1,3,2};
  mc.nrGroups = 6;
  mc.nrOfParts = 4; // groups in palette are numbered from 0 to 3
  return mc;
}

template<typename C>
void groupByMask(Mat image, MaskConfig mc, uint64_t groups, Palette& palette, C onTileDone){
  Mat& mask = mc.mask;
  // accept only images with 3 channels
  CV_Assert(image.channels() == 3);
  uint64_t tileHeight = mask.rows;
  uint64_t tileWidth = mask.cols;
  uint64_t nrCols = image.cols / tileWidth + 2;

  // iterate over tiles of size mask

  vector<Vec3f> currentRow;
  vector<Vec3f> nextRow;
  currentRow.assign(nrCols, {0,0,0});
  nextRow.assign(nrCols, {0,0,0});
  for (int r = 0, rc=0; r < image.rows; r += tileHeight, rc++){
    for (int c = 0, cc=0; c < image.cols; c += tileWidth, cc++){
      Mat tile = image(Range(r, std::min(r + tileHeight, static_cast<uint64_t>(image.rows))),
                       Range(c, std::min(c + tileWidth, static_cast<uint64_t>(image.cols))));//no data copying here
      // create group averages for the tile
      uint64_t count[groups*3];
      double sum[groups*3];
      Palette::ColorSpec* avg[groups];
      for(int i = 0; i < 3*groups; i++){
        count[i] = 0;
        sum[i] = 0;
      }
      for(MatIterator_<Vec3f> it = tile.begin<Vec3f>(), end = tile.end<Vec3f>(); it != end; ++it){
        const auto pos = it.pos();
        const auto group = mask.at<char>(pos);
        // add current position's value to aggregate of group
        for(int i = 0; i < 3; i++){
          count[group*3+i] += 1;
          sum[group*3+i] += (*it)[i];
        }
      }

      vector<Vec3f> groupValues;
      //create group values
      for(int group = 0; group < groups; group++){
        const int i = group*3;
        groupValues.push_back({
                static_cast<float>(sum[i]/std::max(count[i], 1ull)),
                static_cast<float>(sum[i+1]/std::max(count[i+1], 1ull)),
                static_cast<float>(sum[i+2]/std::max(count[i+2], 1ull))});
      }
      //quantize large circle
      //TODO: consider error from current row
      avg[4] = palette.getSpecFromPalette(groupValues[4] + currentRow[cc], mc.groupIdToPart[4]);
      Vec3f quantError = groupValues[4] - static_cast<Vec3f>(avg[4]->colorLab);
      // propagate error to small circle
      groupValues[5] += quantError;
      // quantize small circle
      avg[5] = palette.getSpecFromPalette(groupValues[5], mc.groupIdToPart[5]);
      quantError = groupValues[5] - static_cast<Vec3f>(avg[5]->colorLab);
      quantError /= 4;
      // quantize 1x1
      Vec3f remainingError;
      for(int i=0;i<4;++i){
        Vec3f targetColor = groupValues[i]+quantError;
        avg[i] = palette.getSpecFromPalette(targetColor, mc.groupIdToPart[i]);
        remainingError += targetColor - static_cast<Vec3f>(avg[i]->colorLab);
      }
      // propagate error
      currentRow[cc+1] += remainingError * (7/16.);
      if(cc > 0)
        nextRow[cc-1] += remainingError * (3/16.);
      nextRow[cc] += remainingError * (5/16.);
      nextRow[cc+1] += remainingError * (1/16.);


      //write group values into image
      for(MatIterator_<Vec3f> it = tile.begin<Vec3f>(), end = tile.end<Vec3f>(); it != end; ++it){
        auto pos = it.pos();
        auto group = mask.at<char>(pos);
        // lookup average and write it back to image
        (*it)[0] = avg[group]->colorLab.l;
        (*it)[1] = avg[group]->colorLab.a;
        (*it)[2] = avg[group]->colorLab.b;
      }
      onTileDone(rc,cc,avg);
    }
    swap(currentRow, nextRow);
    nextRow.assign(nrCols, {0,0,0});
  }
}

void sharpen(Mat& img){
  //from http://docs.opencv.org/master/d1/d10/classcv_1_1MatExpr.html#details
  // sharpen image using "unsharp mask" algorithm
  Mat blurred; double sigma = max(1,state.sigma)/100.0, threshold = max(1,state.threshold)/100.0, amount = max(1,state.amount)/100.0;
  GaussianBlur(img, blurred, Size(), sigma, sigma);
  Mat lowContrastMask = abs(img - blurred) < threshold;
  Mat sharpened = img*(1+amount) + blurred*(-amount);
  img.copyTo(sharpened, lowContrastMask);
  img=sharpened;
}

void writeLDrawFile(int x, int y, Palette::ColorSpec* avgs[], ofstream& ldFile)
/// Write mosaic in LDraw format into ldFile 
{
  ldFile << "1 0x2" << std::hex << static_cast<int>(avgs[5]->color.r) << static_cast<int>(avgs[5]->color.g) << static_cast<int>(avgs[5]->color.b)
         << std::dec << " " << (y * 20 * 2) << " -16 " << (-x * 20 * 2 + 10) <<   " 1 0 0 0 1 0 -0 0 1 6141.dat\n";
  ldFile << "1 0x2" << std::hex << static_cast<int>(avgs[4]->color.r) << static_cast<int>(avgs[4]->color.g) << static_cast<int>(avgs[4]->color.b)
         << std::dec << " " << (y * 20 * 2) << " -8 " << (-x * 20 * 2 + 10) <<     " 1 0 0 0 1 0 -0 0 1 18674.dat\n";
  ldFile << "1 0x2" << std::hex << static_cast<int>(avgs[0]->color.r) << static_cast<int>(avgs[0]->color.g) << static_cast<int>(avgs[0]->color.b)
         << std::dec << " " << (y * 20 * 2 - 10) << " 0 " << (-x * 20 * 2) <<      " 1 0 0 0 1 0 -0 0 1 3024.dat\n";
  ldFile << "1 0x2" << std::hex << static_cast<int>(avgs[1]->color.r) << static_cast<int>(avgs[1]->color.g) << static_cast<int>(avgs[1]->color.b)
         << std::dec << " " << (y * 20 * 2 - 10) << " 0 " << (-x * 20 * 2 + 20) << " 1 0 0 0 1 0 -0 0 1 3024.dat\n";
  ldFile << "1 0x2" << std::hex << static_cast<int>(avgs[2]->color.r) << static_cast<int>(avgs[2]->color.g) << static_cast<int>(avgs[2]->color.b)
         << std::dec << " " << (y * 20 * 2 + 10) << " 0 " << (-x * 20 * 2) <<      " 1 0 0 0 1 0 -0 0 1 3024.dat\n";
  ldFile << "1 0x2" << std::hex << static_cast<int>(avgs[3]->color.r) << static_cast<int>(avgs[3]->color.g) << static_cast<int>(avgs[3]->color.b)
         << std::dec << " " << (y * 20 * 2 + 10) << " 0 " << (-x * 20 * 2 + 20) << " 1 0 0 0 1 0 -0 0 1 3024.dat\n";
}

void repaint(){
  state.image = state.original.clone();
  Mat& image = state.image;
  auto mc = circleMask(state.circlesLongSide, max(image.rows, image.cols));
  sharpen(image);
  ofstream ldFile;
  if (state.ldrawFileName.size() && state.writeLDrawFile)
    ldFile = ofstream(state.ldrawFileName);
  map<string,vector<uint64_t>> partCounts;
  for(auto& spec: state.palette.availableColors)
    for(uint32_t part = 0; part < mc.nrOfParts; ++part)
      partCounts[spec.name].push_back(0);
  if(state.showMosaic){
    groupByMask(image, mc, 6, state.palette, [&ldFile, &mc, &partCounts](int x, int y, Palette::ColorSpec* avgs[]){
        if (state.ldrawFileName.size() && state.writeLDrawFile)
          writeLDrawFile(x,y,avgs,ldFile);
        if(state.partList.size() && state.writePartList){
          for(uint32_t group = 0; group < mc.nrGroups; ++group){
            auto part = mc.groupIdToPart[group];
            auto color = avgs[part];
            partCounts[color->name][part] += 1;
          }
        }
      });
    if(state.partList.size() && state.writePartList){
      ofstream partList(state.partList);
      for(auto it : partCounts){
        partList << it.first;
        for(auto part : it.second){
          partList << "," << part;
        }
        partList << "\n";
      }
    }
  }
  Mat disp;
  cvtColor(image, disp, COLOR_Lab2BGR);
  imshow("Display Image", disp);
}

void on_trackbar( int, void* ){
  repaint();
}

Mat resizeToBeDivisible(Mat image)
/// resize image so that it's long side is a multiple of state.circlesLongSide
{
  // resize so that the long side of the image is exactly
  // divisible by state.circlesLongSide
  uint64_t rows;
  uint64_t cols;
  if(image.rows < image.cols){
    auto rest = (image.rows % state.circlesLongSide);
    rows = image.rows - rest;
    cols = image.cols * rows / image.rows;
  }else{
    auto rest = (image.cols % state.circlesLongSide);
    cols = image.cols - rest;
    rows = image.rows * cols / image.cols;
  }
  Mat result;
  resize(image, result, Size(cols, rows));
  return result;
}

int main(int argc, char* argv[]){
  if (argc < 3){
    std::cout << "usage: " << argv[0] << " <paletteFile> <inputImage> [<fileName>]\n";
    return -1;
  }
  state.palette = Palette(argv[1]);
  Mat original = imread(argv[2], 1);
  if (!original.data){
    printf("No image data \n");
    return -1;
  }

  {
    Mat rgbFloat;
    original = resizeToBeDivisible(original);
    original.convertTo(rgbFloat, CV_32FC3, 1./255);
    cvtColor(rgbFloat, state.original, COLOR_BGR2Lab);
  }

  if(argc>=4){
    state.ldrawFileName = string(argv[3]) + ".ldr";
    state.partList = string(argv[3]) + ".csv";
  }

  namedWindow("Display Image",CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
  namedWindow("Toolbar",CV_WINDOW_NORMAL | CV_WINDOW_FREERATIO | CV_GUI_EXPANDED);
  createTrackbar( "Amount", "Toolbar", &state.amount ,1000, on_trackbar);
  createTrackbar( "Sigma", "Toolbar", &state.sigma ,1000, on_trackbar);
  createTrackbar( "BG Threshold", "Toolbar", &state.threshold ,1000, on_trackbar);
  createTrackbar( "Luminance importance", "Toolbar", &state.luminanceFactor ,1000, on_trackbar);
  repaint();
  int pressedKey = 0;
  while(pressedKey != 27 /*Esc*/ && pressedKey != 113 /*q*/){
    pressedKey = waitKey(0);
    state.writeLDrawFile = pressedKey == 51 /* 3 */;
    state.writePartList = pressedKey == 112 /* p */;
    state.palette = Palette(argv[1]);
    if(pressedKey==109)
      state.showMosaic = !state.showMosaic;
    repaint();
    state.writeLDrawFile = false;
    state.writePartList = false;
  }
  if(argc >=4){
    Mat write;
    {
      Mat rgb;
      cvtColor(state.image, rgb, COLOR_Lab2BGR);
      rgb.convertTo(write, CV_8UC3, 255);
    }
    imwrite(string(argv[3])+".jpg",write);
  }
  return 0;
}
