# ComputerVision_BlackJack
Blackjack card recognition (C++ and OpenCV)


Rrequirements

 - CMake
 - Git
 - A C++ compiler toolchain (like gcc)


Install OpenCV:

```console
git clone --depth 1 https://github.com/opencv/opencv.git
cd opencv
mkdir build
cd build

cmake ../ -DCMAKE_BUILD_TYPE="Release"
make -j 2

make install
```


How to build 'opencv_blackjack':

```console
$ mkdir build
$ cd build
$ cmake ../ -DOpenCV_DIR="<insert the path of your opencv/build directory>"
$ make
```


How to run 'opencv_blackjack':

```console
$ cd build
$ make run
```

or, alternatively:

```console
$ make install
$ cd dist/bin
```

and run the generated executable


Federico Marcuzzi, 2020


