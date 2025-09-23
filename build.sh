#!/bin/sh

print_help()
{
    cat <<EOF
Usage: cmake.sh [options]
Options:
  -c, --clean   Clean CMake files instead of running CMake
  -d, --debug   Run CMake in debug mode
  -s, --subdir  Ask CMake to build files in the build/ folder

EOF
}

# Variables
CLEAN=0
DEBUG=0
SUBDIR=0

# Parse options
while true; do
	case "$1" in
		-c|--clean) CLEAN=1 ; shift ;;
		-d|--debug) DEBUG=1 ; shift ;;
		-s|--subdir) SUBDIR=1 ; shift ;;
        "") break ;;
		-h|--help) print_help ; exit 0 ;;
		*) print_help ; exit 1 ;;
	esac
done

# Select build directory for CMake
if [ $SUBDIR -eq 1 ] ; then
	BUILD_DIRECTORY="./build"
else
	BUILD_DIRECTORY="."
fi

if [ $CLEAN -eq 1 ] ; then

	# -- Clean CMake mode

	echo -n "Cleaning CMake files..."
	cd $BUILD_DIRECTORY
	rm -rf CMakeFiles/ CMakeCache.txt cmake_install.cmake Makefile build main test
	echo " Done."

else

	# -- Build CMake mode

	# Select build type for CMake
	if [ $DEBUG -eq 1 ]; then
		BUILD_TYPE="Debug"
	else
		BUILD_TYPE="Release"
	fi

	# Compile with CMake
	cmake -B $BUILD_DIRECTORY -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" .
	if [ $? -eq 0 ] && [ $BUILD_DIRECTORY != "." ]; then
		echo "(i) Please 'cd $BUILD_DIRECTORY' to make and run the program"
	fi

fi