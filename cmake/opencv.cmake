include(cmake/color.cmake)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

if(OpenCV_FOUND)
    message(STATUS "${BoldWhite}Adding OpenCV to LIBRARIES${ColorReset}")
    list(APPEND LIBRARIES ${OpenCV_LIBS})
else ()
    message(STATUS "${BoldRed}OpenCV not found${ColorReset}")
endif()