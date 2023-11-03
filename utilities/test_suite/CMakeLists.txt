################################################################################
#
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
cmake_minimum_required (VERSION 3.5)

set(Python3_FIND_VIRTUALENV FIRST)
find_package(Python3 QUIET)

if(Python3_FOUND)

  add_test(NAME rpp_qa_tests_host_all 
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/utilities/test_suite/HOST/runTests.py 
        --qa_mode 1 --batch_size 3
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      )
  add_test(NAME rpp_qa_tests_host_brightness 
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/utilities/test_suite/HOST/runTests.py 
      --qa_mode 1 --batch_size 3 --case_list 0
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
  add_test(NAME rpp_qa_tests_host_gamma_correction 
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/utilities/test_suite/HOST/runTests.py 
    --qa_mode 1 --batch_size 3 --case_list 1
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  if( "${BACKEND}" STREQUAL "HIP")
    add_test(NAME rpp_qa_tests_hip 
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/utilities/test_suite/HIP/runTests.py 
          --qa_mode 1 --batch_size 3
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
  elseif( "${BACKEND}" STREQUAL "OCL")
    # TBD: Add OCL Tests
  endif()

else()
  message("-- ${Yellow}Test Warning: Python3 must be installed to run RPP test_suite successfully!${ColourReset}")
endif(Python3_FOUND)