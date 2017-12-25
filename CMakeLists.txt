cmake_minimum_required(VERSION 3.0)

project(vulkan_engine)

set(DEPS "D:/depend_library")

include_directories(${DEPS}/glfw/include 
${DEPS}/glm
${DEPS}
${DEPS}/vulkan/include
${DEPS}/boost_1_64_0
${DEPS}/stb
${DEPS}/tinyobjloader)

link_directories(${DEPS}/glfw/lib-vc2015
${DEPS}/glfw/lib
${DEPS}/vulkan/lib
)

file(GLOB SOURCE ./*.*)
add_executable(vulkan_engine ${SOURCE})

target_link_libraries(vulkan_engine 
vulkan-1
glfw3
)

 























