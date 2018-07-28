#ifndef VULKAN_ENGINE_VENGINE_H
#define VULKAN_ENGINE_VENGINE_H

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <array>
#include <set>
#include <list>
#include <unordered_map>

#include "vengine_common.h"
#include "vobject.h"
#include "vscene.h"

#include "vulkan/vulkan.hpp"

namespace ve {
    class VEngine : public ve::VObject
    {
    public:
        explicit VEngine();
        ~VEngine();

        virtual void Run();
        virtual void AddScene(const ve::VScene& scene);

        const VkDevice get_logic_device();
        void AddUniformBufferAndMemory(const VkBuffer& uniform_buffer, const VkDeviceMemory& uniform_buffer_memory);

    protected:
        virtual void Clear();

        void CreateTexture(const std::string& texture_path, VkImage& texture, VkDeviceMemory& texture_memory);
        void CreateTextureView(VkImageView& view, VkImage& texture);
        void CreateTextureSampler(VkSampler& sampler);

    private:
        VkDevice logic_device_;

        std::vector<VkBuffer> uniform_buffers_;
        std::vector<VkDeviceMemory> uniform_buffer_memorys_;


    public:
        void initWindow();
        void initVulkan();
        void mainLoop();
        void cleanupSwapChain();
        void cleanup();
        void recreateSwapChain();
        void createInstance();
        void createSurface();
        void pickPhysicalDevice();
        void createLogicalDevice();
        void createSwapChain();
        void createImageViews();
        void createRenderPass();
        void createDescriptorSetLayout();
        void createGraphicsPipeline();
        void createFramebuffers();
        void createCommandPool();

        void loadModel();
        void createVertexBuffer();
        void createIndexBuffer();
        void createUniformBuffer();
        void createDescriptorPool();
        void createDescriptorSet();
        void createCommandBuffers();

        void createSemaphores();
        void updateUniformBuffer();
        void camera_control();
        void drawFrame();

        void RecreateBufer();

        void SaveOutputColorTexture(const std::string& path);
        void SaveOutputDepthTexture(const std::string& path);

        void setupMultisampleTarget();

    public:
        struct UniformMatrixBufferObject {
            glm::mat4 view;
            glm::mat4 proj;
            glm::vec3 lightPos;
        };

        struct ConstantMatrixModel {
            glm::mat4 model;
        };

        struct QueueFamilyIndices {
            int graphicsFamily = -1;
            int presentFamily = -1;

            bool isComplete() {
                return graphicsFamily >= 0 && presentFamily >= 0;
            }
        };

        struct SwapChainSupportDetails {
            VkSurfaceCapabilitiesKHR capabilities;
            std::vector<VkSurfaceFormatKHR> formats;
            std::vector<VkPresentModeKHR> presentModes;
        };

        struct Vertex {
            glm::vec3 pos;
            glm::vec3 normal;
            glm::vec2 texCoord;

            static VkVertexInputBindingDescription getBindingDescription() {
                VkVertexInputBindingDescription bindingDescription = {};
                bindingDescription.binding = 0;
                bindingDescription.stride = sizeof(Vertex);
                bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

                return bindingDescription;
            }

            static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
                std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};

                attributeDescriptions[0].binding = 0;
                attributeDescriptions[0].location = 0;
                attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
                attributeDescriptions[0].offset = offsetof(Vertex, pos);

                attributeDescriptions[1].binding = 0;
                attributeDescriptions[1].location = 1;
                attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
                attributeDescriptions[1].offset = offsetof(Vertex, normal);

                attributeDescriptions[2].binding = 0;
                attributeDescriptions[2].location = 2;
                attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
                attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

                return attributeDescriptions;
            }

            bool operator==(const Vertex& other) const {
                return pos == other.pos && normal == other.normal && texCoord == other.texCoord;
            }
        };

    private:
        VkBool32 findDepthFormat(VkPhysicalDevice physicalDevice, VkFormat *depthFormat);
        bool hasStencilComponent(VkFormat format);
        VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);
        void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
		void createTestImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);

        void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
        void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
        void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
        VkCommandBuffer beginSingleTimeCommands();
        void endSingleTimeCommands(VkCommandBuffer commandBuffer);
        void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
        uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
        VkShaderModule createShaderModule(const std::vector<char>& code);
        VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
        VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> availablePresentModes);
        VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
        SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
        bool isDeviceSuitable(VkPhysicalDevice device);
        bool checkDeviceExtensionSupport(VkPhysicalDevice device);
        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
        std::vector<const char*> getRequiredExtensions();
        bool checkValidationLayerSupport();
        std::vector<char> readFile(const std::string& filename);

        glm::mat4 GetOrthoMatrix(float left, float right, float bottom, float top, float near, float far);
        uint8_t MapColor(float f);
        glm::vec3 ColorWheel(float normalizeHue);


    private:
        GLFWwindow* window;

        VkInstance instance;
        VkDebugReportCallbackEXT callback;
        VkSurfaceKHR surface;

        VkFormat depthFormat;

        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkDevice device;

        //VkQueue graphicsQueue;
        VkQueue presentQueue;

        VkSwapchainKHR swapChain;
        std::vector<VkImage> swapChainImages;
        VkFormat swapChainImageFormat;
        VkExtent2D swapChainExtent;
        std::vector<VkImageView> swapChainImageViews;
        std::vector<VkFramebuffer> swapChainFramebuffers;

        VkRenderPass renderPass;
        VkPipelineLayout pipelineLayout;
        VkPipeline graphicsPipeline;

        VkCommandPool commandPool;

		VkImage depthImage;
		VkDeviceMemory depthImageMemory;
		VkImageView depthImageView;

        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        VkBuffer vertexBuffer;
        VkDeviceMemory vertexBufferMemory;
        VkBuffer indexBuffer;
        VkDeviceMemory indexBufferMemory;

        VkBuffer uniformMatrixBuffer;
        VkDeviceMemory uniformMatrixBufferMemory;

        VkDescriptorSetLayout descriptorSetLayout;
        VkDescriptorPool descriptorPool;
        VkDescriptorSet descriptorSet;

        std::vector<VkCommandBuffer> commandBuffers;

        VkSemaphore imageAvailableSemaphore;
        VkSemaphore renderFinishedSemaphore;

        uint32_t imageIndex;

        struct {
            struct {
                VkImage image;
                VkImageView view;
                VkDeviceMemory memory;
            } color;
            struct {
                VkImage image;
                VkImageView view;
                VkDeviceMemory memory;
            } depth;
        } multisampleTarget;


    private:
        const std::vector<const char*> validationLayers = {
            "VK_LAYER_LUNARG_standard_validation"
        };

        const std::vector<const char*> deviceExtensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME
        };


    private:
        const int WIDTH = 800;
        const int HEIGHT = 600;

        // Initial position : on +Z
        glm::vec3 position = glm::vec3(0.01, 2, 10.01);
        // Initial horizontal angle : toward -Z
        float horizontalAngle = 3.14f;
        // Initial vertical angle : none
        float verticalAngle = 0.0f;
        // Initial Field of View
        float initialFoV = 45.0f;

        float speed = 10.0f; // 3 units / second
        float mouseSpeed = 0.005f;

        double last_xpos_ = 0.0f;
        double last_ypos_ = 0.0f;

        // light direction 
        glm::vec3 lightPos = glm::vec3(1.0f, 1.0f, 0.001f);

        glm::mat4 clip = glm::mat4{
            1.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.0,
            0.0, 0.0, 0.5, 1.0 };

        bool kFirstPress = true;

        const bool enableValidationLayers = true;

        // version 2
        const std::string MODEL_PATH = "D:/media/model/shadow.obj";

       
    };
}

#endif // !VULKAN_ENGINE_VENGINE_H

